import cv2, os, numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE="/private/tmp/claude-501/-Users-cheng-Maxgent-maxgent-worktree-flywheel/a51db06a-c38b-46e1-bb67-0b38f77be762/scratchpad/"
VIDEO=os.environ.get("VIDEO","/Users/cheng/Downloads/IMG_6258.MOV")
MODEL=BASE+"pose_landmarker_lite.task"
OUT=os.environ.get("OUT",BASE+"debug.mp4")

OH=960                              # output height (portrait, aspect-correct)
MW,MH=80,60                         # rope motion-diff resolution
PARTS=[[0],[11,12],[23,24],[25,26],[27,28]]
PVIS=[[0],[11,12],[23,24],[25,26],[27,28]]
LINKS=[(11,12),(11,23),(12,24),(23,24),(11,13),(13,15),(12,14),(14,16),
       (15,17),(15,19),(16,18),(16,20),(23,25),(25,27),(24,26),(26,28),
       (27,29),(29,31),(27,31),(28,30),(30,32),(28,32)]

base=python.BaseOptions(model_asset_path=MODEL)
opts=vision.PoseLandmarkerOptions(base_options=base,running_mode=vision.RunningMode.VIDEO,
     num_poses=1,min_pose_detection_confidence=0.5,min_pose_presence_confidence=0.5,min_tracking_confidence=0.5)
lmk=vision.PoseLandmarker.create_from_options(opts)

cap=cv2.VideoCapture(VIDEO); fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ok,f=cap.read(); H0,W0=f.shape[:2]; cap.set(cv2.CAP_PROP_POS_FRAMES,0)
OW=int(round(W0/H0*OH))
print(f"src {W0}x{H0} fps={fps} frames={total} -> out {OW}x{OH}",flush=True)
vw=cv2.VideoWriter(OUT,cv2.VideoWriter_fourcc(*"mp4v"),fps,(OW,OH))

# ---- detector state (exact port of the browser) ----
A_FAST,A_SLOW,PROM_FLOOR=0.70,0.04,0.006
A_DEVA=0.03
MINGAP,ROPE_GATE=0.200,0.0025
GK,G_LO,G_HI,G_AMPLO,G_MAXGAP=2,0.55,2.2,0.30,0.85
G_RETRO,G_MINPAUSE=0.5,0.35
pF=[None]*5; pS=[None]*5; cDevA=0.02; cPrev=None; cState='up'; cVMin=0.0
ropeLvl=0.0; lastJump=-9.0
gLast=0.0; gMedInt=0.0; gMedAmp=0.0; gRun=0; gCounting=False; gPausedUntil=0.0
count=0; prev_gray=None; last_commit_t=-9; last_break_t=-9; last_retro=0; committed_times=[]

import os
ALGO=os.environ.get("ALGO","A")
# ---- Version B (CSDN ankle-based): pure ankle y, lookahead local-max + prominence ----
B_WIN,B_LOOK,B_MINN,B_KPROM,B_FLOOR,B_REFR,DB_FRAC=5,2,10,0.5,0.015,0.20,0.5
bBuf=[]; bBase=None; bDevA=0.02; bLast=-9.0; bIvs=[]
def detectB(vy, now):
    global bBuf,bBase,bDevA,bLast,count,last_commit_t,bIvs
    import numpy as _np
    bBase = vy if bBase is None else bBase+0.04*(vy-bBase)
    dv=vy-bBase; bDevA=bDevA+0.03*(abs(dv)-bDevA)
    bBuf.append((now,vy))
    if len(bBuf)>B_MINN: bBuf.pop(0)
    if len(bBuf)>=B_WIN:
        ci=len(bBuf)-1-B_LOOK; ct,cv_=bBuf[ci]
        w=[x[1] for x in bBuf[-B_WIN:]]
        if cv_==max(w):
            prom=cv_-min(x[1] for x in bBuf); thr=max(B_FLOOR,bDevA*B_KPROM)
            if prom>=thr and (ct-bLast)>=B_REFR:
                dbl=False
                if len(bIvs)>=3:
                    med=_np.median(bIvs[-6:])
                    if (ct-bLast) < DB_FRAC*med: dbl=True
                if not dbl:
                    if bLast>0: bIvs.append(ct-bLast); bIvs=bIvs[-8:]
                    bLast=ct; count+=1; last_commit_t=ct; return 1
    return 0

def onCandidate(now,amp):
    global gRun,gLast,gMedAmp,gMedInt,gCounting,gPausedUntil,count
    global last_commit_t,last_break_t,last_retro,committed_times
    if gRun==0: gRun=1; gLast=now; gMedAmp=amp; gMedInt=0.0; return 0
    iv=now-gLast; gLast=now; broke=False
    if gMedInt==0.0:
        if iv<=G_MAXGAP: gMedInt=iv; gMedAmp+=0.3*(amp-gMedAmp); gRun=2
        else: gRun=1; gMedAmp=amp; broke=True
    else:
        inTempo=(G_LO*gMedInt<=iv<=G_HI*gMedInt) and iv<=G_MAXGAP and amp>=G_AMPLO*gMedAmp
        if inTempo: gRun+=1; gMedInt+=0.3*(iv-gMedInt); gMedAmp+=0.3*(amp-gMedAmp)
        else: gRun=1; gMedInt=0.0; gMedAmp=amp; broke=True
    if broke:
        if gCounting:
            removed=0
            while committed_times and committed_times[-1]>now-G_RETRO:
                committed_times.pop(); removed+=1
            if removed>0: count=max(0,count-removed); last_retro=removed
            gPausedUntil=now+G_MINPAUSE; last_break_t=now
        gCounting=False
        return 0
    if (not gCounting) and gRun>=GK and now>=gPausedUntil: gCounting=True
    if gCounting: count+=1; committed_times.append(now); last_commit_t=now; return 1
    return 0

idx=0
while True:
    ok,frame=cap.read()
    if not ok: break
    now=idx/fps
    img=cv2.resize(frame,(OW,OH),interpolation=cv2.INTER_AREA)
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    res=lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),int(now*1000))
    # rope motion
    g=cv2.cvtColor(cv2.resize(img,(MW,MH),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff=None
    if prev_gray is not None:
        d=np.abs(g-prev_gray); d[d<=12]=0; diff=d
    prev_gray=g

    committed=0; gate_open=False; status="—"
    if res.pose_landmarks:
        lm=res.pose_landmarks[0]
        def vis(i): return getattr(lm[i],"visibility",1.0)
        if ALGO=='B':
            # Version B: pure ankle vertical, lookahead local-max + prominence, no gates
            if min(vis(27),vis(28))>0.35:
                committed=detectB((lm[27].y+lm[28].y)/2, now)
            status="ANKLE-B"
        else:
            # rope gate
            rE=0.0
            if min(vis(27),vis(28))>0.35 and diff is not None:
                ax=(lm[27].x+lm[28].x)/2; ay=(lm[27].y+lm[28].y)/2
                gx0,gx1=max(0,int((ax-0.17)*MW)),min(MW,int(np.ceil((ax+0.17)*MW)))
                gy0,gy1=max(0,int((ay-0.05)*MH)),min(MH,int(np.ceil((ay+0.17)*MH)))
                if gx1>gx0 and gy1>gy0: rE=float(diff[gy0:gy1,gx0:gx1].mean()/255.0)
            ropeLvl+=0.15*(rE-ropeLvl); gate_open=ropeLvl>=ROPE_GATE
            # consensus
            num=den=0.0
            for k,ids in enumerate(PARTS):
                y=sum(lm[i].y for i in ids)/len(ids)
                pF[k]=y if pF[k] is None else pF[k]+A_FAST*(y-pF[k])
                pS[k]=y if pS[k] is None else pS[k]+A_SLOW*(y-pS[k])
                w=min(vis(i) for i in PVIS[k])
                if w>=0.3: num+=w*(pF[k]-pS[k]); den+=w
            if den>0:
                c=num/den; cDevA+=A_DEVA*(abs(c)-cDevA)
                kp=max(0.30,1.00-5*0.08); prom=max(cDevA*kp,PROM_FLOOR)
                if cPrev is None: cPrev=c; cVMin=c
                else:
                    if c>cPrev:
                        if cState=='down': cVMin=cPrev; cState='up'
                    elif c<cPrev:
                        if cState=='up':
                            peak=cPrev
                            if gate_open and peak-cVMin>=prom and now-lastJump>=MINGAP:
                                lastJump=now; committed=onCandidate(now,peak-cVMin); cVMin=peak
                            cState='down'
                    cPrev=c
            status = "COUNTING" if gCounting else ("resume %d/%d"%(gRun,GK) if gRun>0 else "paused")
        # draw skeleton
        def P(i): return (int(lm[i].x*OW),int(lm[i].y*OH))
        for a,b in LINKS:
            if vis(a)>0.3 and vis(b)>0.3: cv2.line(img,P(a),P(b),(132,220,61),2,cv2.LINE_AA)
        for i in range(33):
            if vis(i)>0.3: cv2.circle(img,P(i),4,(61,220,132),-1,cv2.LINE_AA)
        # ankle ROI box
        if min(vis(27),vis(28))>0.35:
            ax=(lm[27].x+lm[28].x)/2; ay=(lm[27].y+lm[28].y)/2
            cv2.rectangle(img,(int((ax-0.17)*OW),int((ay-0.05)*OH)),(int((ax+0.17)*OW),int((ay+0.17)*OH)),
                          (0,200,255) if gate_open else (120,120,120),2)

    # overlays
    flash = (now-last_commit_t)<0.25
    brk = (now-last_break_t)<0.8
    cv2.rectangle(img,(0,0),(OW,86),(20,20,28),-1)
    cv2.putText(img,f"COUNT {count}",(12,60),cv2.FONT_HERSHEY_SIMPLEX,1.7,
                (61,220,132) if flash else (255,255,255),4,cv2.LINE_AA)
    cv2.putText(img,f"frame {idx}  t={now:5.2f}s  [ver {ALGO}]",(OW-360,30),cv2.FONT_HERSHEY_SIMPLEX,0.66,(200,210,230),2,cv2.LINE_AA)
    _line = ("ANKLE-B (CSDN)" if ALGO=='B' else f"rope {'OPEN' if gate_open else 'shut'}  {status}")
    cv2.putText(img,_line,(OW-360,64),cv2.FONT_HERSHEY_SIMPLEX,0.62,(0,200,255) if (ALGO=='B' or gate_open) else (150,150,150),2,cv2.LINE_AA)
    if committed>0:
        cv2.putText(img,"+1",(OW-90,150),cv2.FONT_HERSHEY_SIMPLEX,1.4,(61,220,132),4,cv2.LINE_AA)
    if brk:
        cv2.putText(img,f"TRIP  -{last_retro}",(int(OW*0.16),int(OH*0.46)),cv2.FONT_HERSHEY_SIMPLEX,1.2,(60,60,255),4,cv2.LINE_AA)
        cv2.putText(img,"paused",(int(OW*0.30),int(OH*0.53)),cv2.FONT_HERSHEY_SIMPLEX,1.0,(60,60,255),3,cv2.LINE_AA)
    elif not gCounting and res.pose_landmarks:
        cv2.putText(img,"(paused - not stable)",(int(OW*0.16),int(OH*0.5)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(80,140,255),3,cv2.LINE_AA)
    vw.write(img)
    idx+=1
    if idx%200==0: print(f"  {idx}/{total} count={count}",flush=True)

cap.release(); vw.release()
print(f"DONE count={count}  -> {OUT}",flush=True)
