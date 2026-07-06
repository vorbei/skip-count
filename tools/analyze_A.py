import json, numpy as np
from scipy.signal import find_peaks
B="/private/tmp/claude-501/-Users-cheng-Maxgent-maxgent-worktree-flywheel/a51db06a-c38b-46e1-bb67-0b38f77be762/scratchpad/"
A=json.load(open("/Users/cheng/Downloads/skip-video-A.json"))
Bj=json.load(open("/Users/cheng/Downloads/skip-video-B.json"))
D=json.load(open(B+"pose_signal.json")); fps=D["fps"]; recs=D["records"]
def ser(k):
    a=np.array([r.get(k,np.nan) if r["present"] else np.nan for r in recs],float)
    v=~np.isnan(a); idx=np.arange(len(a)); return np.interp(idx,idx[v],a[v])
Y={k:ser(k+"_y") for k in ["nose","sh","hip","kn","ank"]}
Vs={k:np.array([(r.get("vis_"+k,0.0) if r["present"] else 0.0) for r in recs]) for k in ["nose","sh","hip","kn","ank"]}
rope=np.array([r.get("rope",0.0) for r in recs]); t=np.array([i/fps for i in range(len(recs))])
body=(Y["sh"]+Y["hip"])/2
d=body-np.convolve(body,np.ones(int(fps))/int(fps),mode='same')
gtpk=find_peaks(d,distance=int(0.24*fps),prominence=0.3*d.std())[0]; GT=t[gtpk]
At=np.array([j["t"]/1000 for j in A["jumps"]]); Bt=np.array([j["t"]/1000 for j in Bj["jumps"]])
brk=[b["t"]/1000 for b in A["breaks"]]

print(f"GT={len(GT)}  A={len(At)}  B={len(Bt)}")
def misses(ref, got, tol=0.28):
    return [x for x in ref if not len(got) or np.min(np.abs(got-x))>tol]
mA=misses(Bt,At); eA=misses(At,Bt)
print(f"\nA vs B:  A misses {len(mA)} jumps that B caught;  A has {len(eA)} extra not in B")
print("A-missed jump times (s):", [round(x,1) for x in mA])
print("A breaks at (s):", [round(x,1) for x in brk])
# attribute each A-miss: is it within 1.5s after a break (gate-suppressed) ?
for x in mA:
    nb=[b for b in brk if 0<=x-b<1.8]
    print(f"  miss t={x:5.1f}: {'gate-suppressed (break at %.1f)'%nb[0] if nb else 'below-threshold / not detected'}")

# ---- test improvements to A (weighted consensus + gate softening + lookahead) ----
PK=["nose","sh","hip","kn","ank"]
def runA(weights=None, floor=0.006, HI=2.2, RETRO=500, MINPAUSE=350, LOOK=0, wbody=True):
    W=weights or {k:1 for k in PK}
    A_FAST,A_SLOW,ADEVA=0.70,0.04,0.03; MINGAP,RG=0.200,0.0025
    GK,LO,AMPLO,MAXG=2,0.55,0.30,0.85
    pF={k:None for k in PK}; pS={k:None for k in PK}; cDevA=0.02
    ropeLvl=0; last=-9; gLast=0;gMedInt=0;gMedAmp=0;gRun=0;gC=False;gPU=0;count=0;committed=[]
    buf=[]
    def onC(now,amp):
        nonlocal gLast,gMedInt,gMedAmp,gRun,gC,gPU,count
        if gRun==0:gRun=1;gLast=now;gMedAmp=amp;gMedInt=0;return
        iv=now-gLast;gLast=now;broke=False
        if gMedInt==0:
            if iv<=MAXG:gMedInt=iv;gMedAmp+=0.3*(amp-gMedAmp);gRun=2
            else:gRun=1;gMedAmp=amp;broke=True
        else:
            ok=(LO*gMedInt<=iv<=HI*gMedInt) and iv<=MAXG and amp>=AMPLO*gMedAmp
            if ok:gRun+=1;gMedInt+=0.3*(iv-gMedInt);gMedAmp+=0.3*(amp-gMedAmp)
            else:gRun=1;gMedInt=0;gMedAmp=amp;broke=True
        if broke:
            if gC:
                r=0
                while committed and committed[-1]>now-RETRO/1000: committed.pop();r+=1
                count-=r; gPU=now+MINPAUSE/1000
            gC=False;return
        if (not gC) and gRun>=GK and now>=gPU:gC=True
        if gC:count+=1;committed.append(now)
    for i in range(len(recs)):
        now=t[i]; ropeLvl+=0.15*(rope[i]-ropeLvl);gate=ropeLvl>=RG
        num=den=0
        for k in PK:
            y=Y[k][i];w=Vs[k][i]
            pF[k]=y if pF[k] is None else pF[k]+A_FAST*(y-pF[k])
            pS[k]=y if pS[k] is None else pS[k]+A_SLOW*(y-pS[k])
            if w>=0.3:num+=w*W[k]*(pF[k]-pS[k]);den+=w*W[k]
        if den<=0:continue
        c=num/den;cDevA+=ADEVA*(abs(c)-cDevA)
        prom=max(cDevA*0.6,floor)
        buf.append((now,c))
        if len(buf)>12:buf.pop(0)
        # peak with optional lookahead
        if LOOK==0:
            if len(buf)>=2:
                if buf[-1][1]<buf[-2][1] and (len(buf)<3 or buf[-2][1]>=buf[-3][1]):
                    pass
        # simple online (LOOK=0) uses schmitt-like below; LOOK>0 windowed
        # -- reuse online local-max on c:
    # (we run a clean online detector separately for clarity)
    return None

# cleaner: online local-max detector on weighted-c with gate, returns count
def runA2(weights, floor=0.006, HI=2.2, RETRO=0.5, MINPAUSE=0.35, LOOK=0):
    W=weights; A_FAST,A_SLOW,ADEVA=0.70,0.04,0.03; MINGAP,RG=0.200,0.0025
    GK,LO,AMPLO,MAXG=2,0.55,0.30,0.85
    pF={k:None for k in PK}; pS={k:None for k in PK}; cDevA=0.02; ropeLvl=0
    prev=None;state='up';vMin=0;last=-9;gLast=0;gMedInt=0;gMedAmp=0;gRun=0;gC=False;gPU=0;count=0;committed=[]
    win=[]
    def onC(now,amp):
        nonlocal gLast,gMedInt,gMedAmp,gRun,gC,gPU,count
        if gRun==0:gRun=1;gLast=now;gMedAmp=amp;gMedInt=0;return
        iv=now-gLast;gLast=now;broke=False
        if gMedInt==0:
            if iv<=MAXG:gMedInt=iv;gMedAmp+=0.3*(amp-gMedAmp);gRun=2
            else:gRun=1;gMedAmp=amp;broke=True
        else:
            ok=(LO*gMedInt<=iv<=HI*gMedInt) and iv<=MAXG and amp>=AMPLO*gMedAmp
            if ok:gRun+=1;gMedInt+=0.3*(iv-gMedInt);gMedAmp+=0.3*(amp-gMedAmp)
            else:gRun=1;gMedInt=0;gMedAmp=amp;broke=True
        if broke:
            if gC:
                r=0
                while committed and committed[-1]>now-RETRO:committed.pop();r+=1
                count-=r;gPU=now+MINPAUSE
            gC=False;return
        if (not gC) and gRun>=GK and now>=gPU:gC=True
        if gC:count+=1;committed.append(now)
    for i in range(len(recs)):
        now=t[i];ropeLvl+=0.15*(rope[i]-ropeLvl);gate=ropeLvl>=RG
        num=den=0
        for k in PK:
            y=Y[k][i];w=Vs[k][i]
            pF[k]=y if pF[k] is None else pF[k]+A_FAST*(y-pF[k])
            pS[k]=y if pS[k] is None else pS[k]+A_SLOW*(y-pS[k])
            if w>=0.3:num+=w*W[k]*(pF[k]-pS[k]);den+=w*W[k]
        if den<=0:continue
        c=num/den;cDevA+=ADEVA*(abs(c)-cDevA);prom=max(cDevA*0.6,floor)
        if LOOK>0:
            win.append((now,c))
            if len(win)>10:win.pop(0)
            if len(win)>=5:
                ct,cv=win[len(win)-1-LOOK]
                w5=[x[1] for x in win[-5:]]
                if cv==max(w5):
                    p=cv-min(x[1] for x in win)
                    if gate and p>=prom and now-last>=MINGAP:
                        last=ct;onC(ct,p)
        else:
            if prev is None:prev=c;vMin=c;continue
            if c>prev:
                if state=='down':vMin=prev;state='up'
            elif c<prev:
                if state=='up':
                    peak=prev
                    if gate and peak-vMin>=prom and now-last>=MINGAP:
                        last=now;onC(now,peak-vMin)
                    state='down'
            prev=c
    return count

EQ={k:1 for k in PK}
LOWER={'nose':0.4,'sh':0.7,'hip':1.0,'kn':1.6,'ank':2.2}
print("\n== improvement tests (GT=%d, current A=146) =="%len(GT))
print(f"  A current (eq weights, gate on)                 : {runA2(EQ)}")
print(f"  + lower-body weighting                          : {runA2(LOWER)}")
print(f"  + soften gate (no retro, no min-pause)          : {runA2(EQ,RETRO=0,MINPAUSE=0)}")
print(f"  + lower-body + soften gate                      : {runA2(LOWER,RETRO=0,MINPAUSE=0)}")
print(f"  + lower-body + soften + lookahead(2)            : {runA2(LOWER,RETRO=0,MINPAUSE=0,LOOK=2)}")
print(f"  + lower-body + lookahead(2) + KEEP gate         : {runA2(LOWER,LOOK=2)}")
