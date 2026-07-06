import json, numpy as np
B="/private/tmp/claude-501/-Users-cheng-Maxgent-maxgent-worktree-flywheel/a51db06a-c38b-46e1-bb67-0b38f77be762/scratchpad/"
def load(fn):
    D=json.load(open(B+fn)); fps=D["fps"]; recs=D["records"]
    def ser(k):
        a=np.array([r.get(k,np.nan) if r["present"] else np.nan for r in recs],float)
        v=~np.isnan(a); idx=np.arange(len(a)); return np.interp(idx,idx[v],a[v]) if v.any() else np.zeros(len(a))
    Y={k:ser(k+"_y") for k in ["ank"]}
    Vs={"ank":np.array([(r.get("vis_ank",0.0) if r["present"] else 0.0) for r in recs])}
    t=np.array([i/fps*1000 for i in range(len(recs))]); rope=np.array([r.get("rope",0.0) for r in recs])
    return fps,recs,Y,Vs,t,rope

# Combined C: B ankle candidate generation (+ double-bounce guard) -> A rhythm-stability gate
def runC(fn, TRIPGATE=True):
    fps,recs,Y,Vs,t,rope=load(fn)
    B_WIN,B_LOOK,B_MINN,B_KPROM,B_FLOOR,B_REFR,DB_FRAC=5,2,10,0.5,0.015,200,0.5
    GK,LO,HI,AMPLO,MAXG=2,0.55,2.2,0.30,850; RETRO,MINPAUSE=500,350
    RG=0.0025
    bBuf=[]; bBase=None; bDevA=0.02; bLast=0; bIvs=[]
    ropeLvl=0; PAUSE=1000; bLast_c=[0,False]
    gLast=0;gMedInt=0;gMedAmp=0;gRun=0;gC=False;gPU=0;count=0;committed=[]
    def onC(now,amp):
        nonlocal gLast,gMedInt,gMedAmp,gRun,gC,gPU,count,bLast_c
        if TRIPGATE=='gentle':
            # only skip the single jump that resumes after a clear long pause (a trip stop)
            iv=now-bLast_c[0]; bLast_c[0]=now
            if bLast_c[1] and iv>PAUSE:   # long gap = pause/trip → don't count the resume beat
                bLast_c[1]=True; return
            bLast_c[1]=True; count+=1; committed.append(now); return
        if not TRIPGATE:
            count+=1; committed.append(now); return
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
        now=t[i]; ropeLvl+=0.15*(rope[i]-ropeLvl)   # (ankle path ignores rope gate)
        if Vs["ank"][i]<=0.35: continue
        v=Y["ank"][i]
        bBase=v if bBase is None else bBase+0.04*(v-bBase)
        dv=v-bBase; bDevA=bDevA+0.03*(abs(dv)-bDevA)
        bBuf.append((now,v))
        if len(bBuf)>B_MINN: bBuf.pop(0)
        if len(bBuf)>=B_WIN:
            ct,cv=bBuf[len(bBuf)-1-B_LOOK]; w=[x[1] for x in bBuf[-B_WIN:]]
            if cv==max(w):
                prom=cv-min(x[1] for x in bBuf); thr=max(B_FLOOR,bDevA*B_KPROM)
                if prom>=thr and (ct-bLast)>=B_REFR:
                    dbl=False
                    if len(bIvs)>=3:
                        med=np.median(bIvs[-6:])
                        if (ct-bLast)<DB_FRAC*med: dbl=True
                    if not dbl:
                        if bLast>0: bIvs.append(ct-bLast); bIvs=bIvs[-8:]
                        bLast=ct; onC(ct,prom)
    return count

for fn,tgt in [("pose_signal.json",152),("pose_6271.json",43)]:
    print(f"{fn} (true≈{tgt}):  B+DB={runC(fn,TRIPGATE=False)}   +gentle-pause={runC(fn,TRIPGATE='gentle')}   +full-gate={runC(fn,TRIPGATE=True)}")
