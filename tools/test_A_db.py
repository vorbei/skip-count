import json, numpy as np
B="/private/tmp/claude-501/-Users-cheng-Maxgent-maxgent-worktree-flywheel/a51db06a-c38b-46e1-bb67-0b38f77be762/scratchpad/"
def load(fn):
    D=json.load(open(B+fn)); fps=D["fps"]; recs=D["records"]
    def ser(k):
        a=np.array([r.get(k,np.nan) if r["present"] else np.nan for r in recs],float)
        v=~np.isnan(a); idx=np.arange(len(a)); return np.interp(idx,idx[v],a[v]) if v.any() else np.zeros(len(a))
    Y={k:ser(k+"_y") for k in ["nose","sh","hip","kn","ank"]}
    Vs={k:np.array([(r.get("vis_"+k,0.0) if r["present"] else 0.0) for r in recs]) for k in ["nose","sh","hip","kn","ank"]}
    t=np.array([i/fps*1000 for i in range(len(recs))]); rope=np.array([r.get("rope",0.0) for r in recs])
    return fps,recs,Y,Vs,t,rope

def runA(fn, IVGUARD=0.0, MAJ=0.0):
    fps,recs,Y,Vs,t,rope=load(fn); PK=["nose","sh","hip","kn","ank"]
    A_FAST,A_SLOW,PROM_FLOOR,ADEVA=0.70,0.04,0.006,0.03
    MINGAP,RG=200,0.0025; GK,LO,HI,AMPLO,MAXG=2,0.55,2.2,0.30,850; RETRO,MINPAUSE=500,350
    pF={k:None for k in PK}; pS={k:None for k in PK}; cDevA=0.02; cPrev=None; cState='up'; cVMin=0
    ropeLvl=0; last=-9; gLast=0;gMedInt=0;gMedAmp=0;gRun=0;gC=False;gPU=0;count=0;committed=[]
    aLast=[0]; aIvs=[]; aProms=[]
    def do_count(now,amp):
        # apply double-bounce guards at count time
        if IVGUARD>0 and len(aIvs)>=3:
            s=sorted(aIvs[-6:]); med=s[len(s)//2]
            if (now-aLast[0]) < IVGUARD*med: return False
        if MAJ>0 and len(aProms)>=4:
            mx=max(aProms[-8:])
            if amp < MAJ*mx: return False
        if aLast[0]>0: aIvs.append(now-aLast[0])
        aProms.append(amp); aLast[0]=now
        return True
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
        if gC and do_count(now,amp): count+=1;committed.append(now)
    for i in range(len(recs)):
        now=t[i]; ropeLvl+=0.15*(rope[i]-ropeLvl);gate=ropeLvl>=RG
        num=den=0
        for k in PK:
            y=Y[k][i];w=Vs[k][i]
            pF[k]=y if pF[k] is None else pF[k]+A_FAST*(y-pF[k])
            pS[k]=y if pS[k] is None else pS[k]+A_SLOW*(y-pS[k])
            if w>=0.3:num+=w*(pF[k]-pS[k]);den+=w
        if den<=0:continue
        c=num/den;cDevA+=ADEVA*(abs(c)-cDevA);prom=max(cDevA*0.6,PROM_FLOOR)
        if cPrev is None:cPrev=c;cVMin=c;continue
        if c>cPrev:
            if cState=='down':cVMin=cPrev;cState='up'
        elif c<cPrev:
            if cState=='up':
                peak=cPrev
                if gate and peak-cVMin>=prom and now-last>=MINGAP:
                    last=now;onC(now,peak-cVMin);cVMin=peak
                cState='down'
        cPrev=c
    return count

print("A plain:        6258(152)=%d  6271(43)=%d"%(runA("pose_signal.json"),runA("pose_6271.json")))
print("\nA + interval-guard:")
for iv in [0.5,0.55,0.6,0.65]:
    print(f"  IV={iv}: 6258={runA('pose_signal.json',IVGUARD=iv)}  6271={runA('pose_6271.json',IVGUARD=iv)}")
print("\nA + major-peak (amp >= MAJ*recent-max):")
for mj in [0.35,0.45,0.55,0.65]:
    print(f"  MAJ={mj}: 6258={runA('pose_signal.json',MAJ=mj)}  6271={runA('pose_6271.json',MAJ=mj)}")
print("\nA + both (IV=0.5 + MAJ):")
for mj in [0.4,0.5,0.6]:
    print(f"  IV0.5 MAJ={mj}: 6258={runA('pose_signal.json',IVGUARD=0.5,MAJ=mj)}  6271={runA('pose_6271.json',IVGUARD=0.5,MAJ=mj)}")
