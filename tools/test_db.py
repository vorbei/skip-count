import json, numpy as np
B="/private/tmp/claude-501/-Users-cheng-Maxgent-maxgent-worktree-flywheel/a51db06a-c38b-46e1-bb67-0b38f77be762/scratchpad/"

def load(fn):
    D=json.load(open(B+fn)); fps=D["fps"]; recs=D["records"]
    def ser(k):
        a=np.array([r.get(k,np.nan) if r["present"] else np.nan for r in recs],float)
        v=~np.isnan(a); idx=np.arange(len(a)); return np.interp(idx,idx[v],a[v]) if v.any() else np.zeros(len(a))
    Y={k:ser(k+"_y") for k in ["nose","sh","hip","kn","ank"]}
    Vs={k:np.array([(r.get("vis_"+k,0.0) if r["present"] else 0.0) for r in recs]) for k in ["nose","sh","hip","kn","ank"]}
    t=np.array([i/fps*1000 for i in range(len(recs))])
    return fps,recs,Y,Vs,t

# ---- Version B (ankle) with optional double-bounce guard ----
def runB(fn, GUARD=False, LO_FRAC=0.6, AMP_FRAC=0.75, MODE="both"):
    fps,recs,Y,Vs,t=load(fn)
    B_WIN,B_LOOK,B_MINN,B_KPROM,B_FLOOR,B_REFR=5,2,10,0.5,0.015,200
    buf=[]; base=None; devA=0.02; last=-9999; count=0
    ivs=[]; amps=[]           # accepted intervals/amps for the guard
    for i in range(len(recs)):
        now=t[i]
        if Vs["ank"][i]<=0.35: continue
        v=(Y["ank"][i]);
        base=v if base is None else base+0.04*(v-base)
        dv=v-base; devA=devA+0.03*(abs(dv)-devA)
        buf.append((now,v))
        if len(buf)>B_MINN: buf.pop(0)
        if len(buf)>=B_WIN:
            ct,cv=buf[len(buf)-1-B_LOOK]; w=[x[1] for x in buf[-B_WIN:]]
            if cv==max(w):
                prom=cv-min(x[1] for x in buf); thr=max(B_FLOOR,devA*B_KPROM)
                if prom>=thr and (ct-last)>=B_REFR:
                    if GUARD and len(ivs)>=3:
                        medI=np.median(ivs[-6:]); medA=np.median(amps[-6:])
                        early = (ct-last) < LO_FRAC*medI
                        small = prom < AMP_FRAC*medA
                        skip = (early and small) if MODE=="both" else (early if MODE=="iv" else small)
                        if skip: continue    # intermediate double-bounce → skip
                    ivs.append(ct-last); amps.append(prom); last=ct; count+=1
    return count

# ---- Version A (body consensus) ----
def runA(fn):
    fps,recs,Y,Vs,t=load(fn)
    PK=["nose","sh","hip","kn","ank"]
    A_FAST,A_SLOW,PROM_FLOOR,ADEVA=0.70,0.04,0.006,0.03
    MINGAP,RG=200,0.0025; GK,LO,HI,AMPLO,MAXG=2,0.55,2.2,0.30,850; RETRO,MINPAUSE=500,350
    rope=np.array([r.get("rope",0.0) for r in recs])
    pF={k:None for k in PK}; pS={k:None for k in PK}; cDevA=0.02; cPrev=None; cState='up'; cVMin=0
    ropeLvl=0; last=-9; gLast=0;gMedInt=0;gMedAmp=0;gRun=0;gC=False;gPU=0;count=0;committed=[]
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

print("plain: 6258(152)=%d 6271(43)=%d"%(runB("pose_signal.json"),runB("pose_6271.json")))
print("\nMODE=iv (interval-only) sweep LO_FRAC   [6258 must stay ~149, 6271 target 43]")
for lo in [0.5,0.55,0.6,0.65,0.7]:
    a=runB("pose_signal.json",GUARD=True,LO_FRAC=lo,MODE="iv")
    b=runB("pose_6271.json",GUARD=True,LO_FRAC=lo,MODE="iv")
    print(f"  LO={lo}: 6258={a}  6271={b}")
print("\nMODE=both (iv AND small-amp)")
for lo in [0.55,0.62,0.7]:
    for am in [0.6,0.75,0.9]:
        a=runB("pose_signal.json",GUARD=True,LO_FRAC=lo,AMP_FRAC=am,MODE="both")
        b=runB("pose_6271.json",GUARD=True,LO_FRAC=lo,AMP_FRAC=am,MODE="both")
        print(f"  LO={lo} AMP={am}: 6258={a}  6271={b}")
