import json, numpy as np
B="/private/tmp/claude-501/-Users-cheng-Maxgent-maxgent-worktree-flywheel/a51db06a-c38b-46e1-bb67-0b38f77be762/scratchpad/"
D=json.load(open(B+"pose_signal.json")); fps=D["fps"]; recs=D["records"]
PARTS=[[0],[11,12],[23,24],[25,26],[27,28]]
def vis(lm_r,ids): return min(lm_r.get("vis_"+n,0.0) for n in ids) if False else None

def get(r,key,default=np.nan):
    return r.get(key,default) if r.get("present") else default

# reconstruct per-frame part y + visibilities + rope from records
def ser(k):
    a=np.array([get(r,k) for r in recs],float); v=~np.isnan(a); idx=np.arange(len(a))
    return np.interp(idx,idx[v],a[v]) if v.any() else np.zeros(len(a))
Y={ 'nose':ser('nose_y'),'sh':ser('sh_y'),'hip':ser('hip_y'),'kn':ser('kn_y'),'ank':ser('ank_y') }
V={ 'nose':np.array([get(r,'vis_nose',0.0) for r in recs]),
    'sh':np.array([get(r,'vis_sh',0.0) for r in recs]),
    'hip':np.array([get(r,'vis_hip',0.0) for r in recs]),
    'kn':np.array([get(r,'vis_kn',0.0) for r in recs]),
    'ank':np.array([get(r,'vis_ank',0.0) for r in recs]) }
rope=np.array([r.get("rope",0.0) for r in recs])
PKEYS=['nose','sh','hip','kn','ank']
tms=[int(round(i/fps*1000)) for i in range(len(recs))]

def export_A():
    A_FAST,A_SLOW,PROM_FLOOR,A_DEVA=0.70,0.04,0.006,0.03
    MINGAP,ROPE_GATE=200,0.0025
    GK,G_LO,G_HI,G_AMPLO,G_MAXGAP=2,0.55,2.2,0.30,850
    G_RETRO,G_MINPAUSE=500,350
    pF={k:None for k in PKEYS}; pS={k:None for k in PKEYS}; cDevA=0.02; cPrev=None; cState='up'; cVMin=0.0
    ropeLvl=0.0; lastJump=-9999
    gLast=0; gMedInt=0; gMedAmp=0; gRun=0; gCounting=False; gPausedUntil=0
    count=0; jumps=[]; breaks=[]
    def addJump(now,amp):
        nonlocal count
        count+=1; prev=jumps[-1]['t'] if jumps else None
        jumps.append({'n':count,'t':int(now),'amp':round(amp,4),'iv':(int(now)-prev if prev is not None else None)})
    def onBreak(now):
        nonlocal count,gPausedUntil
        removed=0
        while jumps and jumps[-1]['t']>now-G_RETRO: jumps.pop(); removed+=1
        if removed>0: count=max(0,count-removed)
        breaks.append({'t':int(now),'retro':removed,'pauseMs':G_MINPAUSE})
        gPausedUntil=now+G_MINPAUSE
    def onCandidate(now,amp):
        nonlocal gLast,gMedInt,gMedAmp,gRun,gCounting
        if gRun==0: gRun=1; gLast=now; gMedAmp=amp; gMedInt=0; return
        iv=now-gLast; gLast=now; broke=False
        if gMedInt==0:
            if iv<=G_MAXGAP: gMedInt=iv; gMedAmp+=0.3*(amp-gMedAmp); gRun=2
            else: gRun=1; gMedAmp=amp; broke=True
        else:
            inTempo=(G_LO*gMedInt<=iv<=G_HI*gMedInt) and iv<=G_MAXGAP and amp>=G_AMPLO*gMedAmp
            if inTempo: gRun+=1; gMedInt+=0.3*(iv-gMedInt); gMedAmp+=0.3*(amp-gMedAmp)
            else: gRun=1; gMedInt=0; gMedAmp=amp; broke=True
        if broke:
            if gCounting: onBreak(now)
            gCounting=False; return
        if (not gCounting) and gRun>=GK and now>=gPausedUntil: gCounting=True
        if gCounting: addJump(now,amp)
    for i in range(len(recs)):
        now=tms[i]
        ropeLvl+=0.15*(rope[i]-ropeLvl); gate=ropeLvl>=ROPE_GATE
        num=den=0.0
        for k in PKEYS:
            y=Y[k][i]; w=V[k][i]
            pF[k]=y if pF[k] is None else pF[k]+A_FAST*(y-pF[k])
            pS[k]=y if pS[k] is None else pS[k]+A_SLOW*(y-pS[k])
            if w>=0.3: num+=w*(pF[k]-pS[k]); den+=w
        if den<=0: continue
        c=num/den; cDevA+=A_DEVA*(abs(c)-cDevA)
        kp=max(0.30,1.00-5*0.08); prom=max(cDevA*kp,PROM_FLOOR)
        if cPrev is None: cPrev=c; cVMin=c; continue
        if c>cPrev:
            if cState=='down': cVMin=cPrev; cState='up'
        elif c<cPrev:
            if cState=='up':
                peak=cPrev
                if gate and peak-cVMin>=prom and now-lastJump>=MINGAP:
                    lastJump=now; onCandidate(now,peak-cVMin); cVMin=peak
                cState='down'
        cPrev=c
    return count,jumps,breaks

def export_B():
    B_WIN,B_LOOK,B_MINN,B_KPROM,B_FLOOR,B_REFR=5,2,10,0.5,0.015,200
    bBuf=[]; bBase=None; bDevA=0.02; bLast=-9999; count=0; jumps=[]
    for i in range(len(recs)):
        now=tms[i]
        if V['ank'][i]<=0.35: continue
        v=Y['ank'][i]
        bBase=v if bBase is None else bBase+0.04*(v-bBase)
        dv=v-bBase; bDevA=bDevA+0.03*(abs(dv)-bDevA)
        bBuf.append((now,v))
        if len(bBuf)>B_MINN: bBuf.pop(0)
        if len(bBuf)>=B_WIN:
            ct,cv_=bBuf[len(bBuf)-1-B_LOOK]
            w=[x[1] for x in bBuf[-B_WIN:]]
            if cv_==max(w):
                prom=cv_-min(x[1] for x in bBuf); thr=max(B_FLOOR,bDevA*B_KPROM)
                if prom>=thr and (ct-bLast)>=B_REFR:
                    bLast=ct; count+=1; prev=jumps[-1]['t'] if jumps else None
                    jumps.append({'n':count,'t':int(ct),'amp':round(prom,4),'iv':(int(ct)-prev if prev is not None else None)})
    return count,jumps,[]

durationMs=tms[-1]
for algo,fn in [('A',export_A),('B',export_B)]:
    count,jumps,breaks=fn()
    interruptions=[{'tMs':j['t']-j['iv'],'gapMs':j['iv']} for j in jumps if j['iv'] and j['iv']>800]
    data={'app':'skip-count','exportedAt':'(offline from IMG_6258.MOV)','source':'IMG_6258.MOV',
          'algo':algo,'mode':'free','durationMs':durationMs,'count':count,
          'avgCadencePerMin':round(count/(durationMs/60000),1),
          'ampUnit':'normalized frame height · '+('ankle' if algo=='B' else 'body-consensus'),
          'jumps':jumps,'breaks':breaks,'interruptions':interruptions}
    out=f"/Users/cheng/Downloads/skip-video-{algo}.json"
    json.dump(data,open(out,'w'),ensure_ascii=False,indent=2)
    amps=[j['amp'] for j in jumps]
    print(f"[{algo}] count={count}  avg={data['avgCadencePerMin']}/min  breaks={len(breaks)}  interruptions={len(interruptions)}  amp[min/med/max]={min(amps):.3f}/{np.median(amps):.3f}/{max(amps):.3f}  -> {out}")
