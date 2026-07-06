import json, numpy as np
B="/private/tmp/claude-501/-Users-cheng-Maxgent-maxgent-worktree-flywheel/a51db06a-c38b-46e1-bb67-0b38f77be762/scratchpad/"
D=json.load(open(B+"pose_signal.json")); fps=D["fps"]; recs=D["records"]
t=np.array([r["t"]/1000.0 for r in recs])
def series(key):
    a=np.array([r.get(key,np.nan) if r["present"] else np.nan for r in recs],float)
    v=~np.isnan(a); idx=np.arange(len(a)); return np.interp(idx,idx[v],a[v])
bodyc=(series("hip_y")+series("sh_y"))/2
rope=np.array([r["rope"] for r in recs])

# rope level EMA distribution during real jumping
rl=0.0; rls=[]
for x in rope:
    rl=rl+0.15*(x-rl); rls.append(rl)
rls=np.array(rls)
print(f"rope EMA during jumping: min={rls.min():.4f} p10={np.percentile(rls,10):.4f} median={np.median(rls):.4f}")

# detector with rope GATE + absolute prominence floor
def run(sig, ts, rope_arr, sens=5, aFast=0.7, aSlow=0.04, floor=0.012, gate=0.004):
    kp=max(0.30, 1.00 - sens*0.08)
    bF=bS=None; devA=0.02; prev=None; state='up'; vMin=0.0
    rl=0.0; last=-9.0; warmed=False; cnt=0
    for i in range(len(sig)):
        now=ts[i]; p=sig[i]
        rl=rl+0.15*(rope_arr[i]-rl)                 # foot-motion gate level
        bF=p if bF is None else bF+aFast*(p-bF)
        bS=p if bS is None else bS+aSlow*(p-bS)
        dev=bF-bS; devA=devA+0.05*(abs(dev)-devA)
        prom=max(devA*kp, floor)
        if prev is None: prev=dev; vMin=dev; continue
        if dev>prev:
            if state=='down': vMin=prev; state='up'
        elif dev<prev:
            if state=='up':
                peak=prev
                gated = rl >= gate
                if gated and peak-vMin>=prom and now-last>=0.20:
                    dt=now-last; last=now
                    if warmed and dt<2.6: cnt+=1
                    else: warmed=True
                    vMin=peak
                state='down'
        prev=dev
    return cnt

print("\nREAL VIDEO (GT ~158) with gate=0.004, floor=0.012:")
for s in range(1,11): print(f"  sens={s:>2} -> {run(bodyc,t,rope,sens=s)}")

rng=np.random.default_rng(0)
resid=bodyc-np.convolve(bodyc,np.ones(3)/3,mode="same"); jit=resid.std()
N=1800; ts=np.arange(N)/fps
print(f"\njitter std={jit:.4f}")
print("STILL (flat+jitter), rope=0 (static scene -> no foot motion):")
for mult in [1,2,3]:
    still=0.55+rng.normal(0,jit*mult,N); rz=np.zeros(N)
    print(f"  {mult}x jitter, rope=0 : sens5={run(still,ts,rz,sens=5)} sens10={run(still,ts,rz,sens=10)}")
print("STILL but tiny fidget rope=0.002 (below gate):")
still=0.55+rng.normal(0,jit,N); rlow=np.full(N,0.002)
print(f"  sens5={run(still,ts,rlow,sens=5)} sens10={run(still,ts,rlow,sens=10)}")
print("WORST CASE: still + jitter + rope ABOVE gate (0.01) [gate can't help, floor only]:")
print(f"  sens5={run(0.55+rng.normal(0,jit,N),ts,np.full(N,0.01),sens=5)} sens10={run(0.55+rng.normal(0,jit,N),ts,np.full(N,0.01),sens=10)}")
print("SLOW SWAY 0.3Hz amp0.02, rope=0:")
sway=0.55+0.02*np.sin(2*np.pi*0.3*ts)+rng.normal(0,jit,N)
print(f"  sens5={run(sway,ts,np.zeros(N),sens=5)} sens10={run(sway,ts,np.zeros(N),sens=10)}")

print("\nfloor sweep on REAL (sens=5): ", {f:run(bodyc,t,rope,sens=5,floor=f) for f in [0.008,0.010,0.012,0.015,0.02]})
