import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# CJK font
prop=None
for p in ['/System/Library/Fonts/PingFang.ttc','/System/Library/Fonts/STHeiti Medium.ttc',
          '/Library/Fonts/Arial Unicode.ttf','/System/Library/Fonts/Supplemental/Arial Unicode.ttf']:
    try:
        fm.fontManager.addfont(p); prop=fm.FontProperties(fname=p)
        plt.rcParams['font.family']=prop.get_name(); break
    except Exception: pass
plt.rcParams['axes.unicode_minus']=False

A=json.load(open("/Users/cheng/Downloads/skip-video-A.json"))
J=A['jumps']; dur=A['durationMs']/1000; n=A['count']
INK="#0e1420"; PANEL="#151d2e"; PAPER="#e9edf5"; DIM="#8593ab"; HOT="#ff4b3e"; GO="#3ddc84"; LINE="#26324a"

xt=np.array([j['t']/1000 for j in J])
cad=np.array([ (60000/j['iv'] if j['iv'] else np.nan) for j in J])
# rolling-median trend (window 7), ignore nan
def rollmed(y,w=7):
    out=np.full(len(y),np.nan)
    for i in range(len(y)):
        s=y[max(0,i-w//2):i+w//2+1]; s=s[~np.isnan(s)]
        if len(s): out[i]=np.median(s)
    return out
trend=rollmed(cad,7)
gaps=[(j['t']/1000-j['iv']/1000, j['t']/1000, j['iv']/1000) for j in J if j['iv'] and j['iv']>800]

fig,ax=plt.subplots(figsize=(15,7.2)); fig.patch.set_facecolor(INK); ax.set_facecolor(INK)
# per-10s average bars (faint)
for a in range(0,60,10):
    c=sum(1 for j in J if a*1000<=j['t']<(a+10)*1000); v=c*6
    ax.add_patch(plt.Rectangle((a,0),10,v,color=PANEL,zorder=0))
    ax.text(a+5,v+4,f"{c}",color=DIM,ha='center',fontsize=11,fontproperties=prop)
# trip spans
for a,b,g in gaps:
    ax.axvspan(a,b,color=HOT,alpha=0.22,zorder=1)
    ax.text((a+b)/2,238,f"绊绳\n{g:.1f}s",color=HOT,ha='center',va='top',fontsize=10,fontproperties=prop,fontweight='bold')
# pace trend (green, left axis)
ax.plot(xt,trend,color=GO,lw=2.6,zorder=4,label="配速趋势(次/分)")
# vertical height on a twin axis (gold, right axis) — per-jump amplitude, % of frame height
GOLD="#f5c344"
ht=np.array([j['amp']*100 for j in J]); htrend=rollmed(ht,7)
ax2=ax.twinx(); ax2.set_facecolor('none'); ax2.patch.set_visible(False)
ax2.plot(xt,ht,color=GOLD,lw=0.8,alpha=0.30,zorder=5)
ax2.scatter(xt,ht,s=9,color=GOLD,alpha=0.55,zorder=6)
ax2.plot(xt,htrend,color=GOLD,lw=2.4,zorder=6,label="垂直高度趋势")
ax2.set_ylim(0,8); ax2.set_xlim(0,60.6)
ax2.set_ylabel("垂直高度 (% 画面高度)",color=GOLD,fontsize=12,fontproperties=prop)
ax2.tick_params(axis='y',colors=GOLD)
for s in ax2.spines.values(): s.set_color(LINE)
# reference lines
avg=n/dur*60
ax.axhline(avg,color=PAPER,ls='--',lw=1,alpha=0.6,zorder=2)
ax.text(0.3,avg+3,f"平均 {avg:.0f}/分",color=PAPER,fontsize=11,fontproperties=prop)
ax.axhline(160,color="#5b8cff",ls=':',lw=1.4,alpha=0.8,zorder=2)
ax.text(59.6,163,"目标匀速 160/分",color="#8fb0ff",fontsize=11,ha='right',fontproperties=prop)

ax.set_xlim(0,60.6); ax.set_ylim(0,250)
ax.set_xlabel("时间 (秒)",color=PAPER,fontsize=12,fontproperties=prop)
ax.set_ylabel("配速 (次/分钟)",color=PAPER,fontsize=12,fontproperties=prop)
ax.set_title(f"跳绳配速 + 垂直高度 · 共 {n} 次 / {dur:.1f}s（{avg:.0f}/分）· 绊绳 {len(gaps)} 次全在后段",
             color=PAPER,fontsize=15,fontproperties=prop,pad=14,fontweight='bold')
ax.tick_params(colors=DIM); [s.set_color(LINE) for s in ax.spines.values()]
ax.set_xticks(range(0,61,5)); ax.grid(axis='y',color=LINE,alpha=0.35)
h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
lg=ax.legend(h1+h2,l1+l2,loc='lower left',facecolor=PANEL,edgecolor=LINE,labelcolor=PAPER,fontsize=11)
for txt in lg.get_texts(): txt.set_fontproperties(prop)
# annotation: fast start / late collapse
ax.annotate("前段偏猛 ~174/分",xy=(10,174),xytext=(13,205),color=HOT,fontsize=11,fontproperties=prop,
            arrowprops=dict(color=HOT,arrowstyle='->'))
ax.annotate("后段崩 ~72/分",xy=(55,72),xytext=(44,40),color=HOT,fontsize=11,fontproperties=prop,fontweight='bold',
            arrowprops=dict(color=HOT,arrowstyle='->'))
plt.tight_layout()
out="/Users/cheng/Downloads/pace_chart.png"
plt.savefig(out,dpi=120,facecolor=INK); print("saved",out)
