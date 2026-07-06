import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle
import numpy as np

OUT="/private/tmp/claude-501/-Users-cheng-Maxgent-maxgent-worktree-flywheel/a51db06a-c38b-46e1-bb67-0b38f77be762/scratchpad/jump-rope/"
INK="#0e1420"; PAPER="#e9edf5"; HOT="#ff4b3e"; GO="#3ddc84"

def draw(px, pad=0.0):
    fig=plt.figure(figsize=(px/100,px/100),dpi=100)
    ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.add_patch(plt.Rectangle((0,0),1,1,color=INK,zorder=0))
    lw=px*0.052
    # rope: big lower arc sweeping under the feet, red
    ax.add_patch(Arc((0.5,0.50),0.66,0.86,angle=0,theta1=200,theta2=340,color=HOT,lw=lw,zorder=1,capstyle="round"))
    # rope: faint upper arc over the head
    ax.add_patch(Arc((0.5,0.62),0.66,0.86,angle=0,theta1=20,theta2=160,color=HOT,lw=lw*0.7,alpha=0.45,zorder=1,capstyle="round"))
    # jumper (white stick figure)
    fl=px*0.045
    ax.add_patch(Circle((0.5,0.66),0.058,color=PAPER,zorder=3))          # head
    ax.plot([0.5,0.5],[0.60,0.45],color=PAPER,lw=fl,solid_capstyle="round",zorder=3)  # torso
    ax.plot([0.5,0.34],[0.57,0.52],color=PAPER,lw=fl,solid_capstyle="round",zorder=3) # left arm
    ax.plot([0.5,0.66],[0.57,0.52],color=PAPER,lw=fl,solid_capstyle="round",zorder=3) # right arm
    ax.plot([0.5,0.42],[0.45,0.30],color=PAPER,lw=fl,solid_capstyle="round",zorder=3) # left leg
    ax.plot([0.5,0.58],[0.45,0.30],color=PAPER,lw=fl,solid_capstyle="round",zorder=3) # right leg
    # green hands (rope handles)
    ax.add_patch(Circle((0.34,0.52),0.028,color=GO,zorder=4))
    ax.add_patch(Circle((0.66,0.52),0.028,color=GO,zorder=4))
    fig.savefig(OUT+f"icon-{px}.png",dpi=100)
    plt.close(fig)
    return OUT+f"icon-{px}.png"

for px in [512,192,180]:
    print("wrote", draw(px))
