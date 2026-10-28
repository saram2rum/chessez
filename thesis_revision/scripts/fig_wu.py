import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fp='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(fp); plt.rcParams['font.family']=fm.FontProperties(fname=fp).get_name()
E=np.load('/home/claude/ana/E.npy'); T=np.array([50,100,150,200,250,300,350,400,450,500]); wu=E.mean(0)
R=json.load(open('/home/claude/ana/results.json'))
fig,ax=plt.subplots(figsize=(7.2,4.2),dpi=220)
INK='#1f2937'; MUTED='#6b7280'; LINE='#2563eb'
ax.plot(T,wu,color=LINE,lw=2,marker='o',ms=5,zorder=3,label='균일 정책 wu(t)')
pts=[(245.7,1.8355,'S/M/L 휴리스틱','s',(320,1.98)),
     (243,float(R['PERM_MEAN']),'무작위 배분 기준선','v',(320,2.13)),
     (243,1.1185,'이론적 상한 (best_t)','D',(320,1.12))]
for x,y,lab,mk,(tx,ty) in pts:
    ax.scatter([x],[y],marker=mk,s=46,color=INK,zorder=4,edgecolor='white',linewidth=1.2)
    ax.annotate(lab,(x,y),xytext=(tx,ty),textcoords='data',fontsize=9,color=INK,va='center',ha='left',
                arrowprops=dict(arrowstyle='-',color='#9ca3af',lw=0.8,shrinkA=2,shrinkB=4))
ax.set_xlabel('탐색 예산 t (ms 상당 노드, 360×t nodes)',fontsize=10,color=INK)
ax.set_ylabel('MAE (승률 %p)',fontsize=10,color=INK)
ax.set_xticks(T); ax.tick_params(colors=MUTED,labelsize=9)
ax.set_ylim(0.9,2.6); ax.set_xlim(30,560)
ax.grid(axis='y',color='#e5e7eb',lw=0.8); ax.set_axisbelow(True)
for s in ['top','right']: ax.spines[s].set_visible(False)
for s in ['left','bottom']: ax.spines[s].set_color('#9ca3af')
ax.legend(frameon=False,fontsize=9,loc='upper right')
fig.tight_layout(); fig.savefig('/home/claude/w/img/fig_wu_curve.png',facecolor='white')
print('saved')
