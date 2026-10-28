from common import *
import csv
fens=json.load(open('fens.json')); E=np.load('E.npy'); G=np.load('G.npy')
Gm={f:G[i] for i,f in enumerate(fens)}; Ta=np.array(T); wu=E.mean(0)
tr={}
for l in open('/home/claude/exp/trace_500.jsonl'):
    d=json.loads(l); tr[d['fen']]=d['trace']
raw500={r['fen']:r['cp'] for r in csv.DictReader(open(f'{RAW}/searchtime_500ms_raw.csv'))}
keep=[f for f in fens if f in tr and tr[f]]
def W(t,v): return wp(v) if t=='cp' else (100.0 if v>0 else 0.0)
# validation: last trace value vs raw500
agree=sum(1 for f in keep if (tr[f][-1][2]=='cp' and str(tr[f][-1][3])==raw500[f]) or (tr[f][-1][2]=='mate' and raw500[f] in ('','None')))
print('n',len(keep),'last-trace==raw500',agree/len(keep))
idx={f:i for i,f in enumerate(fens)}
sel=np.array([idx[f] for f in keep])
def policy(delta,k,dmin):
    errs=[];cost=[]
    for f in keep:
        t_=tr[f]; ws=[W(x[2],x[3]) for x in t_]
        stop=len(t_)-1; run=0
        for j in range(1,len(t_)):
            run = run+1 if abs(ws[j]-ws[j-1])<=delta else 0
            if t_[j][0]>=dmin and run>=k: stop=j; break
        errs.append(abs(ws[stop]-Gm[f])); cost.append(min(t_[stop][1],180000)/360)
    return np.mean(errs),np.mean(cost)
print('wu on subset',[round(E[sel,k].mean(),4) for k in range(10)])
wus=E[sel].mean(0)
for dmin in [8,12]:
  for k in [1,2,3]:
    for delta in [0.25,0.5,1.0,2.0]:
        e,c=policy(delta,k,dmin); print(f'dmin{dmin} k{k} delta{delta}: MAE {e:.4f} mean {c:.1f}ms  wu@same {np.interp(c,Ta,wus):.4f}')
