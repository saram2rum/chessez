from common import *
import csv
fens=json.load(open('fens.json')); E=np.load('E.npy'); G=np.load('G.npy'); P=np.load('P.npy'); Ta=np.array(T); N=len(fens); idx=np.arange(N)
gb={}
for fn in ['/home/claude/exp/gtB_0.csv','/home/claude/exp/gtB_1.csv']:
    for r in csv.DictReader(open(fn)):
        v=int(r['value']); gb[r['fen']]= wp(v) if r['type']=='cp' else (100.0 if v>0 else 0.0)
GB=np.array([gb[f] for f in fens])
print('GT A vs B: MAE %.4f, identical %.1f%%, median abs %.3f, p90 %.3f'%(np.abs(G-GB).mean(),(np.isclose(G,GB)).mean()*100,np.median(np.abs(G-GB)),np.percentile(np.abs(G-GB),90)))
EB=np.abs(P-GB[:,None])
print('wu curve on B',np.round(EB.mean(0),4))
bA=best_t_idx(E); bB=best_t_idx(EB)
# p25 grid
p25={}
for fn in ['/home/claude/exp/grid_p25.csv','/home/claude/exp/grid_p25_rev.csv']:
    for r in csv.DictReader(open(fn)):
        v=int(r['value']); p25[(r['fen'],int(r['t']))]= wp(v) if r['type']=='cp' else (100.0 if v>0 else 0.0)
P2=np.array([[p25[(f,t)] for t in T] for f in fens]); E2B=np.abs(P2-GB[:,None]); E2A=np.abs(P2-G[:,None])
res={}
res['gtAB_mae']=np.abs(G-GB).mean()
res['oracle_A']=E[idx,bA].mean(); res['wu250_A']=E[:,4].mean()
res['oracle_B']=EB[idx,bB].mean(); res['wu250_B']=EB[:,4].mean()
res['cross_gt']=EB[idx,bA].mean()          # same measurement, independent GT
res['cross_both']=E2B[idx,bA].mean()       # independent measurement and GT
res['wu275_2B']=E2B[:,4].mean(); res['wu250_2B_interp']=np.interp(268,Ta+25,E2B.mean(0))
res['cross_both_t']=(Ta[bA]+25).mean()
for k,v in res.items(): print(k,round(float(v),4))
print('gap A: %.1f%%'%((res['wu250_A']-res['oracle_A'])/res['wu250_A']*100))
print('cross GT (meas same): %.1f%% vs wu250 on B'%((res['wu250_B']-res['cross_gt'])/res['wu250_B']*100))
print('cross both: %.1f%% vs wu275 on grid2/B'%((res['wu275_2B']-res['cross_both'])/res['wu275_2B']*100))
from scipy import stats
print('best_t A vs B spearman',stats.spearmanr(bA,bB)[0],'agree',(bA==bB).mean())
b2B=best_t_idx(E2B); print('best_t (gridA,GTA) vs (grid2,GTB) spearman',stats.spearmanr(bA,b2B)[0])
np.save('GB.npy',GB); np.save('E2B.npy',E2B); np.save('EB.npy',EB)
json.dump({k:float(v) for k,v in res.items()},open('gtb.json','w'),indent=1)
