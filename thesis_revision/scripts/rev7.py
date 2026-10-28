from common import *
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
fens=json.load(open('fens.json')); E=np.load('E.npy'); EB=np.load('EB.npy'); X=np.load('X.npy'); g=np.load('game.npy'); G=np.load('G.npy')
Ta=np.array(T); N=len(fens); idx=np.arange(N); out={}
games=np.unique(g); rng=np.random.default_rng(7)
def cboot(d,n=5000):
    by={k:d[g==k] for k in games}; s=[]
    for _ in range(n):
        pick=rng.choice(games,len(games)); s.append(np.concatenate([by[k] for k in pick]).mean())
    return np.percentile(s,[2.5,97.5])
from app.utils.position_complexity import evaluate_position_complexity
bk=np.array([evaluate_position_complexity(f).bucket for f in fens]); cxi=np.array([{'S':1,'M':4,'L':7}[x] for x in bk])
d=E[idx,cxi]-E[:,4]; out['sml_d']=d.mean(); out['sml_ci_game']=list(cboot(d))
dB=EB[idx,cxi]-EB[:,4]; out['sml_d_B']=dB.mean(); out['sml_ci_game_B']=list(cboot(dB))
# spearman cluster CI
bt=Ta[best_t_idx(E)]
rows=[]
for j,n in enumerate(FEAT):
    r=stats.spearmanr(X[:,j],bt)[0]; s=[]
    for _ in range(1000):
        pick=rng.choice(games,len(games)); ii=np.concatenate([np.where(g==k)[0] for k in pick]); s.append(stats.spearmanr(X[ii,j],bt[ii])[0])
    lo,hi=np.nanpercentile(s,[2.5,97.5]); rows.append((n,r,lo,hi))
out['spear_game']=rows
# RF: game-level judgement + predicted distribution, on A and on B targets
def rf_run(Emat):
    btm=Ta[best_t_idx(Emat)]; ds=[]; wins=[0,0]; hist=np.zeros(10)
    for seed in range(10):
        tr,te=next(GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=seed).split(X,btm,groups=g))
        rf=RandomForestRegressor(n_estimators=400,min_samples_leaf=5,random_state=seed,n_jobs=2).fit(X[tr],btm[tr])
        k=np.abs(rf.predict(X[te])[:,None]-Ta[None,:]).argmin(1); hist+=np.bincount(k,minlength=10)
        diff=Emat[te,k]-Emat[te,4]; gt=g[te]; gs=np.unique(gt); by={q:diff[gt==q] for q in gs}
        s=[np.concatenate([by[q] for q in rng.choice(gs,len(gs))]).mean() for _ in range(3000)]
        lo,hi=np.percentile(s,[2.5,97.5]); ds.append(diff.mean()); wins[0]+=lo>0; wins[1]+=hi<0
    return np.mean(ds),np.std(ds,ddof=1),wins,hist/hist.sum()
m,sd,w,h=rf_run(E); out['rf_A']=[m,sd,w,list(h)]
m,sd,w,h=rf_run(EB); out['rf_B']=[m,sd,w,list(h)]
# mate exclusion
pool={r['fen']:int(float(r['eval_value'])) for r in __import__('csv').DictReader(open('/home/claude/cz/paper_legacy/data/mixed_workload_pool.csv'))}
nm=np.array([abs(pool[f])!=10000 for f in fens])
Em=E[nm]; bm=best_t_idx(Em)
out['nomate']=dict(n=int(nm.sum()),wu250=Em[:,4].mean(),wu500=Em[:,9].mean(),oracle=Em[np.arange(nm.sum()),bm].mean(),
  sml=Em[np.arange(nm.sum()),cxi[nm]].mean())
# error contribution of top 5% positions to wu250 MAE
e=np.sort(E[:,4])[::-1]; out['top5_share']=e[:int(N*0.05)].sum()/e.sum()
out['median_err_wu250']=float(np.median(E[:,4]))
json.dump(out,open('rev7.json','w'),indent=1,default=float)
print(json.dumps(out,indent=1,default=float))
