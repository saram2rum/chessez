from common import *
from scipy import stats
import chess
fens=json.load(open('fens.json')); E=np.load('E.npy'); G=np.load('G.npy'); X=np.load('X.npy'); g=np.load('game.npy'); P=np.load('P.npy')
N=len(fens); idx=np.arange(N); Ta=np.array(T)
b=best_t_idx(E); bt=Ta[b]
rng=np.random.default_rng(42)
def boot_diff(a,bv,n=10000):
    d=a-bv; m=d.mean(); bs=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(n)]
    return m, np.percentile(bs,[2.5,97.5])
out={}
# S/M/L policy
from app.utils.position_complexity import evaluate_position_complexity
bk=[]; 
for f in fens:
    c=evaluate_position_complexity(f); bk.append(c.bucket)
bk=np.array(bk); tmap={'S':1,'M':4,'L':7}
cxi=np.array([tmap[x] for x in bk]); cx=E[idx,cxi]
print('buckets',{k:(bk==k).sum() for k in 'SML'}, 'mean t',Ta[cxi].mean())
print('cx MAE',cx.mean(),'wu250',E[:,4].mean(), 'diff CI',boot_diff(cx,E[:,4]))
# weighted / band metrics
band=(G>=25)&(G<=75); print('band 25-75 share',band.mean())
w=np.exp(-((G/100-0.5)**2)/(2*0.10**2))
for name,vec in [('wu250',E[:,4]),('wu500',E[:,9]),('oracle',E[idx,b]),('cx',cx)]:
    print(name,'MAE',round(vec.mean(),4),'band',round(vec[band].mean(),4),'gw0.10',round((w*vec).sum()/w.sum(),4))
print('wu curve band',[round(E[band,k].mean(),3) for k in range(10)])
# oracle within band
bb=best_t_idx(E[band]); print('oracle band',E[band][np.arange(band.sum()),bb].mean(),'mean t',Ta[bb].mean(),'wu250 band',E[band,4].mean())
# Spearman CI bootstrap
for j,n in enumerate(FEAT):
    r,p=stats.spearmanr(X[:,j],bt)
    bs=[]
    for _ in range(1000):
        s=rng.integers(0,N,N); bs.append(stats.spearmanr(X[s,j],bt[s])[0])
    print(n,round(r,4),'p=%.3f'%p,'CI',np.round(np.nanpercentile(bs,[2.5,97.5]),4))
# alternative targets
err50=E[:,0]; red=E[:,0]-E[:,9]
eps=1.0
conv=[]
for i in range(N):
    ok=E[i]<=eps; t_=550
    for k in range(10):
        if ok[k:].all(): t_=T[k]; break
    conv.append(t_)
conv=np.array(conv)
print('conv time dist', {t:int((conv==t).sum()) for t in T+[550]})
nontie=~np.isclose(E,E.min(1,keepdims=True),atol=1e-9).all(1)
print('all-tie excluded n',nontie.sum())
for name,y,mask in [('err50',err50,None),('reduction50to500',red,None),('conv_time_eps1',conv,None),('best_t_no_alltie',bt,nontie)]:
    m=np.ones(N,bool) if mask is None else mask
    rs=[stats.spearmanr(X[m,j],y[m])[0] for j in range(7)]
    print(name,[round(r,3) for r in rs], 'max|rho|',round(max(abs(r) for r in rs),3))
np.save('conv.npy',conv); np.save('bk.npy',bk)
