from common import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
fens=json.load(open('fens.json')); E=np.load('E.npy'); X=np.load('X.npy'); g=np.load('game.npy')
N=len(fens); idx=np.arange(N); Ta=np.array(T)
b=best_t_idx(E); bt=Ta[b]; red=E[:,0]-E[:,9]
rng=np.random.default_rng(0)
def snap(v): return np.abs(v[:,None]-Ta[None,:]).argmin(1)
def ci(d,n=5000):
    bs=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(n)]; return np.percentile(bs,[2.5,97.5])
def run(splitter,label):
    res=[]
    for seed in range(10):
        sp=splitter(seed); tr,te=next(sp.split(X,bt,groups=g))
        m=RandomForestRegressor(n_estimators=400,min_samples_leaf=5,random_state=seed,n_jobs=2).fit(X[tr],bt[tr])
        pi=snap(m.predict(X[te])); pol=E[te,pi]; wu=E[te,4]
        tr_pol=E[tr,snap(m.predict(X[tr]))].mean()
        d=pol-wu; c=ci(d)
        # quantile policy on predicted error reduction
        m2=RandomForestRegressor(n_estimators=400,min_samples_leaf=5,random_state=seed,n_jobs=2).fit(X[tr],red[tr])
        pr=m2.predict(X[te]); order=np.argsort(np.argsort(pr)); q=(order*5)//len(te)  # 0..4 low->high reduction
        qi=np.array([0,2,4,6,8])[q]  # 50,150,250,350,450 -> mean 250
        pol2=E[te,qi]; d2=pol2-wu; c2=ci(d2)
        res.append(dict(seed=seed,ntest=len(te),reg=pol.mean(),reg_train=tr_pol,wu=wu.mean(),mt=Ta[pi].mean(),d=d.mean(),ci=c,
                        red=pol2.mean(),dred=d2.mean(),ci2=c2,mt2=Ta[qi].mean()))
    print('==',label)
    for r in res: print(r['seed'],r['ntest'],'RF best_t %.4f (train %.4f) wu %.4f mt %.1f d %+.4f CI [%+.4f,%+.4f] | RF-reduction %.4f d %+.4f CI [%+.4f,%+.4f]'%(r['reg'],r['reg_train'],r['wu'],r['mt'],r['d'],*r['ci'],r['red'],r['dred'],*r['ci2']))
    for k in ['reg','reg_train','wu','d','red','dred','mt']:
        v=np.array([r[k] for r in res]); print(k,'mean %.4f sd %.4f'%(v.mean(),v.std(ddof=1)))
    print('verdict best_t: wu better',sum(r['ci'][0]>0 for r in res),'policy better',sum(r['ci'][1]<0 for r in res))
    print('verdict reduction: wu better',sum(r['ci2'][0]>0 for r in res),'policy better',sum(r['ci2'][1]<0 for r in res))
run(lambda s: GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=s),'GroupShuffleSplit by game')
