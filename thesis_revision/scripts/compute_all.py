# 논문 표·수치 생성 (results.json, tables.json)
from common import *
from scipy import stats
import csv
fens=json.load(open('/home/claude/ana/fens.json')); E=np.load('/home/claude/ana/E.npy'); G=np.load('/home/claude/ana/G.npy')
N=len(fens); idx=np.arange(N); Ta=np.array(T); wu=E.mean(0)
R={}; TB={}
band=(G>=25)&(G<=75)
b=best_t_idx(E)
# --- depth table
dep={t:[] for t in T}
for r in csv.DictReader(open('/home/claude/exp/grid_orig_sub.csv')): dep[int(r['t'])].append(int(r['depth']))
rows=[['예산 (ms)','노드 수','깊이 중앙값','깊이 평균','MAE (%p)','접전 MAE (%p)']]
for k,t in enumerate(T):
    rows.append([str(t),f'{360*t:,}',f'{np.median(dep[t]):.0f}',f'{np.mean(dep[t]):.1f}',f'{wu[k]:.4f}',f'{E[band,k].mean():.4f}'])
TB['WU_TABLE']=rows
# --- permutation
rng=np.random.default_rng(0)
perm=np.array([E[idx,rng.permutation(b)].mean() for _ in range(2000)])
R['PERM_MEAN']=f'{perm.mean():.3f}'; pl,ph=np.percentile(perm,[2.5,97.5])
# --- p25 grid
p25={}
import os
for r in [x for fn in ['/home/claude/exp/grid_p25.csv','/home/claude/exp/grid_p25_rev.csv'] if os.path.exists(fn) for x in csv.DictReader(open(fn))]:
    v=int(r['value']); p25[(r['fen'],int(r['t']))]= wp(v) if r['type']=='cp' else (100.0 if v>0 else 0.0)
have=[i for i,f in enumerate(fens) if all((f,t) in p25 for t in T)]
hv=np.array(have); R['P25_N']=len(have)
E25=np.array([[abs(p25[(fens[i],t)]-G[i]) for t in T] for i in have])
cross=E25[np.arange(len(have)),b[hv]].mean()
crosst=(Ta[b[hv]]+25).mean()
wu25=E25.mean(0)  # grid times 75..525
# compare to wu on grid2 at nearest (275) and interpolated at crosst
wu25_interp=np.interp(crosst,Ta+25,wu25)
k275=4  # 250+25
R['P25_CROSS']=f'{cross:.4f}'; R['P25_WU_NAME']='wu(275)'; R['P25_WU']=f'{wu25[k275]:.4f}'
R['P25_CROSS_PCT']=f'{(wu25[k275]-cross)/wu25[k275]*100:.0f}'
o1=E[hv][np.arange(len(have)),b[hv]].mean(); w1=E[hv,4].mean()
R['P25_KEEP']=f'{((wu25[k275]-cross)/wu25[k275])/((w1-o1)/w1)*100:.0f}'
b25=best_t_idx(E25); R['P25_RHO']=f'{stats.spearmanr(b[hv],b25)[0]:.2f}'
R['P25_AGREE']=f'{(b[hv]==b25).mean()*100:.1f}'
R['P25_IDENT']=f'{np.isclose(E[hv],E25,atol=1e-9).mean()*100:.1f}'
o25=E25[np.arange(len(have)),b25].mean()
R['P25_ORACLE2']=f'{o25:.4f}'
R['P25_CROSST']=f'{crosst:.0f}'
TB['ORACLE_TABLE']=[['정책','평균 예산','MAE (%p)','wu(250) 대비','비고'],
 ['wu(250)','250 ms',f'{wu[4]:.4f}','-','균일 정책'],
 ['wu(500)','500 ms',f'{wu[9]:.4f}',f'{(wu[4]-wu[9])/wu[4]*100:.1f}%','시간 2배'],
 ['이론적 상한 (best_t)','243 ms',f'{E[idx,b].mean():.4f}',f'{(wu[4]-E[idx,b].mean())/wu[4]*100:.1f}%','사후 선택 포함'],
 ['무작위 배분 기준선','243 ms',f'{perm.mean():.4f}',f'{(wu[4]-perm.mean())/wu[4]*100:.1f}%',f'2,000회 평균, 95% 구간 [{pl:.3f}, {ph:.3f}]'],
 ['교차 평가 (두 번째 격자)',f'{crosst:.0f} ms',f'{cross:.4f}',f'{(wu25[k275]-cross)/wu25[k275]*100:.1f}%*','사후 선택 제외'],
]
R['ORACLE_NOTE_WU275']=f'{wu25[k275]:.4f}'
# --- spearman table
X=np.load('/home/claude/ana/X.npy'); bt=Ta[b]
desc={'imbalance':'재료 불균형(퀸 9·룩 5·비숍/나이트 3)','capture_count':'현재 측의 합법 캡처 수','legal_moves':'합법 수 총 개수(분기 인자)',
'total_material':'양측 비폰 재료 가치 합(게임 단계)','forcing_total':'양측 캡처+체크 수 합','is_check':'현재 체크 상태(0/1)','is_endgame':'비폰 재료 합 ≤ 24(종반 여부)'}
rows=[['feature','설명','Spearman ρ','p값','95% 신뢰구간']]
rng=np.random.default_rng(42)
for j,n in enumerate(FEAT):
    r,p=stats.spearmanr(X[:,j],bt); bs=[]
    for _ in range(1000):
        s=rng.integers(0,N,N); bs.append(stats.spearmanr(X[s,j],bt[s])[0])
    lo,hi=np.nanpercentile(bs,[2.5,97.5])
    f=lambda x: ('+' if x>=0 else '−')+f'{abs(x):.4f}'
    rows.append([n,desc[n],f(r),f'{p:.3f}',f'[{f(lo)}, {f(hi)}]'])
TB['SPEARMAN_TABLE']=rows
json.dump(R,open('/home/claude/ana/results.json','w'),ensure_ascii=False,indent=1)
json.dump(TB,open('/home/claude/ana/tables.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(R,ensure_ascii=False,indent=1))

# ================= part 2: RF, alt, dyn, band, mate =================
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
g=np.load('/home/claude/ana/game.npy'); red=E[:,0]-E[:,9]
def snap(v): return np.abs(v[:,None]-Ta[None,:]).argmin(1)
rng=np.random.default_rng(0)
def ci(d,n=5000):
    bs=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(n)]; return np.percentile(bs,[2.5,97.5])
res=[]
for seed in range(10):
    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=seed).split(X,bt,groups=g))
    m=RandomForestRegressor(n_estimators=400,min_samples_leaf=5,random_state=seed,n_jobs=2).fit(X[tr],bt[tr])
    pi=snap(m.predict(X[te])); pol=E[te,pi]; w=E[te,4]; mt=Ta[pi].mean()
    winterp=np.array([np.interp(mt,Ta,E[te].mean(0))])[0]
    trm=E[tr,snap(m.predict(X[tr]))].mean()
    c=ci(pol-w)
    m2=RandomForestRegressor(n_estimators=400,min_samples_leaf=5,random_state=seed,n_jobs=2).fit(X[tr],red[tr])
    pr=m2.predict(X[te]); q=(np.argsort(np.argsort(pr))*5)//len(te); qi=np.array([0,2,4,6,8])[q]; pol2=E[te,qi]; c2=ci(pol2-w)
    res.append(dict(reg=pol.mean(),tr=trm,wu=w.mean(),wi=winterp,mt=mt,c=c,red=pol2.mean(),c2=c2,nte=len(te),ngte=len(set(g[te]))))
A_=lambda k: np.array([r[k] for r in res])
R['RF_REG']=f"{A_('reg').mean():.3f}"; R['RF_WU']=f"{A_('wu').mean():.3f}"; R['RF_D']=f"{(A_('reg')-A_('wu')).mean():+.3f}"
R['RF_DSD']=f"{(A_('reg')-A_('wu')).std(ddof=1):.3f}"; R['RF_MT']=f"{A_('mt').mean():.1f}"; R['RF_TR']=f"{A_('tr').mean():.3f}"
R['RF_WI']=f"{A_('wi').mean():.3f}"; R['RF_DWI']=f"{(A_('reg')-A_('wi')).mean():+.3f}"
R['RF_WUWIN']=str(sum(r['c'][0]>0 for r in res)); R['RF_POLWIN']=str(sum(r['c'][1]<0 for r in res))
R['RED_MEAN']=f"{A_('red').mean():.3f}"; R['RED_D']=f"{(A_('red')-A_('wu')).mean():+.3f}"; R['RED_DSD']=f"{(A_('red')-A_('wu')).std(ddof=1):.3f}"
R['RED_POLWIN']=str(sum(r['c2'][1]<0 for r in res)); R['RED_WUWIN']=str(sum(r['c2'][0]>0 for r in res))
R['RF_NTE']=f"{A_('nte').mean():.0f}"
TB['RF_TABLE']=[['항목','RandomForest 회귀','wu(250)','차이 (회귀 − wu)'],
 ['평가 국면 MAE (%p, 10회 평균)',R['RF_REG'],R['RF_WU'],R['RF_D']],
 ['분할 간 표준편차 (%p)',f"{A_('reg').std(ddof=1):.3f}",f"{A_('wu').std(ddof=1):.3f}",R['RF_DSD']],
 ['평균 예산 (ms)',R['RF_MT'],'250','-'],
 ['같은 예산으로 보간한 wu (%p)','-',R['RF_WI'],R['RF_DWI']],
 ['학습 국면 MAE (%p)',R['RF_TR'],'-','-'],
 ['유의하게 우세한 분할 수','0' if R['RF_POLWIN']=='0' else R['RF_POLWIN'],R['RF_WUWIN'],'10회 중'],
]
# alt targets
from scipy import stats as st
eps=1.0; conv=[]
for i in range(N):
    ok=E[i]<=eps; t_=550
    for k in range(10):
        if ok[k:].all(): t_=T[k]; break
    conv.append(t_)
conv=np.array(conv)
nontie=~np.isclose(E,E.min(1,keepdims=True),atol=1e-9).all(1)
def maxr(y,m=None):
    m=np.ones(N,bool) if m is None else m
    rs=[st.spearmanr(X[m,j],y[m])[0] for j in range(7)]; j=int(np.argmax(np.abs(rs))); return FEAT[j],rs[j]
rows=[['목표값','최대 |ρ| (feature)','배분 정책 결과']]
for name,y,m,pol in [('best_t (표 8)',bt,None,f"RF 회귀: wu 대비 {R['RF_D']}%p"),
                     ('best_t (동률 국면 제외)',bt,nontie,'-'),
                     ('50ms에서의 오차',E[:,0],None,'-'),
                     ('50→500ms 오차 감소량',red,None,f"5등분 배분: wu 대비 {R['RED_D']}%p"),
                     ('수렴 예산 (1%p 이내)',conv,None,'-')]:
    f,r=maxr(y,m); rows.append([name,f"{abs(r):.3f} ({f})",pol])
TB['ALT_TABLE']=rows
# band table + dyn (cp50)
P=np.load('/home/claude/ana/P.npy')
cp50=np.array([math.log(p/(100-p))/0.00368208 if 0<p<100 else (1e4 if p>=100 else -1e4) for p in P[:,0]])
order=np.argsort(np.argsort(-np.abs(cp50))); q=(order*5)//N; ti=np.array([0,2,4,6,8])[q]; pol=E[idx,ti]
cost=np.where(ti==0,50,50+Ta[ti]).mean()
R['CP50_MAE']=f'{pol.mean():.4f}'; R['CP50_COST']=f'{cost:.0f}'; R['CP50_WUI']=f'{np.interp(cost,Ta,wu):.4f}'
R['CP50_BAND']=f'{pol[band].mean():.4f}'; R['CP50_BANDWUI']=f'{np.interp(cost,Ta,[E[band,k].mean() for k in range(10)]):.4f}'
from app.utils.position_complexity import evaluate_position_complexity
bk=np.array([evaluate_position_complexity(f).bucket for f in fens]); cxi=np.array([{'S':1,'M':4,'L':7}[x] for x in bk]); cx=E[idx,cxi]
TB['BAND_TABLE']=[['정책','평균 예산','전체 MAE (%p)','접전 MAE (%p)'],
 ['wu(250)','250 ms',f'{wu[4]:.4f}',f'{E[band,4].mean():.4f}'],
 ['wu(500)','500 ms',f'{wu[9]:.4f}',f'{E[band,9].mean():.4f}'],
 ['S/M/L 휴리스틱','245.7 ms',f'{cx.mean():.4f}',f'{cx[band].mean():.4f}'],
 ['이론적 상한','243 ms',f'{E[idx,b].mean():.4f}',f'{E[band][np.arange(band.sum()),b[band]].mean():.4f}'],
 ['2단계 동적 정책',f'{cost:.0f} ms (실제 비용)',R['CP50_MAE'],R['CP50_BAND']],
 [f'wu({cost:.0f}) (보간)',f'{cost:.0f} ms',R['CP50_WUI'],R['CP50_BANDWUI']],
]
# mate table
fen='r6k/6nP/1pP1p1P1/3p1p1R/3q4/3P4/4KPB1/8 w - - 0 39'; i=fens.index(fen)
cpr={}
for t in T:
    for r in csv.DictReader(open(f'{RAW}/searchtime_{t}ms_raw.csv')):
        if r['fen']==fen: cpr[t]=r['cp'] or 'M'
sel=[50,100,150,250,400,500]
TB['MATE_TABLE']=[['예산 (ms)']+[str(t) for t in sel],
 ['평가치 (cp)']+[('흑 메이트' if cpr[t] in ('M','None') else cpr[t].replace('-','−')) for t in sel],
 ['승률 오차 (%p)']+[f'{E[i,T.index(t)]:.1f}' for t in sel]]
json.dump(R,open('/home/claude/ana/results.json','w'),ensure_ascii=False,indent=1)
json.dump(TB,open('/home/claude/ana/tables.json','w'),ensure_ascii=False,indent=1)
print('part2 done')

# ================= part 3: early stop =================
tr={}
for l in open('/home/claude/exp/trace_500.jsonl'):
    d=json.loads(l); tr[d['fen']]=d['trace']
Gm={f:G[i] for i,f in enumerate(fens)}
def W_(t,v): return wp(v) if t=='cp' else (100.0 if v>0 else 0.0)
def es(delta,k,dmin):
    errs=[];cost=[]
    for f in fens:
        t_=tr[f]; ws=[W_(x[2],x[3]) for x in t_]; stop=len(t_)-1; run=0
        for j in range(1,len(t_)):
            run = run+1 if abs(ws[j]-ws[j-1])<=delta else 0
            if t_[j][0]>=dmin and run>=k: stop=j; break
        errs.append(abs(ws[stop]-Gm[f])); cost.append(min(t_[stop][1],180000)/360)
    return np.mean(errs),np.mean(cost)
cfgs=[(d,k,dm) for dm in [8,12] for k in [1,2,3] for d in [0.25,0.5,1.0,2.0]]
worse=0; allres=[]
for d,k,dm in cfgs:
    e,c=es(d,k,dm); wi=np.interp(c,Ta,wu); allres.append((d,k,dm,e,c,wi)); worse+= e>wi
R['ES_NCFG']=str(len(cfgs)); R['ES_WORSE']=str(worse)
e1,c1=es(0.5,2,12); e2,c2=es(0.5,3,12)
R['DYN_DMIN']='12'; R['DYN_DELTA']='0.5'; R['DYN_K']='2'
TB['DYN_TABLE']=[['정책','평균 비용','MAE (%p)','같은 비용의 wu (%p)'],
 ['wu(250)','250 ms',f'{wu[4]:.4f}',f'{wu[4]:.4f}'],
 ['2단계 정책 (50ms 탐색 후 5등분)',f"{R['CP50_COST']} ms",R['CP50_MAE'],R['CP50_WUI']],
 ['조기 중단 (깊이≥12, 0.5%p, 2회)',f'{c1:.0f} ms',f'{e1:.4f}',f'{np.interp(c1,Ta,wu):.4f}'],
 ['조기 중단 (깊이≥12, 0.5%p, 3회)',f'{c2:.0f} ms',f'{e2:.4f}',f'{np.interp(c2,Ta,wu):.4f}'],
]
R['DYN_TEXT']=(f"결과는 표 11과 같다. 2단계 정책의 MAE는 {R['CP50_MAE']}%p로, 실제 비용({R['CP50_COST']}ms)에 해당하는 균일 정책(보간값 {R['CP50_WUI']}%p)보다 컸다. "
 f"반복 심화 조기 중단 정책도 평균 비용 {c1:.0f}ms에서 MAE {e1:.4f}%p로 같은 비용의 균일 정책({np.interp(c1,Ta,wu):.4f}%p)보다 컸으며, "
 f"중단 조건(깊이 하한 8·12, 허용 변화 0.25~2%p, 연속 횟수 1~3회)을 바꾼 {len(cfgs)}개 설정 중 {worse}개에서 같은 비용의 균일 정책보다 오차가 컸다. "
 "이는 평가치가 몇 단계 동안 거의 변하지 않다가 더 깊은 단계에서 바뀌는 국면에서, ‘평가치가 안정되면 멈춘다’는 규칙이 필요한 탐색을 끊기 때문으로 보인다. "
 "다만 조기 중단 정책은 반복 심화의 각 단계가 끝난 시점의 평가치를 사용하고, 균일 정책은 노드 예산에서 탐색을 끊은 시점의 평가치를 사용하므로 두 값의 성격이 완전히 같지는 않다. "
 "즉 MAE 기준으로는 탐색 중의 정보를 이용하는 동적 정책도 균일 정책을 이기지 못하였으며, 탐색 이전에 예산을 정하는 본 기법의 구조적 선택이 품질 면에서 큰 손해는 아니었다.")
json.dump(R,open('/home/claude/ana/results.json','w'),ensure_ascii=False,indent=1)
json.dump(TB,open('/home/claude/ana/tables.json','w'),ensure_ascii=False,indent=1)
print('part3 done', worse, len(cfgs))
