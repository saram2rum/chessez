import csv, sys, os, math, time
sys.path.insert(0,'/home/claude/sf'); from uci import SF
mult=float(sys.argv[1]); out=sys.argv[2]
TIMES=[50,100,150,200,250,300,350,400,450,500]
fens=[r['fen'] for r in csv.DictReader(open('/home/claude/cz/paper_legacy/data/mixed_workload_pool.csv'))]
if len(sys.argv)>3:
    import random; random.seed(int(sys.argv[3])); fens=random.sample(fens,600)
done=set()
if os.path.exists(out):
    for r in csv.DictReader(open(out)): done.add((r['fen'],int(r['t'])))
f=open(out,'a',newline=''); w=csv.writer(f)
if not done: w.writerow(['fen','t','nodes_limit','type','value','depth','seldepth','nodes'])
e=SF('/home/claude/sf/stockfish/stockfish-ubuntu-x86-64-avx2')
t0=time.time(); n=0
for fen in fens:
    for t in TIMES:
        if (fen,t) in done: continue
        nl=int(round(360*t*mult))
        r=e.eval(fen,nodes=nl)
        w.writerow([fen,t,nl,r['type'],r['value'],r['depth'],r['seldepth'],r['nodes']]); n+=1
    f.flush()
print('done',n,time.time()-t0)
