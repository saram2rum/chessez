import csv, sys, os, json
sys.path.insert(0,'/home/claude/sf'); from uci import SF
part,nparts,depth,out=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]
fens=[r['fen'] for r in csv.DictReader(open('/home/claude/cz/paper_legacy/data/mixed_workload_pool.csv'))][part::nparts]
done=set()
if os.path.exists(out):
    for r in csv.DictReader(open(out)): done.add(r['fen'])
f=open(out,'a',newline=''); w=csv.writer(f)
if not done: w.writerow(['fen','type','value','depth','nodes'])
e=SF('/home/claude/sf/stockfish/stockfish-ubuntu-x86-64-avx2', threads=1, hash_mb=16)
for fen in fens:
    if fen in done: continue
    r=e.eval(fen, depth=depth)   # ucinewgame each time -> deterministic
    w.writerow([fen,r['type'],r['value'],r['depth'],r['nodes']]); f.flush()
print('done')
