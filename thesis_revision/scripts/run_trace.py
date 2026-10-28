# 각 FEN을 500ms 상당(180,000 nodes)까지 탐색하며 깊이별 (depth, nodes, score) 기록
import csv, sys, os, re, time, json, subprocess
fens=[r['fen'] for r in csv.DictReader(open('/home/claude/cz/paper_legacy/data/mixed_workload_pool.csv'))]
out='/home/claude/exp/trace_500.jsonl'
done=set()
if os.path.exists(out):
    for l in open(out): done.add(json.loads(l)['fen'])
p=subprocess.Popen(['/home/claude/sf/stockfish/stockfish-ubuntu-x86-64-avx2'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
def cmd(c): p.stdin.write(c+'\n'); p.stdin.flush()
def wait(t):
    while True:
        l=p.stdout.readline()
        if l.startswith(t): return
cmd('uci'); wait('uciok'); cmd('setoption name Threads value 1'); cmd('setoption name Hash value 16'); cmd('isready'); wait('readyok')
f=open(out,'a')
for fen in fens:
    if fen in done: continue
    cmd('ucinewgame'); cmd('isready'); wait('readyok')
    cmd('position fen '+fen); cmd('go nodes 180000')
    tr=[]; white = fen.split()[1]=='w'
    while True:
        l=p.stdout.readline()
        if l.startswith('bestmove'): break
        if l.startswith('info') and ' score ' in l and ' depth ' in l and 'bound' not in l:
            d=int(re.search(r' depth (\d+)',l).group(1)); n=int(re.search(r' nodes (\d+)',l).group(1))
            t,v=re.search(r' score (cp|mate) (-?\d+)',l).groups(); v=int(v)
            if not white: v=-v
            tr.append([d,n,t,v])
    f.write(json.dumps({'fen':fen,'trace':tr})+'\n'); f.flush()
cmd('quit'); print('done')
