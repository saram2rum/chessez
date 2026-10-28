import subprocess, re
class SF:
    def __init__(s, path, threads=1, hash_mb=16):
        s.p=subprocess.Popen([path],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        s.cmd('uci'); s.wait('uciok')
        s.cmd(f'setoption name Threads value {threads}'); s.cmd(f'setoption name Hash value {hash_mb}')
        s.cmd('isready'); s.wait('readyok')
    def cmd(s,c): s.p.stdin.write(c+'\n'); s.p.stdin.flush()
    def wait(s,tok):
        while True:
            l=s.p.stdout.readline()
            if l.startswith(tok): return l
    def eval(s, fen, nodes=None, depth=None, newgame=True):
        if newgame: s.cmd('ucinewgame')
        s.cmd('isready'); s.wait('readyok')
        s.cmd('position fen '+fen); s.cmd(f'go nodes {nodes}' if nodes else f'go depth {depth}')
        last=None
        while True:
            l=s.p.stdout.readline()
            if l.startswith('info') and ' score ' in l and 'multipv' in l or (l.startswith('info') and ' score ' in l): last=l
            if l.startswith('bestmove'): break
        m=re.search(r' depth (\d+)',last); sd=re.search(r' seldepth (\d+)',last); n=re.search(r' nodes (\d+)',last)
        sc=re.search(r' score (cp|mate) (-?\d+)',last)
        t,v=sc.group(1),int(sc.group(2))
        # convert to white POV
        stm_white = fen.split()[1]=='w'
        if not stm_white: v=-v
        return dict(type=t,value=v,depth=int(m.group(1)),seldepth=int(sd.group(1)) if sd else None,nodes=int(n.group(1)))
    def quit(s): s.cmd('quit')
