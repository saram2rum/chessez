import csv, math, json, sys, io
import numpy as np
sys.path.insert(0,'/home/claude/cz')
T=[50,100,150,200,250,300,350,400,450,500]
RAW='/home/claude/cz/paper_legacy/raw/searchtime_raw_local'
def wp(cp): return 50+50*(2/(1+math.exp(-0.00368208*cp))-1)
def load_orig():
    gt={}; pred={}
    for t in T:
        for r in csv.DictReader(open(f'{RAW}/searchtime_{t}ms_raw.csv')):
            gt[r['fen']]=float(r['ground_truth_win_pct_0_100'])
            pred.setdefault(r['fen'],{})[t]=float(r['pred_win_pct_0_100'])
    fens=[r['fen'] for r in csv.DictReader(open('/home/claude/cz/paper_legacy/data/mixed_workload_pool.csv'))]
    P=np.array([[pred[f][t] for t in T] for f in fens]); G=np.array([gt[f] for f in fens])
    return fens,P,G
def load_grid(path,fens):
    d={}
    for r in csv.DictReader(open(path)):
        v=int(r['value'])
        p = wp(v) if r['type']=='cp' else (100.0 if v>0 else 0.0)
        d[(r['fen'],int(r['t']))]=(p,int(r['depth']))
    return d
def best_t_idx(E):  # shortest among ties
    m=E.min(axis=1,keepdims=True)
    return np.argmax(np.isclose(E,m,atol=1e-9),axis=1)
def fen_to_game():
    import chess.pgn
    txt=open('/home/claude/cz/paper_legacy/data/lichess_rapid_1200_1600.pgn').read()
    pg=io.StringIO(txt); m={}; gi=0
    while True:
        g=chess.pgn.read_game(pg)
        if g is None: break
        b=g.board(); m.setdefault(b.fen(),gi)
        for mv in g.mainline_moves():
            b.push(mv); m.setdefault(b.fen(),gi)
        gi+=1
    return m,gi
def features(fens):
    import chess
    from app.utils.position_complexity import _extract_features
    X=[]
    for f in fens:
        ft=_extract_features(chess.Board(f))
        X.append([ft.material_imbalance, ft.capture_count, ft.legal_moves, ft.total_material, ft.white_forcing+ft.black_forcing, int(ft.is_check), int(ft.is_endgame)])
    return np.array(X,dtype=float)
FEAT=['imbalance','capture_count','legal_moves','total_material','forcing_total','is_check','is_endgame']
