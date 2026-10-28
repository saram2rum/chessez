# 기준값 B(depth 20, 국면마다 해시 초기화)가 측정 탐색의 연장인지 확인 (논문 5.3절)
# 500ms 예산 안에 depth 20을 마친 국면에서, 반복 심화 기록의 depth 20 평가치·노드 수와 기준값 B를 비교한다.
# thesis_revision 폴더 안에서 실행: python3 scripts/check_prefix.py
import json, csv, gzip
B = {r['fen']: r for r in csv.DictReader(open('data/ground_truth_B.csv'))}
n = same = 0
for line in gzip.open('data/trace_500.jsonl.gz', 'rt'):
    d = json.loads(line); tr = d['trace']; b = B[d['fen']]
    t20 = [x for x in tr if x[0] == 20]
    if not t20 or not any(x[0] > 20 for x in tr): continue
    n += 1; t = t20[-1]
    same += (t[1] == int(b['nodes']) and t[2] == b['type'] and str(t[3]) == b['value'])
print(f'{same}/{n} identical')
