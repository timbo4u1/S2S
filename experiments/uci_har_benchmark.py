"""
S2S Benchmark v3 — stable training, correct field names.
"""
import os, sys, json, math, random, time, argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DOMAINS = ['PRECISION','SOCIAL','LOCOMOTION','DAILY_LIVING','SPORT']
DOMAIN_TO_IDX = {d:i for i,d in enumerate(DOMAINS)}
random.seed(42)

def softmax(x):
    m = max(x)
    e = [math.exp(max(-60.0, min(60.0, v-m))) for v in x]
    s = sum(e)
    if s == 0 or s != s:  # s==0 or NaN
        return [1.0/len(x)] * len(x)
    return [v/s for v in e]

def clip(x,lo,hi): return max(lo,min(hi,x))

class MLP:
    """Stable MLP with per-parameter adaptive lr (AdaGrad-style)."""
    def __init__(self, input_dim, hidden=32, output=5, lr=0.01):
        self.lr = lr
        s1 = math.sqrt(1.0/input_dim)
        s2 = math.sqrt(1.0/hidden)
        self.W1 = [[random.uniform(-s1,s1) for _ in range(input_dim)] for _ in range(hidden)]
        self.b1 = [0.0]*hidden
        self.W2 = [[random.uniform(-s2,s2) for _ in range(hidden)] for _ in range(output)]
        self.b2 = [0.0]*output
        # AdaGrad accumulators
        self.gW1 = [[1e-8]*input_dim for _ in range(hidden)]
        self.gb1 = [1e-8]*hidden
        self.gW2 = [[1e-8]*hidden for _ in range(output)]
        self.gb2 = [1e-8]*output

    def forward(self, x):
        h = [max(0.0, min(50.0, sum(self.W1[j][i]*x[i] for i in range(len(x)))+self.b1[j]))
             for j in range(len(self.b1))]
        logits = [sum(self.W2[k][j]*h[j] for j in range(len(h)))+self.b2[k]
                  for k in range(len(self.b2))]
        return h, softmax(logits)

    def backward(self, x, h, probs, label):
        no, nh = len(self.b2), len(self.b1)
        dl = [clip(p - (1.0 if i==label else 0.0), -2, 2) for i,p in enumerate(probs)]
        for k in range(no):
            for j in range(nh):
                g = dl[k]*h[j]
                self.gW2[k][j] += g*g
                self.W2[k][j] -= self.lr/math.sqrt(self.gW2[k][j])*g
            self.gb2[k] += dl[k]*dl[k]
            self.b2[k] -= self.lr/math.sqrt(self.gb2[k])*dl[k]
        dh = [clip(sum(self.W2[k][j]*dl[k] for k in range(no)),-2,2)*(1 if h[j]>0 else 0)
              for j in range(nh)]
        for j in range(nh):
            for i in range(len(x)):
                g = dh[j]*x[i]
                self.gW1[j][i] += g*g
                self.W1[j][i] -= self.lr/math.sqrt(self.gW1[j][i])*g
            self.gb1[j] += dh[j]*dh[j]
            self.b1[j] -= self.lr/math.sqrt(self.gb1[j])*dh[j]

    def _clip_weights(self, max_norm=5.0):
        """Clip weight matrix norms to prevent explosion."""
        for row in self.W1:
            n = math.sqrt(sum(v*v for v in row))
            if n > max_norm:
                s = max_norm / n
                for i in range(len(row)): row[i] *= s
        for row in self.W2:
            n = math.sqrt(sum(v*v for v in row))
            if n > max_norm:
                s = max_norm / n
                for i in range(len(row)): row[i] *= s

    def train_epoch(self, samples):
        random.shuffle(samples)
        loss = 0.0
        nan_count = 0
        for feat,label,_ in samples:
            h,probs = self.forward(feat)
            step_loss = -math.log(max(probs[label],1e-10))
            if step_loss != step_loss:  # NaN check
                nan_count += 1
                continue
            loss += step_loss
            self.backward(feat,h,probs,label)
        self._clip_weights(max_norm=5.0)
        if nan_count > 0:
            print(f'    [warn] {nan_count} NaN losses skipped this epoch')
        return loss/max(len(samples)-nan_count,1)

    def evaluate(self, samples):
        correct=0; pc={d:0 for d in DOMAINS}; pt={d:0 for d in DOMAINS}
        for feat,label,_ in samples:
            _,probs = self.forward(feat)
            pred = probs.index(max(probs))
            pt[DOMAINS[label]] += 1
            if pred==label: correct+=1; pc[DOMAINS[label]]+=1
        acc = correct/max(len(samples),1)
        f1 = {d: pc[d]/max(pt[d],1) for d in DOMAINS}
        return acc, sum(f1.values())/len(f1), f1

def featurize(rec):
    domain = rec.get('domain','')
    if domain not in DOMAIN_TO_IDX: return None
    jerk    = float(rec.get('jerk_p95_ms3', 0))
    coupling= float(rec.get('imu_coupling_r', 0))
    score   = float(rec.get('physics_score', rec.get('physical_law_score', 0)))
    tier    = rec.get('physics_tier', rec.get('tier','REJECTED'))
    laws    = rec.get('physics_laws_passed', rec.get('laws_passed',[]))
    tier_v  = {'GOLD':1.0,'SILVER':0.67,'BRONZE':0.33,'REJECTED':0.0}.get(tier,0.0)
    jerk_log= math.log1p(jerk)/math.log1p(20000)   # stable log norm
    feat = [
        clip(jerk_log, 0, 1),
        clip(coupling, -1, 1)*0.5+0.5,  # shift to 0-1
        score/100.0,
        tier_v,
        len(laws)/7.0,
    ]
    return feat, DOMAIN_TO_IDX[domain], score

def load_data(path):
    samples = []
    for root,_,files in os.walk(path):
        for f in files:
            if not f.endswith('.json'): continue
            try:
                r = json.load(open(os.path.join(root,f)))
                s = featurize(r)
                if s: samples.append(s)
            except: pass
    return samples

def run(dataset_dir, output_path, epochs=30):
    print("S2S Benchmark v3 — AdaGrad, stable training")
    print("="*50)
    data = load_data(dataset_dir)
    print(f"Loaded {len(data)} samples")

    random.shuffle(data)
    n_test = int(len(data)*0.2)
    test  = data[:n_test]
    train = data[n_test:]
    cert  = [(f,l,s) for f,l,s in train if s >= 60]
    print(f"Train all: {len(train)}  Certified: {len(cert)} ({100*len(cert)/len(train):.1f}%)  Test: {len(test)}")

    results = {}
    for cond, td in [("A_all_data", train), ("B_certified", cert)]:
        print(f"\n{'─'*40}")
        print(f"Condition {cond}  n={len(td)}")
        m = MLP(input_dim=5, hidden=32, output=5, lr=0.01)
        t0 = time.time()
        for ep in range(epochs):
            loss = m.train_epoch(td)
            if (ep+1) % 10 == 0:
                acc, f1, _ = m.evaluate(test)
                print(f"  Epoch {ep+1:3d}/{epochs}  loss={loss:.4f}  acc={acc:.3f}  f1={f1:.3f}")
        acc, f1, pf1 = m.evaluate(test)
        print(f"  Final: {acc:.4f} ({acc*100:.1f}%)  F1={f1:.4f}  [{time.time()-t0:.0f}s]")
        results[cond] = {"accuracy":round(acc,4),"macro_f1":round(f1,4),
                         "per_domain_f1":{k:round(v,4) for k,v in pf1.items()},
                         "train_n":len(td),"test_n":len(test)}

    print(f"\n{'='*50}  SUMMARY")
    for c,r in results.items():
        print(f"  {c}: {r['accuracy']:.4f} acc  {r['macro_f1']:.4f} f1  n={r['train_n']}")

    if 'A_all_data' in results and 'B_certified' in results:
        diff = results['B_certified']['accuracy'] - results['A_all_data']['accuracy']
        print(f"\n  Certification effect: {'+' if diff>=0 else ''}{diff*100:.2f}%")

    out = {"experiment":"s2s_v3","timestamp":time.strftime("%Y-%m-%dT%H:%M:%S"),
           "conditions":results,
           "note":"AdaGrad optimizer, 5 features (jerk_log, coupling, score, tier, laws_passed)"}
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path,'w') as f: json.dump(out,f,indent=2)
    print(f"\nSaved to {output_path}")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='s2s_dataset/')
    p.add_argument('--out', default='experiments/results.json')
    p.add_argument('--epochs', type=int, default=30)
    p.parse_args()
    args = p.parse_args()
    run(args.dataset, args.out, args.epochs)
