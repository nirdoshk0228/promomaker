"""Train the learned ranker and report an honest held-out evaluation.

  python scripts/train_ranker.py data/sample_show data/sample_answer_key.json --out data/model/ranker.json

Cross-validation: episodes are split into 5 folds (episode number mod 5). For each fold the model is trained
only on candidates from the other 28 episodes and judged on the 7 it never saw, so every episode is scored
by a model that did not learn from it. The shipped model is then trained on all episodes.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from promofinder.candidates import Config
from promofinder.evaluate import evaluate
from promofinder.learn import fit, training_rows, featurize
from promofinder.parser import load_show_from_dir
from promofinder.pipeline import score_all
from promofinder.selection import select

def per_episode_result(cands, episodes, k=3):
    per = {}
    for ep in episodes:
        picks = select(cands, k=k, max_overlap=0.3, episode=ep, only_episode=True)
        per[ep] = [{"rank": r, "id": str(r), "type": c.kind,
                    "segments": [{"episode": s.episode, "start": s.start, "end": s.end} for s in c.segments]}
                   for r, c in enumerate(picks, 1)]
    return {"promos": [], "per_episode": per}

def metrics(cands, episodes, key):
    return evaluate(per_episode_result(cands, episodes), key)["per_episode"]

def merge(parts):
    tot = sum(p["episodes_evaluated"] for p in parts)
    keys = ["hit@1_strict", "hit@3_strict", "hit@1_loose", "hit@3_loose", "top1_trap_rate"]
    out = {k: round(sum(p[k] * p["episodes_evaluated"] for p in parts) / tot, 3) for k in keys}
    # edge metrics are averaged over episodes that have a single-clip gold promo; weight by fold size (approx.)
    for k in ("hook_start_within_3s", "cliff_end_within_3s"):
        vals = [p[k] for p in parts if p[k] is not None]
        out[k] = round(sum(vals) / len(vals), 3)
    out["episodes"] = tot
    return out

ap = argparse.ArgumentParser()
ap.add_argument("show_dir"); ap.add_argument("answer_key")
ap.add_argument("--out", default="data/model/ranker.json")
ap.add_argument("--C", type=float, default=0.3)
ap.add_argument("--title", default="Naukrani Nahi, Rajkumari Hoon Main")
a = ap.parse_args()
t0 = time.time()
show = load_show_from_dir(a.show_dir, title=a.title)
key = json.load(open(a.answer_key, encoding="utf-8"))
idxs, cands, _ = score_all(show, Config())
eps = [e.number for e in show.episodes]
folds = [{e for e in eps if e % 5 == f} for f in range(5)]

res = {"rules": [], "learned": [], "blend": []}
for f, test in enumerate(folds):
    train = set(eps) - test
    X, y, names = training_rows(cands, key, train)
    model = fit(X, y, names, C=a.C)
    for c in cands:
        c.score = c.scores["rules"]
    res["rules"].append(metrics(cands, sorted(test), key))
    probs = {id(c): model.prob(c) for c in cands}
    for c in cands:
        c.score = probs[id(c)]
    res["learned"].append(metrics(cands, sorted(test), key))
    for c in cands:
        c.score = 0.5 * c.scores["rules"] + 0.5 * probs[id(c)]
    res["blend"].append(metrics(cands, sorted(test), key))

summary = {k: merge(v) for k, v in res.items()}
print(f"Held-out evaluation (5 folds by episode, C={a.C}):")
cols = ["hit@3_strict", "hit@1_strict", "hit@1_loose", "hook_start_within_3s", "cliff_end_within_3s", "top1_trap_rate"]
print(f"{'':10}" + "".join(f"{c:>22}" for c in cols))
for k, m in summary.items():
    print(f"{k:10}" + "".join(f"{m[c]:>22}" for c in cols))

X, y, names = training_rows(cands, key)
model = fit(X, y, names, C=a.C, meta={
    "trained_on": show.title, "episodes": len(eps), "rows": len(X), "positives": sum(y), "C": a.C,
    "cv": summary, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")})
model.save(a.out)
print(f"\nSaved {a.out}: {len(X)} candidates, {sum(y)} positive. Top weights:")
for w in model.top_weights(10):
    print(f"  {w['feature']:<28} {w['weight']:+.3f}")
print(f"done in {time.time()-t0:.1f}s")
