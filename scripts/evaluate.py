"""CLI: python scripts/evaluate.py data/sample_show data/sample_answer_key.json [--llm]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from promofinder.parser import load_show_from_dir
from promofinder.pipeline import analyze
from promofinder.candidates import Config
from promofinder.evaluate import evaluate, baseline_result, summarize

ap = argparse.ArgumentParser()
ap.add_argument("show_dir"); ap.add_argument("answer_key")
ap.add_argument("--llm", action="store_true", help="use the LLM re-ranker configured via env vars")
ap.add_argument("--json", help="write full metrics JSON here")
a = ap.parse_args()
show = load_show_from_dir(a.show_dir)
key = json.load(open(a.answer_key, encoding="utf-8"))
reranker = None
if a.llm:
    from promofinder.llm import make_reranker_from_env
    reranker = make_reranker_from_env()
res = analyze(show, Config(), reranker=reranker)
m = evaluate(res, key)
print(f"Show: {show.title} — {len(show.episodes)} episodes, {res['stats']['candidates']} candidates in {res['stats']['total_seconds']}s")
print("\n== PromoFinder" + (" + LLM" if reranker else " (heuristic)"))
for line in summarize(m): print("  " + line)
for kind in ("episode_end", "random"):
    b = evaluate(baseline_result(show, kind), key)
    e = b["per_episode"]
    print(f"\n== Baseline: {kind}\n  top pick hit@1 strict {e['hit@1_strict']} loose {e['hit@1_loose']}; hook start within 3s {e['hook_start_within_3s']}; trap rate {e['top1_trap_rate']}")
if a.json:
    json.dump({"promofinder": m}, open(a.json, "w"), indent=2)
