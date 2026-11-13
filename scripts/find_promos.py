"""CLI: find promos in a folder or .zip of episodes.

  python scripts/find_promos.py data/sample_show --top 10 --csv promos.csv --edl promos.edl
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from promofinder.candidates import Config
from promofinder.exporters import to_csv, to_edl
from promofinder.parser import iter_texts_from_upload, load_show_from_dir, parse_show
from promofinder.pipeline import analyze

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("source", help="folder of episode .txt files, or a .zip")
ap.add_argument("--top", type=int, default=10)
ap.add_argument("--min", type=float, default=20); ap.add_argument("--max", type=float, default=60)
ap.add_argument("--no-stitch", action="store_true")
ap.add_argument("--llm", action="store_true", help="re-rank with the LLM configured in env vars")
ap.add_argument("--csv"); ap.add_argument("--edl")
a = ap.parse_args()
if os.path.isdir(a.source):
    show = load_show_from_dir(a.source)
else:
    with open(a.source, "rb") as fh:
        show = parse_show(list(iter_texts_from_upload(os.path.basename(a.source), fh.read())))
reranker = None
if a.llm:
    from promofinder.llm import make_reranker_from_env
    reranker = make_reranker_from_env()
    if reranker is None:
        sys.exit("set PROMOFINDER_LLM_PROVIDER and PROMOFINDER_LLM_API_KEY to use --llm")
res = analyze(show, Config(min_len=a.min, max_len=a.max, top_k=a.top, allow_stitched=not a.no_stitch), reranker=reranker)
print(f"{show.title}: {len(show.episodes)} episodes, {res['stats']['candidates']} cuts scored in {res['stats']['total_seconds']}s\n")
for p in res["promos"]:
    where = " + ".join(f"Ep{s['episode']} {s['start_tc']}-{s['end_tc']}" for s in p["segments"])
    opening = (p.get("opening") or {}).get("text", "")[:70]
    print(f"{p['rank']:>2}. {p['score']:.2f}  {p['duration']:5.1f}s  {where}\n      opens: {opening}")
if a.csv:
    open(a.csv, "w", encoding="utf-8").write(to_csv(res)); print(f"\nwrote {a.csv}")
if a.edl:
    open(a.edl, "w", encoding="utf-8").write(to_edl(res)); print(f"wrote {a.edl}")
