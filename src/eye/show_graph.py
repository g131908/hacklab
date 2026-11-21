"""Print the metric graph as a tree, with validation results.  python src/eye/show_graph.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eye.graph import MetricGraph  # noqa: E402
from eye.store import Store  # noqa: E402


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    g = MetricGraph.build(Store.load())
    print("[DEMO DATA] metric graph:", len(g.metrics), "metrics,", len(g.edges), "edges\n")
    for r in g.roots():
        print("\n".join(g.tree(r)))
        print()
    print("decompositions:")
    for d in g.decomps:
        res = "" if d.residual is None else f" residual {d.residual:.1e} over {d.checked:,} points"
        print(f"  [{d.status:<9}] {d.kind:<10} {'*' if d.primary else ' '} {d.text}{res}")
    print("\nagent notes:")
    for line in g.log:
        print("  -", line)
    kinds = {}
    for e in g.edges:
        kinds[e.type] = kinds.get(e.type, 0) + 1
    print("\nedges by type:", kinds)


if __name__ == "__main__":
    main()
