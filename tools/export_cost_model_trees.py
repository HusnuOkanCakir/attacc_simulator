#!/usr/bin/env python3
"""Export trained HistGBR pkl models to a compact binary format for C++ inference.

Each .bin file contains all 8 target ensembles for one route.
The C++ MlCostModel class loads these files and runs tree inference directly,
replacing the nearest-neighbor CSV lookup in CostTable::estimate().

Usage:
  python tools/export_cost_model_trees.py \\
    --models cluster_outputs/cost_models_full_energy/gpu_only.pkl \\
             cluster_outputs/cost_models_full_energy/lpddr5_pim_bank.pkl \\
    --out-dir cluster_outputs/cost_models_full_energy

Binary layout per file:
  [header]  magic(u32) version(u32) n_targets(u32)
  [target]  name_len(u32) name(bytes) base(f64) n_trees(u32)
  [tree]    n_nodes(u32)
  [node]    is_leaf(u8) feature_idx(i32) threshold(f64) left(i32) right(i32) value(f64)
            = 29 bytes per node
"""
import argparse
import pickle
import struct
from pathlib import Path

MAGIC   = 0xC05774EE
VERSION = 1

# Struct format per node: native byte order, no padding
# B=uint8, i=int32, d=float64 → 1+4+8+4+4+8 = 29 bytes
NODE_FMT = "=Bidiid"
NODE_SIZE = struct.calcsize(NODE_FMT)  # B+i+d+i+i+d = 1+4+8+4+4+8 = 29


def export_bundle(bundle: dict, out_path: Path) -> None:
    targets = list(bundle["target_names"])

    with open(out_path, "wb") as f:
        # File header
        f.write(struct.pack("=III", MAGIC, VERSION, len(targets)))

        for tname in targets:
            model = bundle["estimators"][tname]
            base  = float(model._baseline_prediction.flat[0])
            # _predictors is a list-of-lists; each inner list has one TreePredictor
            # (one per class, but regression has only one class)
            trees = [p[0] for p in model._predictors]

            # Target header
            name_b = tname.encode("utf-8")
            f.write(struct.pack("=I", len(name_b)))
            f.write(name_b)
            f.write(struct.pack("=dI", base, len(trees)))

            for tree in trees:
                nodes = tree.nodes  # structured numpy array
                f.write(struct.pack("=I", len(nodes)))
                for nd in nodes:
                    is_leaf     = int(nd["is_leaf"])
                    feature_idx = int(nd["feature_idx"]) if not is_leaf else -1
                    threshold   = float(nd["num_threshold"])
                    left        = int(nd["left"])
                    right       = int(nd["right"])
                    value       = float(nd["value"])
                    f.write(struct.pack(NODE_FMT,
                                        is_leaf, feature_idx, threshold,
                                        left, right, value))

    # Summary
    total_nodes = sum(
        len(p[0].nodes)
        for tname in targets
        for p in bundle["estimators"][tname]._predictors
    )
    size_kb = out_path.stat().st_size / 1024
    print(f"[export] {bundle['route']:<22}  targets={len(targets)}"
          f"  trees/target={len(trees)}"
          f"  total_nodes={total_nodes}"
          f"  {size_kb:.0f} KB  ->  {out_path}")


def verify_roundtrip(bundle: dict, out_path: Path) -> None:
    """Quick Python-side roundtrip check: re-read the binary and compare one prediction."""
    import struct, numpy as np

    # Read back and predict manually
    with open(out_path, "rb") as f:
        magic, version, n_targets = struct.unpack("=III", f.read(12))
        assert magic == MAGIC, f"bad magic {magic:#x}"
        ensembles = {}
        for _ in range(n_targets):
            name_len, = struct.unpack("=I", f.read(4))
            tname = f.read(name_len).decode()
            base, n_trees = struct.unpack("=dI", f.read(12))
            trees = []
            for _ in range(n_trees):
                n_nodes, = struct.unpack("=I", f.read(4))
                nodes = []
                for _ in range(n_nodes):
                    nd = struct.unpack(NODE_FMT, f.read(NODE_SIZE))
                    nodes.append(nd)  # (is_leaf, feature_idx, thresh, left, right, value)
                trees.append(nodes)
            ensembles[tname] = (base, trees)

    def tree_predict(nodes, lin, lout, bs):
        n = 0
        while not nodes[n][0]:  # not is_leaf
            feat = [lin, lout, bs][nodes[n][1]]
            n = nodes[n][3] if feat <= nodes[n][2] else nodes[n][4]
        return nodes[n][5]

    def ensemble_predict(base_trees, lin, lout, bs):
        base, trees = base_trees
        return base + sum(tree_predict(t, lin, lout, bs) for t in trees)

    # Compare vs sklearn _raw_predict for a few points
    import pandas as pd
    test_pts = [(512, 2, 1), (2048, 16, 8), (4096, 64, 1)]
    tname = list(bundle["target_names"])[4]  # decode_e2e_ms
    model = bundle["estimators"][tname]
    print(f"  Roundtrip check for '{tname}':")
    for lin, lout, bs in test_pts:
        X = [[float(lin), float(lout), float(bs)]]
        import numpy as np
        sk_pred = float(model._raw_predict(np.array(X)).flat[0])
        my_pred = ensemble_predict(ensembles[tname], lin, lout, bs)
        diff = abs(sk_pred - my_pred)
        status = "OK" if diff < 1e-6 else "MISMATCH"
        print(f"    Lin={lin:4d} Lout={lout:3d} bs={bs}  "
              f"sklearn={sk_pred:.6f}  binary={my_pred:.6f}  diff={diff:.2e}  {status}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True,
                    help="Paths to .pkl model files.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Directory to write *_trees.bin files into.")
    ap.add_argument("--verify", action="store_true", default=True,
                    help="Run Python roundtrip check after export (default: on).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for pkl_path in args.models:
        with open(pkl_path, "rb") as f:
            bundle = pickle.load(f)
        out = args.out_dir / f"{bundle['route']}_trees.bin"
        export_bundle(bundle, out)
        if args.verify:
            verify_roundtrip(bundle, out)


if __name__ == "__main__":
    main()
