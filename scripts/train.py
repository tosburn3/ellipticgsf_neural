"""
train.py -- train the neural preconditioner for one m (Sec. IV B).

    python train.py --m 6                          # start from the shipped network of the nearest m (here m = 7)
    python train.py --m 6 --init ../ellipticgsf_neural/models/gnn_m6.pt --epochs 10
    python train.py --m 6 --install                # also copy the result into ellipticgsf_neural/models/

The recipe is the one that produced the shipped networks, settings.TRAINING: 17 training orbits
(M omega / m = 0.03 ... 0.07, r0 = 5.89 ... 10.36), one weight update per orbit with the GMRES restart-cycle
loss, Adam (learning rate 3e-4, linear warm-up over 8 updates, gradient-norm clipping at 1), each epoch
scored by the total GMRES iterations on held-out orbits at r0 = 6, 8.376 and 6.812; an epoch that does not
cut them by 0.5% is reverted and the learning rate halved, and training stops after 3 such epochs in a
row. Start from a trained network of a neighbouring m rather than from scratch: m = 6 started from the
m = 7 network solves the r0 = 6 orbit that m = 6 started from m = 2 never did. Expect 20-40 minutes per
epoch on 8 cores, 6-22 epochs.

Output in --out (default runs/m<m>/): gnn_m<m>.pt (best), gnn_m<m>_last.pt, training.jsonl, train.log.
"""

import settings  # noqa: I001  first: sets the thread counts, makes the package importable

import argparse
import shutil
from dataclasses import replace
from pathlib import Path

from ellipticgsf_neural import neural

# ------------------------------------------------------------------ SETTINGS
M = 2 # m-mode
# (the training recipe is settings.TRAINING; --epochs, --learning-rate and --patience override it)
# ----------------------------------------------------------------------------


def nearest_shipped_network(m: int) -> Path:
    """The shipped network of the nearest other m, the larger m first (the harder neighbour transfers better)."""
    shipped = sorted(int(p.stem.split("_m")[1]) for p in neural.MODEL_DIR.glob("gnn_m*.pt"))
    others = [k for k in shipped if k != m] or shipped
    k = min(others, key=lambda k: (abs(k - m), -k))
    return neural.MODEL_DIR / f"gnn_m{k}.pt"


def main():
    training = neural.TrainingParameters(**settings.TRAINING)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, default=M, choices=sorted(settings.GRID))
    ap.add_argument("--init", help="starting checkpoint (default: the shipped network of the nearest m)")
    ap.add_argument("--out", help="output folder (default runs/m<m>)")
    ap.add_argument("--epochs", type=int, default=training.max_epochs)
    ap.add_argument("--learning-rate", type=float, default=training.learning_rate)
    ap.add_argument("--patience", type=int, default=training.patience)
    ap.add_argument("--install", action="store_true", help="copy the best network to ellipticgsf_neural/models/")
    args = ap.parse_args()
    training = replace(training, max_epochs=args.epochs, learning_rate=args.learning_rate, patience=args.patience)

    init = Path(args.init) if args.init else nearest_shipped_network(args.m)
    out = Path(args.out or settings.RUNS / f"m{args.m}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.log", "a") as logfile:
        best = neural.train(args.m, init, out, grid_parameters=settings.GRID[args.m], gmres=settings.GMRES,
                            training=training, log=lambda line: (logfile.write(line + "\n"), logfile.flush()))
    print(f"best network: {best}")
    if args.install:
        target = neural.MODEL_DIR / f"gnn_m{args.m}.pt"
        shutil.copyfile(best, target)
        print(f"installed as {target}")


if __name__ == "__main__":
    main()
