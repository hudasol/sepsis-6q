"""Draw the eight qubit entangling feature map as a Qiskit circuit figure in the pink theme.

The angle labels are abbreviated (2x_i for the single qubit phase, phi_ij for the pair phase) so the picture stays
readable; the gate order is checked against `sepsis6q.quantum.parameterized_feature_map` before drawing.
Writes results/feature_map_circuit.png.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit import Parameter  # noqa: E402

from sepsis6q.quantum import N_QUBITS, linear_pairs, parameterized_feature_map  # noqa: E402

PEARL, INK, ROSE, BLUSH, PETAL = "#FBF4F2", "#4A2F38", "#A3274F", "#E9A6BC", "#F2D3DC"
STYLE = {
    "backgroundcolor": PEARL, "linecolor": INK, "textcolor": INK, "gatetextcolor": INK,
    "gatefacecolor": PETAL, "edgecolor": INK, "fontsize": 13, "subfontsize": 10,
    "displaycolor": {"h": (BLUSH, INK), "p": (PETAL, INK), "cx": (ROSE, "#FFFFFF")},
}


def labelled_map(n, reps, entangled):
    """Same gates in the same order as the real map, with abbreviated angle labels."""
    qc = QuantumCircuit(n)
    single = [Parameter(f"2x{i}") for i in range(n)]
    pair = {(i, j): Parameter(f"phi{i}{j}") for i, j in linear_pairs(n)}
    for _ in range(reps):
        qc.h(range(n))
        for i in range(n):
            qc.p(single[i], i)
        if entangled:
            for i, j in linear_pairs(n):
                qc.cx(i, j)
                qc.p(pair[(i, j)], j)
                qc.cx(i, j)
    return qc


def signature(qc):
    return [(inst.operation.name, tuple(qc.find_bit(q).index for q in inst.qubits)) for inst in qc.data]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/feature_map_circuit.png")
    args = ap.parse_args()
    reps, n = 1, N_QUBITS
    drawn = labelled_map(n, reps, True)
    assert signature(drawn) == signature(parameterized_feature_map(n, reps, True)), "figure differs from the map"
    fig = drawn.draw("mpl", style=STYLE, fold=-1, plot_barriers=False)
    fig.set_facecolor(PEARL)
    fig.suptitle("Entangling feature map U(x), 8 qubits, 1 repetition (ZZ map, linear entanglement)",
                 color=ROSE, fontsize=15, y=0.93)
    fig.text(0.5, 0.12,
             "x_i = b · π · u_i for scaled feature u_i and bandwidth b;  single qubit phase P(2 x_i);  "
             "pair phase φ_ij = 2 (π − x_i)(π − x_j) between neighbouring qubits.\n"
             "Kernel: k(x, y) = |⟨0|U(y)† U(x)|0⟩|², the probability of reading all zeros after U(x) "
             "followed by the inverse of U(y).  The product map control drops every CX, P, CX block.",
             ha="center", va="top", color=INK, fontsize=11)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=PEARL)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
