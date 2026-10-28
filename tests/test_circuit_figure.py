"""The circuit figure must show the same gate order as the map used for the resource counts."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("qiskit")

from sepsis6q.quantum import parameterized_feature_map  # noqa: E402

spec = importlib.util.spec_from_file_location("draw_circuit", Path(__file__).parents[1] / "scripts" / "draw_circuit.py")
draw_circuit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(draw_circuit)


@pytest.mark.parametrize("reps", [1, 2])
@pytest.mark.parametrize("entangled", [True, False])
def test_labelled_map_has_the_same_gates_as_the_real_map(reps, entangled):
    drawn = draw_circuit.labelled_map(8, reps, entangled)
    real = parameterized_feature_map(8, reps, entangled)
    assert draw_circuit.signature(drawn) == draw_circuit.signature(real)
