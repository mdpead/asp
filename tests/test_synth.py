"""Synthetic execution-reasoning task generation."""

import pytest

from src import synth


@pytest.mark.parametrize("inputs_per_fn", [0, 1])
def test_generate_rejects_too_few_inputs(inputs_per_fn):
    """Fewer than two inputs can never satisfy the distinctness filter.

    Without this check the while loop rejects every candidate and spins forever, which is
    far harder to diagnose than an exception.
    """
    with pytest.raises(ValueError, match="inputs_per_fn must be at least 2"):
        list(synth.generate(1, tier="easy", inputs_per_fn=inputs_per_fn))


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_generate_produces_solvable_records(tier):
    """Every record must carry a result the checker accepts, for each difficulty tier."""
    records = list(synth.generate(6, tier=tier, seed=0, inputs_per_fn=4))
    assert len(records) == 6
    for r in records:
        _, completion = synth.format_output_task(r, include_trace=True)
        assert synth.check_output_answer(r, completion), f"{tier}: own answer rejected"
        assert r["tier"] == tier


def test_generated_functions_depend_on_their_arguments():
    """The filter exists so output prediction cannot be memorised from the source alone."""
    records = list(synth.generate(24, tier="easy", seed=3, inputs_per_fn=4))
    by_source = {}
    for r in records:
        by_source.setdefault(r["source"], set()).add(repr(r["result"]))
    assert any(len(v) > 1 for v in by_source.values()), "no function varied with its input"
