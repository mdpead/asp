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


# --- completion format ---


def _record():
    return list(synth.generate(1, tier="easy", seed=42, inputs_per_fn=4))[0]


def test_completion_wraps_trace_and_answer_in_markers():
    """The markers are the contract between the formatter and the reward path.

    Round-tripping format through extract cannot catch a change here, since moving both
    together stays self-consistent while breaking anything trained on the old shape.
    """
    r = _record()
    _, completion = synth.format_output_task(r, include_trace=True)

    assert completion.startswith(synth.THINK_OPEN)
    assert completion.endswith(synth.ANSWER_CLOSE)
    trace = completion.split(synth.THINK_OPEN)[1].split(synth.THINK_CLOSE)[0]
    answer = completion.split(synth.ANSWER_OPEN)[1].split(synth.ANSWER_CLOSE)[0]
    assert trace == r["trace"]
    assert answer == repr(r["result"])


def test_untraced_completion_has_no_thinking_markers():
    """RL generates untraced, so the prompt must not imply a trace the reward ignores."""
    r = _record()
    _, completion = synth.format_output_task(r, include_trace=False)

    assert synth.THINK_OPEN not in completion
    assert completion == f"{synth.ANSWER_OPEN}{r['result']!r}{synth.ANSWER_CLOSE}"


def test_markers_cannot_occur_in_generated_source():
    """Why the delimiters are pipe-guarded tokens rather than "#" comments: a comment
    marker could in principle be generated, and would split the answer in the wrong place."""
    for r in synth.generate(200, tier="easy", seed=11, inputs_per_fn=4):
        assert "<|" not in r["source"]
        assert "<|" not in r["trace"]


# --- extract_answer ---


def test_extract_answer_reads_a_complete_rollout():
    r = _record()
    _, completion = synth.format_output_task(r, include_trace=True)
    assert synth.extract_answer(completion) == repr(r["result"])


def test_extract_answer_recovers_a_rollout_truncated_before_the_closing_marker():
    """A rollout that hit its token budget mid-answer still has a usable first line;
    returning the whole tail instead would never parse."""
    r = _record()
    _, completion = synth.format_output_task(r, include_trace=True)
    truncated = completion.replace(synth.ANSWER_CLOSE, "")

    assert synth.extract_answer(truncated) == repr(r["result"])


def test_extract_answer_stops_at_the_first_line_when_generation_runs_on():
    r = _record()
    _, completion = synth.format_output_task(r, include_trace=True)
    runaway = completion.replace(synth.ANSWER_CLOSE, "\nand then some more text")

    assert synth.extract_answer(runaway) == repr(r["result"])


def test_extract_answer_is_none_when_no_answer_was_reached():
    """Distinct from a wrong answer: the checker must score it False rather than raise."""
    assert synth.extract_answer(f"{synth.THINK_OPEN}a = 4") is None
    assert synth.extract_answer("") is None
