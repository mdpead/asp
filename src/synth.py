import ast
import random
import sys

# Values stay small on purpose: the task should be hard because tracing control flow is
# hard, not because a 30M model cannot multiply four-digit numbers. Programs that escape
# the range are rejected after execution rather than clamped, since clamping would mean
# sprinkling modulo noise through the source.
VALUE_MIN, VALUE_MAX = -500, 500
MAX_RESULT_CHARS = 80

# Loop variables are named per nesting depth so an inner loop never shadows its parent.
LOOP_VARS = ["i", "j", "k"]
INT_VARS = ["a", "b", "c", "d"]
LIST_VARS = ["xs", "ys"]

# A nested loop whose bound is len() of a list the body appends to still terminates, but it
# can run for hundreds of lines. Capping executed lines bounds both the trace length and the
# work spent generating a sample that would be discarded for being too long anyway. Retune
# these against your own tokenizer: the budget that matters is trace tokens, and RL needs
# headroom above the pretraining length for responses to grow into.
TIERS = {
    # max_depth 0 is straight-line code: no branches, no loops, so the answer is a short
    # chain of arithmetic. This is the tier a freshly pretrained model can actually do.
    "easy": dict(n_stmts=(3, 5), max_depth=0, n_ints=2, n_lists=0, loop_range=(2, 3), max_executed=25),
    "medium": dict(n_stmts=(4, 7), max_depth=1, n_ints=2, n_lists=1, loop_range=(2, 4), max_executed=45),
    "hard": dict(n_stmts=(6, 10), max_depth=2, n_ints=3, n_lists=2, loop_range=(2, 5), max_executed=70),
}


class _TooLong(Exception):
    """Raised from inside the tracer to abandon an over-long run mid-execution."""


def _gen_expr(rng, scope, depth=0):
    # Depth here is expression nesting, unrelated to statement nesting; two levels keeps
    # lines readable while still forcing more than one arithmetic step per line.
    choices = ["int_var", "literal"]
    if scope["lists"]:
        choices += ["index", "len"]
    if depth < 2:
        choices += ["binop", "binop"]

    kind = rng.choice(choices)
    if kind == "int_var":
        return rng.choice(scope["ints"])
    if kind == "literal":
        return str(rng.randint(-5, 9))
    if kind == "len":
        return f"len({rng.choice(scope['lists'])})"
    if kind == "index":
        # Lists are only ever appended to, never emptied, so they are always non-empty and
        # the modulo keeps the index in bounds however long the list has grown.
        name = rng.choice(scope["lists"])
        return f"{name}[{rng.choice(scope['ints'])} % len({name})]"
    left = _gen_expr(rng, scope, depth + 1)
    right = _gen_expr(rng, scope, depth + 1)
    return f"({left} {rng.choice(['+', '-', '*'])} {right})"


def _gen_cond(rng, scope):
    op = rng.choice(["<", ">", "<=", ">=", "==", "!="])
    return f"{_gen_expr(rng, scope)} {op} {rng.randint(-5, 15)}"


def _gen_stmt(rng, scope, depth, cfg, indent):
    pad = "    " * indent
    kinds = ["assign", "assign", "augmented"]
    if scope["lists"]:
        kinds.append("append")
    if depth < cfg["max_depth"]:
        kinds += ["if", "for"]

    kind = rng.choice(kinds)
    if kind == "assign":
        return [f"{pad}{rng.choice(scope['ints'])} = {_gen_expr(rng, scope)}"]
    if kind == "augmented":
        op = rng.choice(["+=", "-=", "*="])
        return [f"{pad}{rng.choice(scope['ints'])} {op} {_gen_expr(rng, scope)}"]
    if kind == "append":
        return [f"{pad}{rng.choice(scope['lists'])}.append({_gen_expr(rng, scope)})"]
    if kind == "if":
        lines = [f"{pad}if {_gen_cond(rng, scope)}:"]
        lines += _gen_block(rng, scope, depth + 1, cfg, indent + 1)
        if rng.random() < 0.4:
            lines.append(f"{pad}else:")
            lines += _gen_block(rng, scope, depth + 1, cfg, indent + 1)
        return lines

    var = LOOP_VARS[depth]
    # range() over a literal or over len(list). Both evaluate their bound once, before the
    # first iteration, so a body that appends to the list it is iterating still terminates —
    # which "for x in xs: xs.append(...)" would not. That is why the len() form is used
    # instead of iterating the list directly.
    if scope["lists"] and rng.random() < 0.4:
        bound = f"len({rng.choice(scope['lists'])})"
    else:
        bound = str(rng.randint(*cfg["loop_range"]))
    lines = [f"{pad}for {var} in range({bound}):"]
    inner = dict(scope, ints=scope["ints"] + [var])
    lines += _gen_block(rng, inner, depth + 1, cfg, indent + 1)
    return lines


def _gen_block(rng, scope, depth, cfg, indent):
    n = rng.randint(1, 2) if depth else rng.randint(*cfg["n_stmts"])
    lines = []
    for _ in range(n):
        lines += _gen_stmt(rng, scope, depth, cfg, indent)
    return lines


def _gen_function(rng, cfg):
    """Build a function whose parameters are the task's input slot."""
    ints = INT_VARS[: cfg["n_ints"]]
    lists = LIST_VARS[: cfg["n_lists"]]
    scope = {"ints": ints, "lists": lists}
    params = ints + lists

    lines = [f"def f({', '.join(params)}):"]
    lines += _gen_block(rng, scope, 0, cfg, 1)
    returned = rng.sample(params, k=min(2, len(params)))
    lines.append(f"    return {', '.join(returned)}")
    return "\n".join(lines), [(n, "list" if n in lists else "int") for n in params]


def _sample_args(rng, params):
    args = []
    for _, kind in params:
        if kind == "int":
            args.append(rng.randint(-3, 9))
        else:
            args.append([rng.randint(0, 9) for _ in range(rng.randint(2, 5))])
    return args


def _snapshot(frame):
    # Lists are copied because append mutates in place, which an identity-sharing snapshot
    # would compare equal to itself and report as unchanged.
    return {
        k: (list(v) if isinstance(v, list) else v)
        for k, v in frame.f_locals.items()
        if not k.startswith("__")
    }


def _run_traced(source, code_obj, func, args, max_executed):
    """Call func(*args) under a tracer, returning (result, trace_steps, n_executed).

    Returns None if the call raises or any value escapes the allowed range. Rejection
    rather than repair: the generator produces thousands per second, so throwing a sample
    away is cheaper than patching it up.
    """
    src_lines = source.split("\n")
    steps = []
    prev = {}
    overflowed = []

    def tracer(frame, event, arg):
        if frame.f_code is not code_obj:
            return None
        if event == "call":
            # Seed from the bound parameters so the trace opens with the first real
            # statement rather than restating the arguments as if they had changed.
            prev.update(_snapshot(frame))
            return tracer
        if event == "line":
            # Python reports a line event before executing that line, so the state visible
            # here is the result of the previous one.
            current = _snapshot(frame)
            changed = {k: v for k, v in current.items() if prev.get(k, object()) != v}
            for value in changed.values():
                flat = value if isinstance(value, list) else [value]
                if any(isinstance(v, int) and not VALUE_MIN <= v <= VALUE_MAX for v in flat):
                    overflowed.append(True)
            if changed and steps:
                shown = ", ".join(f"{k} = {v!r}" for k, v in sorted(changed.items()))
                steps[-1] = f"{steps[-1]}  ->  {shown}"
            prev.clear()
            prev.update(current)
            steps.append(src_lines[frame.f_lineno - 1].strip())
            if len(steps) > max_executed:
                # Raising here unwinds the traced call itself, so an over-long program is
                # abandoned partway rather than run to completion and then discarded.
                raise _TooLong
        return tracer

    sys.settrace(tracer)
    try:
        result = func(*[list(a) if isinstance(a, list) else a for a in args])
    except Exception:
        return None
    finally:
        sys.settrace(None)

    if overflowed or len(repr(result)) > MAX_RESULT_CHARS:
        return None
    return result, steps, len(steps)


def generate(n, tier="medium", seed=0, inputs_per_fn=4, with_trace=True):
    """Yield n records, sharing each generated function across several sampled inputs."""
    # The distinctness filter below compares results across a function's inputs, so with
    # fewer than two it can never be satisfied: every candidate is rejected and the loop
    # spins forever instead of returning. Fail here rather than hang.
    if inputs_per_fn < 2:
        raise ValueError(
            f"inputs_per_fn must be at least 2, got {inputs_per_fn}. One input cannot show "
            "that the result depends on the arguments, which is what the task requires."
        )
    rng = random.Random(seed)
    cfg = TIERS[tier]
    seen = set()
    produced = 0

    while produced < n:
        source, params = _gen_function(rng, cfg)
        if source in seen:
            continue
        seen.add(source)

        code_obj = compile(source, "<synth>", "exec")
        namespace = {}
        exec(code_obj, namespace)
        func = namespace["f"]
        # compile() above yields the module code; the tracer needs the function's own code
        # object, since that is the frame the body executes in.
        func_code = func.__code__

        batch = []
        for _ in range(inputs_per_fn):
            args = _sample_args(rng, params)
            outcome = _run_traced(source, func_code, func, args, cfg["max_executed"])
            if outcome is None:
                continue
            result, steps, n_executed = outcome
            batch.append(
                {
                    "source": source,
                    "args": args,
                    "result": result,
                    "tier": tier,
                    # Executed-line count is the difficulty signal worth filtering on, and
                    # it varies across inputs to the same function whenever a loop bound
                    # depends on a list argument.
                    "n_executed": n_executed,
                    **({"trace": "\n".join(steps)} if with_trace else {}),
                }
            )

        # A function that ignores its arguments makes output prediction memorisable and
        # input prediction trivially satisfiable by anything, so require that the input
        # actually moves the result.
        if len({repr(r["result"]) for r in batch}) < 2:
            continue

        for record in batch:
            if produced >= n:
                break
            yield record
            produced += 1


def _call_repr(args):
    return f"f({', '.join(repr(a) for a in args)})"


# Single tokens in the trained tokenizer, so a delimiter costs one token rather than the
# nine "# Trace:\n" takes once digits split. They also cannot occur in generated source,
# unlike a "#" comment, so splitting on them can never catch a fragment of the function.
THINK_OPEN, THINK_CLOSE = "<|think|>", "<|/think|>"
ANSWER_OPEN, ANSWER_CLOSE = "<|answer|>", "<|/answer|>"


def _wrap_trace(record, include_trace):
    return f"{THINK_OPEN}{record['trace']}{THINK_CLOSE}" if include_trace else ""


def format_output_task(record, include_trace):
    """Given the function and its arguments, predict the return value."""
    prompt = f"{record['source']}\n\n# What does {_call_repr(record['args'])} return?\n"
    body = _wrap_trace(record, include_trace)
    return prompt, f"{body}{ANSWER_OPEN}{record['result']!r}{ANSWER_CLOSE}"


def format_input_task(record, include_trace):
    """Given the function and a target return value, find arguments that produce it."""
    prompt = f"{record['source']}\n\n# What arguments make f return {record['result']!r}?\n"
    body = _wrap_trace(record, include_trace)
    answer = ", ".join(repr(a) for a in record["args"])
    return prompt, f"{body}{ANSWER_OPEN}{answer}{ANSWER_CLOSE}"


def extract_answer(completion):
    """Pull the answer text from a rollout, or None if the model never reached one."""
    if ANSWER_OPEN not in completion:
        return None
    answer = completion.split(ANSWER_OPEN)[-1]
    # Bounded by the closing marker when the rollout produced one, and by the first line
    # break otherwise — a run that hit its token budget mid-answer still has a usable
    # first line, and returning the whole remaining tail would never parse.
    return answer.split(ANSWER_CLOSE)[0].strip().split("\n")[0].strip()


def check_output_answer(record, completion):
    answer = extract_answer(completion)
    if answer is None:
        return False
    try:
        return ast.literal_eval(answer) == record["result"]
    except (ValueError, SyntaxError):
        return False


def check_input_answer(record, completion):
    """Reward any arguments reaching the target, not just the ones used to build the task.

    literal_eval parses literals only and never executes, so the model's answer cannot run
    code — the only thing executed is the generator's own function, called with the parsed
    values.
    """
    answer = extract_answer(completion)
    if answer is None:
        return False
    try:
        args = ast.literal_eval(f"({answer},)")
    except (ValueError, SyntaxError):
        return False

    namespace = {}
    exec(compile(record["source"], "<synth>", "exec"), namespace)
    try:
        return namespace["f"](*args) == record["result"]
    except Exception:
        return False
