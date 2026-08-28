import warnings

import torch
import torch.nn.functional as F


def make_kv_cache(model, batches, seq_length, device):

    d_h = model.d_h
    num_layers = model.num_layers
    num_kv_heads = model.num_kv_heads
    dtype = next(model.parameters()).dtype

    kv_cache = torch.zeros(
        (num_layers, 2, batches, num_kv_heads, seq_length, d_h), device=device, dtype=dtype
    )

    return kv_cache


def generate(model, tokenizer, input_texts, device, max_new_tokens=None, temperature=0):
    """Continue each prompt, returning (token_ids, completion_mask, logprobs, seq_starts,
    finished).

    `token_ids` is (batch, length): each row is <bos>, the prompt, and its continuation,
    left-padded so every row's real tokens end flush against the same right-hand edge and
    right-padded from wherever that row stopped. `completion_mask` is True exactly where
    the model chose the token — the continuation up to and including its <eos> — and False
    on the prompt and on padding at either end, none of which the model decided.
    `seq_starts` is each row's first real column, which a caller that feeds these ids back
    through the model must pass to see the same context generation saw. `finished[i]` is
    True when row i stopped on <eos> rather than running out of budget, which is the
    difference between a wrong answer and a truncated one.

    `logprobs` is the log-probability the sampling distribution gave the token actually
    drawn, which a policy gradient needs to weigh an updated policy against the one that
    produced the rollout. It is None under a greedy decode: the distribution there is
    degenerate, so every drawn token has log-probability 0, and a caller that trains on
    that would silently compute nonsense ratios. None makes the misuse fail instead.

    `temperature` defaults to 0, meaning greedy: eval wants the model's best guess, and it
    wants it reproducible and independent of how rows were batched, neither of which
    survives sampling. RL passes a temperature explicitly, because a policy gradient needs
    rollouts drawn from the policy rather than one deterministic answer per prompt.

    Token ids rather than text is the useful boundary: decoding is lossy for anything that
    needs to line up with the tokens the model actually emitted, so callers that score or
    train on the rollout work from here and generate_texts decodes on top.
    """
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")

    max_length = model.max_length

    # Encode without special tokens so we control them: <bos> leads, no trailing <eos>
    prompt_ids = tokenizer(input_texts, add_special_tokens=False)["input_ids"]
    prompt_ids = [[tokenizer.bos_token_id] + ids for ids in prompt_ids]

    # A prompt that needs truncating would fill the context outright, leaving nothing to
    # generate, so trimming it only corrupts the input for no gain. Refuse instead.
    over = [len(ids) for ids in prompt_ids if len(ids) > max_length]
    if over:
        raise ValueError(
            f"{len(over)} prompt(s) exceed max_length={max_length} (longest is {max(over)})"
        )
    encoded = tokenizer.pad({"input_ids": prompt_ids}, return_tensors="pt", padding_side="left")

    input_ids = encoded["input_ids"].to(device)
    batch_size, input_length = input_ids.shape

    # Whatever the prompt leaves is the generation budget, optionally capped further
    room = max_length - input_length
    new_tokens = room if max_new_tokens is None else min(max_new_tokens, room)
    if new_tokens < 1:
        # The prompt fills the context, so there is genuinely nothing to generate. Hand
        # back the prompts rather than stealing a token from them to manufacture output.
        warnings.warn(f"prompt fills {input_length} of {max_length} tokens, generating nothing")
    elif max_new_tokens is not None and new_tokens < max_new_tokens:
        warnings.warn(
            f"max_new_tokens reduced from {max_new_tokens} to {new_tokens}; "
            f"the prompt uses {input_length} of {max_length} tokens"
        )

    # Left padding puts every row's real tokens flush against the same right-hand edge, so
    # the batch shares one frontier and each row's pads form a leading block
    seq_starts = (input_length - encoded["attention_mask"].sum(dim=-1)).to(
        device=device, dtype=torch.int32
    )  # (batch,) first real column
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Preallocate the whole buffer, then fill forward from the prompt
    token_ids = F.pad(input_ids, (0, new_tokens), value=tokenizer.pad_token_id)
    completion_mask = torch.zeros_like(token_ids, dtype=torch.bool)
    logprobs = torch.zeros_like(token_ids, dtype=torch.float32) if temperature > 0 else None

    # Framing this function writes and decoding strips, so sampling one would train on a
    # token that never appears in the text being scored. <eos> is deliberately not here:
    # stopping is a real choice, and masking it off is how a model learns to never stop.
    blocked = [tokenizer.bos_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id]
    blocked = torch.tensor([i for i in blocked if i is not None], device=device)

    # Make kv cache
    kv_cache = make_kv_cache(model, batch_size, input_length + new_tokens, device)

    num_cached = 0
    cur_length = input_length
    training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(new_tokens):

            kv_cache_step = kv_cache[..., 0:num_cached, :]
            token_ids_step = token_ids[:, num_cached:cur_length]

            logits, _, new_kv = model(token_ids_step, seq_starts, kv_cache_step)
            kv_cache[..., num_cached:cur_length, :] = new_kv

            # Choose the next token
            if temperature > 0:
                # bf16 carries ~3 significant digits, so softmaxing in it collapses the
                # tail of the vocabulary onto a handful of values and skews the draw
                step_logits = logits[:, -1, :].float()
                step_logits[:, blocked] = -float("inf")
                # One log_softmax rather than a softmax beside it: exp recovers the
                # probabilities multinomial wants, and the log form is what RL scores with
                step_logprobs = F.log_softmax(step_logits / temperature, dim=-1)
                pred_token_ids = torch.multinomial(step_logprobs.exp(), num_samples=1).squeeze(-1)
                # Gathered before the <pad> overwrite below, whose logit is -inf: the mask
                # excludes those positions anyway, but -inf * False is nan, not 0, so a
                # single finished row would poison any sum a caller takes over the mask.
                logprobs[:, cur_length] = step_logprobs.gather(
                    1, pred_token_ids[:, None]
                ).squeeze(1)
            else:
                pred_token_ids = logits[:, -1, :].argmax(dim=-1)  # (batch,)
            pred_token_ids = torch.where(finished, tokenizer.pad_token_id, pred_token_ids)

            token_ids[:, cur_length] = pred_token_ids
            # finished still describes the state before this token, so ~finished marks the
            # rows that were genuinely choosing here — its own <eos> included, since
            # stopping is a decision — and masks off the filler after a row is done.
            completion_mask[:, cur_length] = ~finished
            num_cached = cur_length
            cur_length += 1

            finished |= pred_token_ids == tokenizer.eos_token_id
            if finished.all():
                break

    model.train(training)

    # Trimmed to what was actually written: the buffer is sized for the full budget, but
    # an all-finished batch stops early and the tail is pad the caller never asked for.
    if logprobs is not None:
        logprobs = logprobs[:, :cur_length]
    return (
        token_ids[:, :cur_length],
        completion_mask[:, :cur_length],
        logprobs,
        seq_starts,
        finished,
    )


def decode_rollouts(tokenizer, token_ids):
    """Decode rows of token ids to text, dropping only this module's own framing.

    <bos> is prepended by generate, <pad> fills the left padding and every position after a
    row finished, and <eos> is the stop signal, reported separately as `finished`. Anything
    else the model emitted survives — the reasoning markers included, which callers split on
    to find the answer and could not recover once stripped. That is what skip_special_tokens
    would take with it, which is why it stays False and the three are filtered by hand.

    Leaving them in is not harmless: a rollout that hit its token budget mid-answer has no
    closing marker to bound the answer, so the <eos> lands inside it and ast.literal_eval
    rejects a value that was otherwise correct — scoring a truncation as a wrong answer.
    """
    framing = {tokenizer.bos_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id}
    return [
        tokenizer.decode([i for i in row if i not in framing], skip_special_tokens=False)
        for row in token_ids.tolist()
    ]


def generate_texts(model, tokenizer, input_texts, device, max_new_tokens=None):
    """Decode generate's rollouts, returning (texts, finished).

    Each text is the prompt plus its continuation, with this function's own framing —
    <bos>, padding, <eos> — removed and everything else the model emitted left intact.
    `finished[i]` is True when row i stopped on <eos> rather than running out of budget,
    which is the difference between a wrong answer and a truncated one.
    """
    token_ids, _, _, _, finished = generate(
        model, tokenizer, input_texts, device, max_new_tokens
    )

    texts = decode_rollouts(tokenizer, token_ids)

    # Whether each row stopped on <eos> rather than exhausting its budget, returned rather
    # than left in the text. A truncated rollout is a different failure from a completed
    # one that answered wrongly, and the alternative — reading it back out of the string —
    # is what breaks scoring, since ast.literal_eval rejects an answer with <eos> stuck to
    # it. finished is already tracked to halt decoding, so carrying it out costs nothing.
    return texts, finished.tolist()
