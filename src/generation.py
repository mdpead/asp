import warnings

import torch


def generate_texts(model, tokenizer, input_texts, device, max_new_tokens=None):
    # The prompt wins: keep as much of the caller's input as the context allows, then
    # generate into whatever room is left. max_new_tokens is an optional further cap on
    # the output, not a reservation that eats into the prompt.
    max_length = model.max_length

    prompt_budget = max_length - 1  # -1 leaves room for <bos>

    # Encode without special tokens so we control them: <bos> leads, no trailing <eos>
    prompt_ids = tokenizer(input_texts, add_special_tokens=False)["input_ids"]

    # A prompt that needs truncating would fill the context outright, leaving nothing to
    # generate, so trimming it only corrupts the input for no gain. Refuse instead.
    over = [len(ids) for ids in prompt_ids if len(ids) > prompt_budget]
    if over:
        raise ValueError(
            f"{len(over)} prompt(s) exceed the {prompt_budget}-token limit "
            f"(longest is {max(over)}, max_length is {max_length})"
        )
    prompt_ids = [[tokenizer.bos_token_id] + ids for ids in prompt_ids]
    encoded = tokenizer.pad({"input_ids": prompt_ids}, return_tensors="pt", padding_side="left")

    input_ids = encoded["input_ids"].to(device)
    prompt_lengths = encoded["attention_mask"].sum(dim=-1).to(device)  # (batch,)

    batch_size, input_length = input_ids.shape

    # Whatever the prompt leaves is the generation budget, optionally capped further
    room = max_length - input_length
    new_tokens = room if max_new_tokens is None else min(max_new_tokens, room)
    if new_tokens < 1:
        # The prompt fills the context, so there is genuinely nothing to generate. Hand
        # back the prompts rather than stealing a token from them to manufacture output.
        warnings.warn(
            f"prompt fills {input_length} of {max_length} tokens, generating nothing"
        )
    if max_new_tokens is not None and new_tokens < max_new_tokens:
        warnings.warn(
            f"max_new_tokens reduced from {max_new_tokens} to {new_tokens}; "
            f"the prompt uses {input_length} of {max_length} tokens"
        )

    # Left padding puts every row's real tokens flush against the same right-hand edge, so
    # the batch shares one frontier and each row's pads form a leading block
    seq_starts = (input_length - prompt_lengths).to(torch.int32)  # (batch,) first real column
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Preallocate the whole buffer, then fill forward from the prompt
    token_ids = torch.full(
        (batch_size, input_length + new_tokens),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    token_ids[:, :input_length] = input_ids
    cur_length = input_length

    training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(new_tokens):
            logits, _ = model(token_ids[:, 0:cur_length], seq_starts)  # (batch, cur_length, vocab)

            pred_token_ids = logits[:, -1, :].argmax(dim=-1)  # (batch,)
            pred_token_ids = torch.where(finished, tokenizer.pad_token_id, pred_token_ids)

            token_ids[:, cur_length] = pred_token_ids
            cur_length += 1

            finished |= pred_token_ids == tokenizer.eos_token_id
            if finished.all():
                break

    model.train(training)
    texts = tokenizer.batch_decode(token_ids[:, :cur_length], skip_special_tokens=True)
    return texts
