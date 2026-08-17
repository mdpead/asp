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


def generate_texts(model, tokenizer, input_texts, device, max_new_tokens=None):

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

            pred_token_ids = logits[:, -1, :].argmax(dim=-1)  # (batch,)
            pred_token_ids = torch.where(finished, tokenizer.pad_token_id, pred_token_ids)

            token_ids[:, cur_length] = pred_token_ids
            num_cached = cur_length
            cur_length += 1

            finished |= pred_token_ids == tokenizer.eos_token_id
            if finished.all():
                break

    model.train(training)
    texts = tokenizer.batch_decode(token_ids[:, :cur_length], skip_special_tokens=True)
    return texts
