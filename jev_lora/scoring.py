"""PorTAL continuation scoring: sum log p(answer tokens | prompt) / answer characters.

Matches portallib.evaluation's separate prompt/continuation tokenization, boundary
whitespace, left prompt truncation, and empty-prompt BOS/EOS fallback. No chat
wrapper, answer letters, reasoning generation, or gold label is used here.
"""
import math


def encode_choices(tokenizer, prompt, choices, max_prompt=768):
    if max_prompt < 1 or len(choices) < 2:
        raise ValueError("Positive prompt budget and at least two choices required")
    normalized = prompt.rstrip()
    boundary = prompt[len(normalized):]
    original = list(tokenizer(normalized, add_special_tokens=True).input_ids)
    prefix = original[-max_prompt:]
    if not prefix:
        start = getattr(tokenizer, "bos_token_id", None)
        if start is None:
            start = getattr(tokenizer, "eos_token_id", None)
        if start is None:
            raise ValueError("Empty prompt requires BOS or EOS")
        prefix = [int(start)]
    encoded = []
    for choice in choices:
        continuation = boundary + choice
        answer = list(tokenizer(continuation, add_special_tokens=False).input_ids)
        if not answer:
            raise ValueError("Empty tokenized continuation")
        encoded.append({"ids": prefix + answer, "prompt_length": len(prefix),
                        "answer_tokens": len(answer), "characters": max(len(continuation), 1)})
    return encoded, {"original_prompt_tokens": len(original), "prompt_tokens": len(prefix),
                     "truncated": len(original) > max_prompt}


def normalized_choice_scores(logprobs, characters):
    if len(logprobs) < 2 or len(logprobs) != len(characters):
        raise ValueError("Missing choice scores")
    if any(not math.isfinite(v) for v in logprobs) or any(c <= 0 for c in characters):
        raise ValueError("Non-finite scores or invalid character counts")
    scores = [p / c for p, c in zip(logprobs, characters)]
    return scores, max(range(len(scores)), key=lambda i: scores[i])


def score_choices(model, tokenizer, prompt, choices, max_prompt=768, batch_size=1):
    import torch
    import torch.nn.functional as F
    if batch_size < 1:
        raise ValueError("choice-batch-size must be positive")
    encoded, info = encode_choices(tokenizer, prompt, choices, max_prompt)
    max_context = getattr(model.config, "max_position_embeddings", None)
    if max_context and any(len(r["ids"]) > max_context for r in encoded):
        raise ValueError("Prompt plus continuation exceeds context; answers are never silently truncated")
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    if pad is None:
        raise ValueError("Tokenizer has neither pad nor EOS token")
    device = next(model.parameters()).device
    logprobs = []
    with torch.inference_mode():
        for offset in range(0, len(encoded), batch_size):
            chunk = encoded[offset:offset + batch_size]
            width = max(len(r["ids"]) for r in chunk)
            ids = torch.full((len(chunk), width), pad, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            for i, row in enumerate(chunk):
                ids[i, :len(row["ids"])] = torch.tensor(row["ids"], device=device)
                mask[i, :len(row["ids"])] = 1
            # use_cache=False avoids allocating KV caches for teacher-forced scoring.
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            for i, row in enumerate(chunk):
                start, end = row["prompt_length"], len(row["ids"])
                answer_logits = logits[i, start - 1:end - 1].float()
                logprobs.append(float(-F.cross_entropy(answer_logits, ids[i, start:end], reduction="sum")))
            del logits
    chars = [r["characters"] for r in encoded]
    scores, predicted = normalized_choice_scores(logprobs, chars)
    return dict(info, predicted_idx=predicted, choice_scores=scores, choice_logprobs=logprobs,
                choice_characters=chars, choice_token_counts=[r["answer_tokens"] for r in encoded])
