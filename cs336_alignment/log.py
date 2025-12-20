from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class GenerationLogSummary:
    avg_len: float
    avg_len_correct: float
    avg_len_incorrect: float
    num_correct: int
    num_total: int


def _token_entropy_from_logits(step_logits: torch.Tensor) -> torch.Tensor:
    """
    step_logits: [V] or [B, V]
    returns entropy per item: [ ] or [B]
    """
    logp = F.log_softmax(step_logits, dim=-1)
    p = logp.exp()
    ent = -(p * logp).sum(dim=-1)
    return ent


@torch.no_grad()
def log_generations(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    true_answers: list[str],
    reward_fn: Callable[[str, str], dict],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 1.0,
    num_examples: int = 8,
    device: Optional[torch.device] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> GenerationLogSummary:
    """
    Logs generations "in the loop" and returns length summary stats.

    Logs for each example:
      1) prompt
      2) model response
      3) ground truth answer
      4) reward dict (format / answer / total, etc.)
      5) avg token entropy over response tokens
      6) response length stats overall / correct / incorrect (returned as summary)
    """
    if logger is None:
        logger = print

    assert len(prompts) == len(true_answers), "prompts and true_answers must have same length"
    n = min(num_examples, len(prompts))
    prompts = prompts[:n]
    true_answers = true_answers[:n]

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Tokenize prompts
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Generate with scores so we can compute token entropy
    gen_out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=(temperature is not None and temperature > 0),
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,  # IMPORTANT: gives per-step logits
    )

    # sequences: [B, prompt_len + new_len] (padded prompt_len can differ; sequences are uniform for batch)
    sequences = gen_out.sequences  # token ids
    scores = gen_out.scores        # list length = #generated steps; each is [B, V]

    # Figure out response token ids per example.
    # In batched generation with padding, prompt lengths differ, so we compute each prompt length from attention_mask.
    if attention_mask is None:
        # Fallback: assume all prompts same length as input_ids.shape[1]
        prompt_lens = torch.full((n,), input_ids.shape[1], device=device, dtype=torch.long)
    else:
        prompt_lens = attention_mask.sum(dim=1)  # [B]

    # Decode full text and response-only text
    # response token span is [prompt_len : prompt_len + new_tokens] per example.
    responses: list[str] = []
    response_lengths: list[int] = []
    avg_entropies: list[float] = []
    rewards: list[dict] = []
    is_correct: list[bool] = []

    # scores[t] corresponds to the logits used to sample token at generated step t (0-based).
    # So token entropies apply exactly to generated tokens; we’ll average over generated steps that belong to the response.
    # In generate(), all examples generate the same number of steps unless they hit EOS early; EOS handling varies by model.
    # We'll compute entropy over the actually generated response tokens for each example by stopping at first EOS if present.
    eos_id = tokenizer.eos_token_id

    for i in range(n):
        pl = int(prompt_lens[i].item())
        full_ids = sequences[i]

        # Extract generated token ids (candidate response tokens)
        gen_ids = full_ids[pl:]

        # If EOS exists, cut at EOS (inclusive or exclusive depending on your preference)
        if eos_id is not None:
            eos_pos = (gen_ids == eos_id).nonzero(as_tuple=False)
            if eos_pos.numel() > 0:
                cut = int(eos_pos[0].item())  # first eos
                gen_ids = gen_ids[:cut]  # exclude eos token from "response"
        resp_len = int(gen_ids.numel())
        response_lengths.append(resp_len)

        # Decode response text
        resp_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        responses.append(resp_text)

        # Compute average token entropy over response tokens.
        # Entropy at step t uses scores[t][i] and corresponds to generated token t.
        # Need to align response tokens to steps: response step 0 corresponds to scores[0].
        # But if model stopped early, resp_len may be < len(scores). We'll clamp.
        T = min(resp_len, len(scores))
        if T == 0:
            avg_ent = float("nan")
        else:
            ents = []
            for t in range(T):
                ents.append(_token_entropy_from_logits(scores[t][i]).detach().float().cpu())
            avg_ent = float(torch.stack(ents).mean().item())
        avg_entropies.append(avg_ent)

        # Reward info
        r = reward_fn(resp_text, true_answers[i])
        rewards.append(r)

        # Define correctness: prefer reward dict if it has something explicit; otherwise compare extracted answers upstream.
        # Common: r["answer_reward"] is 1.0 for correct, 0.0 for incorrect.
        if "answer_reward" in r:
            correct = bool(float(r["answer_reward"]) > 0.0)
        elif "reward" in r:
            # fallback heuristic: treat positive total reward as correct-ish
            correct = bool(float(r["reward"]) > 0.0)
        else:
            correct = False
        is_correct.append(correct)

    # Per-example logging
    for i in range(n):
        logger("=" * 80)
        logger(f"[Example {i}]")
        logger("1) Prompt:")
        logger(prompts[i])
        logger("\n2) Model response:")
        logger(responses[i])
        logger("\n3) Ground-truth answer:")
        logger(true_answers[i])
        logger("\n4) Reward info:")
        logger(str(rewards[i]))
        logger("\n5) Avg token entropy (response):")
        logger(f"{avg_entropies[i]:.4f}")
        logger("\n6) Response length (tokens):")
        logger(str(response_lengths[i]))
        logger(f"Correct? {is_correct[i]}")

    # Summary stats requested in (6)
    lengths = torch.tensor(response_lengths, dtype=torch.float32)
    avg_len = float(lengths.mean().item()) if len(lengths) else float("nan")

    correct_mask = torch.tensor(is_correct, dtype=torch.bool)
    if correct_mask.any():
        avg_len_correct = float(lengths[correct_mask].mean().item())
    else:
        avg_len_correct = float("nan")

    if (~correct_mask).any():
        avg_len_incorrect = float(lengths[~correct_mask].mean().item())
    else:
        avg_len_incorrect = float("nan")

    num_correct = int(correct_mask.sum().item())
    num_total = n

    logger("=" * 80)
    logger("SUMMARY")
    logger(f"Avg response length: {avg_len:.2f} tokens")
    logger(f"Avg length (correct): {avg_len_correct:.2f} tokens")
    logger(f"Avg length (incorrect): {avg_len_incorrect:.2f} tokens")
    logger(f"Correct: {num_correct}/{num_total}")

    return GenerationLogSummary(
        avg_len=avg_len,
        avg_len_correct=avg_len_correct,
        avg_len_incorrect=avg_len_incorrect,
        num_correct=num_correct,
        num_total=num_total,
    )
