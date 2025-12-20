import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100

    prompt_input_ids = []
    output_input_ids = []

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_input_ids.append(torch.tensor(prompt_tokens))
        output_tokens = tokenizer.encode(output, add_special_tokens=False)
        output_input_ids.append(torch.tensor(output_tokens))

    seq_len = [len(p_ids) + len(o_ids) for p_ids, o_ids in zip(prompt_input_ids, output_input_ids)]
    max_length = max(seq_len)

    concatenated_input_ids = []
    concatenated_labels = []
    response_masks = []

    for (p_ids, o_ids) in zip(prompt_input_ids, output_input_ids):
        input_ids = torch.cat([p_ids, o_ids], dim=0)
        pad_length = max_length - input_ids.shape[0]
        padded_input_ids = F.pad(input_ids, (0, pad_length), value=pad_token_id)

        response_mask = torch.cat([torch.zeros_like(p_ids).bool(), torch.ones_like(o_ids).bool()], dim=0)
        padded_response_mask = F.pad(response_mask, (0, pad_length), value=False)

        concatenated_input_ids.append(padded_input_ids[:-1])
        concatenated_labels.append(padded_input_ids[1:])
        response_masks.append(padded_response_mask[1:])

    input_ids_tensor = torch.stack(concatenated_input_ids)
    label_tensor = torch.stack(concatenated_labels)
    response_mask_tensor = torch.stack(response_masks)

    return {"input_ids": input_ids_tensor, "labels": label_tensor, "response_mask": response_mask_tensor}


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -torch.sum(probs * log_probs, dim=-1)


def get_response_log_probs(model: PreTrainedModel, input_ids: torch.Tensor, labels: torch.Tensor,
                           return_token_entropy: bool = False) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    log_sum_exp = torch.logsumexp(logits, dim=-1)
    logit_y = logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    label_token_log_softmax = logit_y - log_sum_exp

    res = {"log_probs": label_token_log_softmax}
    if return_token_entropy:
        res['token_entropy'] = compute_entropy(logits)
    return res


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, normalize_constant: float,
                     dim: int | None = None) -> torch.Tensor:
    assert normalize_constant > 0, "Normalization constant must be > 0"

    masked_tensor = torch.where(mask, tensor, 0.0)
    total = torch.sum(masked_tensor, dim=dim)
    return total / normalize_constant
