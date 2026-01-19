from importlib.metadata import metadata
from typing import Callable, Literal
from urllib import response

import einops
import torch


def compute_group_normalized_rewards(
        reward_fn: Callable[[str, str], dict[str, float]],
        rollout_responses: list[str],
        repeated_ground_truths: list[str],
        group_size: int,
        advantage_eps: float,
        normalized_by_std: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    raw_rewards = []
    for rollout_response, gt_response in zip(rollout_responses, repeated_ground_truths):
        reward = reward_fn(rollout_response, gt_response)["reward"]
        raw_rewards.append(reward)

    raw_rewards = torch.tensor(raw_rewards)
    rewards_per_group = raw_rewards.reshape((-1, group_size))
    mean_rewards_per_group = torch.mean(rewards_per_group, dim=-1, keepdim=True)
    advantage = rewards_per_group - mean_rewards_per_group

    if normalized_by_std:
        std_reward_per_group = torch.std(rewards_per_group, dim=-1, keepdim=True)
        advantage /= std_reward_per_group + advantage_eps

    advantage = advantage.flatten()

    metadata = {
        "mean": torch.mean(raw_rewards).item(),
        "std": torch.std(raw_rewards).item(),
        "max": torch.max(raw_rewards).item(),
        "min": torch.min(raw_rewards).item(),
    }

    return advantage, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    _, seq_len = policy_log_probs.shape
    loss = - raw_rewards_or_advantages * policy_log_probs
    return loss


def compute_grpo_clip_loss(
        advantages: torch.Tensor,  # [B, 1]
        policy_log_probs: torch.Tensor,  # [B, T]
        old_log_probs: torch.Tensor,  # [B, T]
        cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    advantages = advantages.detach()

    pi_ratio = torch.exp(policy_log_probs - old_log_probs)
    pi_ratio_clip = pi_ratio.clamp(1 - cliprange, 1 + cliprange)
    v = advantages * pi_ratio
    v_clip = advantages * pi_ratio_clip
    loss = - torch.min(v, v_clip)

    with torch.no_grad():
        clip_frac = (pi_ratio != pi_ratio_clip).float().mean()
        approx_kl = (old_log_probs - policy_log_probs).mean()
        metadata = {
            "ratio_mean": pi_ratio.mean(),
            "ratio_min": pi_ratio.min().detach(),
            "ratio_max": pi_ratio.max().detach(),
            "clip_frac": clip_frac,
            # "approx_kl": approx_kl,  # proxy
        }

    return loss, metadata


def compute_policy_gradient_loss(
        policy_log_probs: torch.Tensor,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    assert loss_type in ["no_baseline", "reinforce_with_baseline", "grpo_clip"], f"Unknown loss type {loss_type}"
    B, T = policy_log_probs.shape

    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards must not be None for no_baseline"
        assert raw_rewards.shape == (B, 1), f"Invalid shape for raw_rewards {raw_rewards.shape}"
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata: dict[str, torch.Tensor] = {"mean_raw_reward": torch.mean(raw_rewards).item()}
        return loss, metadata
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages must not be None for reinforce_with_baseline"
        assert advantages.shape == (B, 1), f"Invalid shape for advantages {advantages.shape}"
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata: dict[str, torch.Tensor] = {"mean_advantages": torch.mean(advantages).item()}
        return loss, metadata

    # loss type == grpo_clip
    assert advantages is not None, "advantages must not be None for grpo_clip"
    assert old_log_probs is not None, "old_log_probs must not be None for grpo_clip"
    assert cliprange is not None, "cliprange must not be None for grpo_clip"
    assert advantages.shape == (B, 1), f"Invalid shape for advantages {advantages.shape}"
    assert old_log_probs.shape == (B, T), f"Invalid shape for old_log_probs {old_log_probs.shape}"
    assert cliprange > 0, "cliprange must be > 0"

    loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)

    return loss, metadata


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    masked_tensor = tensor * mask
    return torch.sum(masked_tensor, dim=dim) / torch.sum(mask, dim=dim)


def grpo_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    with torch.no_grad():
        denom = response_mask.sum().clamp_min(1)
        metadata['kl_denom'] = denom.item()
        approx_kl = ((old_log_probs - policy_log_probs) * response_mask).sum() / denom
        metadata['approx_kl'] = approx_kl.detach()
        # PPO-style approximate KL
        log_ratio = policy_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        approx_kl_ppo = ((ratio - 1 - log_ratio) * response_mask).sum() / denom
        metadata['approx_kl_ppo'] = approx_kl_ppo.detach()

    loss = masked_mean(loss, response_mask)
    loss = loss / gradient_accumulation_steps
    loss.backward()

    return loss, metadata
