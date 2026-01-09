from importlib.metadata import metadata
from typing import Callable, Literal

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
        advantages: torch.Tensor, policy_log_probs: torch.Tensor, old_log_probs: torch.Tensor, cliprange: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pi_ratio = torch.exp(policy_log_probs - old_log_probs)
    _, seq_len = policy_log_probs.shape
    v = advantages * pi_ratio
    v_clip = torch.clip(pi_ratio, min=1 - cliprange, max=1 + cliprange) * advantages

    metadata = {}
    return -torch.min(v, v_clip), metadata


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
