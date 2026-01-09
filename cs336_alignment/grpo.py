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
