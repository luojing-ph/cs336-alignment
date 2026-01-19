"""
Single-GPU GRPO trainer.

Key idea (same as your EI single-GPU script):
- vLLM and HF cannot coexist on the same GPU reliably.
- So each GRPO step:
    A) Save HF checkpoint (pi_old), move HF+optimizer -> CPU
    B) Init vLLM from checkpoint on cuda:0 and sample rollouts
    C) Delete vLLM, clear GPU
    D) Move HF+optimizer -> cuda:0
    E) Build rollout dataset + compute old_log_probs with HF (pi_old)
    F) Run a few PPO/GRPO clipped updates on that rollout dataset
    G) Periodic eval uses the same swap pattern.

This is a "drop-in" adaptation of the reference GRPO script.
"""

import gc
import logging
import math
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field

import dotenv
import fire
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from cs336_alignment.data_utils import load_and_format_prompts
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.grpo import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
)
from cs336_alignment.sft_utils import get_response_log_probs, tokenize_prompt_and_output
from cs336_alignment.utils import (
    get_run_name,
    print_color,
    print_rich_dict,
    save_model_and_tokenizer,
)
from cs336_alignment.vllm_utils import init_vllm_from_path
from train_sft import evaluate_vllm

logging.getLogger("vllm").setLevel(logging.WARNING)


# ----------------------------
# Memory helpers (single GPU)
# ----------------------------
def clear():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def cycle_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def move_model_to_cpu(model: torch.nn.Module):
    model.to("cpu")
    clear()


def move_model_to_gpu(model: torch.nn.Module, device: str):
    model.train()
    model.to(device)
    clear()


def move_optimizer_to_cpu(optimizer: torch.optim.Optimizer):
    # Helps avoid keeping Adam moments on GPU when vLLM needs memory.
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cpu()


def move_optimizer_to_gpu(optimizer: torch.optim.Optimizer, device: str):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


# ----------------------------
# Configs
# ----------------------------
@dataclass
class TrainConfig:
    # Basic
    experiment_name_base: str = "experiments"
    experiment_name: str = "grpo-qwen2.5"
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    data_path: str = "./data/gsm8k/train.jsonl"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"
    num_example: int = 128

    # Microbatch / grad accumulation
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 32

    # GRPO
    n_grpo_steps: int = 20
    question_per_grpo_step: int = 64
    group_size: int = 8

    # How much to train on each rollout batch
    n_train_epochs_per_rollout_batch: int = 1
    n_train_steps_per_rollout_batch: int = 1  # will be overwritten in __post_init__

    advantage_eps: float = 1e-6
    use_std_normalization: bool = True

    mixed_precision_training: bool = True
    learning_rate: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.95)
    cliprange: float = 0.2
    max_grad_norm: float = 1.0

    # Single GPU
    device: str = "cuda:0"

    # vLLM sampling
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 1024
    stop_tokens: list[str] = field(default_factory=lambda: ["</answer>"])
    include_stop_str_in_output: bool = True
    min_tokens: int = 4
    vllm_seed: int = 42
    vllm_gpu_memory_utilization: float = 0.30  # lower if OOM

    # Eval
    eval_steps: int = 5

    def __post_init__(self):
        total_data_points = (
            self.n_train_epochs_per_rollout_batch
            * self.question_per_grpo_step
            * self.group_size
        )
        effective_batch = self.micro_batch_size * self.gradient_accumulation_steps
        # number of optimizer steps to cover the rollout dataset once per epoch
        self.n_train_steps_per_rollout_batch = max(
            1, total_data_points // effective_batch
        )


@dataclass
class EvaluateConfig:
    data_path: str = "./data/gsm8k/test.jsonl"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"
    temperature: float = 0.1
    top_p: float = 0.9
    stop_tokens: list[str] = field(default_factory=lambda: ["</answer>"])
    max_tokens: int = 1024
    include_stop_str_in_output: bool = True


# ----------------------------
# Base dataset for sampling questions
# ----------------------------
class GRPODataset(Dataset):
    def __init__(self, train_prompts, train_cot, train_answers):
        self.train_prompts = train_prompts
        self.train_cot = train_cot
        self.train_answers = train_answers

    def __len__(self):
        return len(self.train_prompts)

    def __getitem__(self, idx: int):
        return (
            self.train_prompts[idx],
            self.train_cot[idx],
            self.train_answers[idx].strip(),
        )


# ----------------------------
# Old log-probs (pi_old) computation using HF
# ----------------------------
@torch.no_grad()
def get_old_log_probs(
    model,
    input_ids: torch.Tensor,  # shape: [Q * G, T]
    labels: torch.Tensor,  # shape: [Q * G, T]
    train_config: TrainConfig,
) -> tuple[list[list[float]], list[list[float]]]:
    """
    Computes token-level log-probabilities under the old policy (pi_old).

    Returns:
        log_probs:      list length (Q * G), each element is List[float] of length T
        token_entropy: list length (Q * G), each element is List[float] of length T
    """

    # Save and force eval mode (important for PPO stability)
    was_training: bool = model.training
    model.eval()

    log_probs: list[list[float]] = []
    token_entropy: list[list[float]] = []

    # Move rollout tensors to device
    input_ids = input_ids.to(train_config.device)  # [Q*G, T]
    labels = labels.to(train_config.device)  # [Q*G, T]

    # Number of prompts and group size
    Q = train_config.question_per_grpo_step
    G = train_config.group_size

    # Loop over questions (not responses!)
    for q_i in trange(Q, desc="old_log_probs"):
        start = q_i * G
        end = start + G

        # Slice one question's group
        input_part = input_ids[start:end]  # [G, T]
        labels_part = labels[start:end]  # [G, T]

        out = get_response_log_probs(
            model=model,
            input_ids=input_part,  # [G, T]
            labels=labels_part,  # [G, T]
            return_token_entropy=True,
        )
        # out["log_probs"]      : Tensor [G, T]
        # out["token_entropy"]  : Tensor [G, T]

        log_probs.extend(out["log_probs"].tolist())  # adds G lists of length T
        token_entropy.extend(out["token_entropy"].tolist())  # adds G lists of length T

        del out, input_part, labels_part
        clear()

    # Restore original training state
    model.train(was_training)

    # Final shapes:
    #   log_probs      : List length (Q*G), each element List[T]
    #   token_entropy  : List length (Q*G), each element List[T]
    return log_probs, token_entropy


# ----------------------------
# Rollout dataset (prompt, response, reward, advantage, old_log_probs)
# ----------------------------
class GRPORolloutDataset(Dataset):
    def __init__(
        self,
        model,
        prompts,
        responses,
        raw_rewards: torch.Tensor,
        advantages: torch.Tensor,
        train_config: TrainConfig,
        tokenizer,
    ):
        # Store rewards/advantages on CPU
        self.raw_rewards = raw_rewards.cpu()
        self.advantages = advantages.cpu()

        encoded = tokenize_prompt_and_output(prompts, responses, tokenizer)
        self.input_ids = encoded["input_ids"]
        self.labels = encoded["labels"]
        self.response_mask = encoded["response_mask"]

        # Compute old log probs with current HF model (pi_old for this rollout batch)
        self.old_log_probs, self.token_entropy = get_old_log_probs(
            model, self.input_ids, self.labels, train_config
        )
        self.old_log_probs = torch.tensor(
            self.old_log_probs, dtype=torch.float32
        )  # [N, T]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        input_ids = self.input_ids[idx]
        labels = self.labels[idx]
        response_mask = self.response_mask[idx]
        raw_reward = self.raw_rewards[idx]
        advantage = self.advantages[idx].unsqueeze(
            -1
        )  # shape [1] -> [1,1] style broadcasting
        old_log_probs = self.old_log_probs[idx]
        return input_ids, labels, response_mask, raw_reward, advantage, old_log_probs


# ----------------------------
# Policy update (GRPO clip) on one rollout batch
# ----------------------------
def update_policy_on_rollouts(
    model,
    optimizer,
    tokenizer,
    train_config: TrainConfig,
    prompts,
    responses,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    global_step: int,
):
    # Build dataset; no need for grad in dataset creation
    with torch.no_grad():
        dataset = GRPORolloutDataset(
            model=model,
            prompts=prompts,
            responses=responses,
            raw_rewards=raw_rewards,
            advantages=advantages,
            train_config=train_config,
            tokenizer=tokenizer,
        )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=train_config.micro_batch_size,
        shuffle=True,
        num_workers=0,  # single-GPU stability
        pin_memory=False,
    )
    cycled = cycle_dataloader(dataloader)

    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if train_config.mixed_precision_training
        else nullcontext()
    )

    optimizer.zero_grad(set_to_none=True)
    model.train()

    micro = 0
    opt_steps = 0
    batch_loss_accum = 0.0
    approx_kl_acc = 0.0
    approx_kl_ppo_acc = 0.0
    kl_denom_acc = 0
    global_step_ = global_step

    while opt_steps < train_config.n_train_steps_per_rollout_batch:
        batch = next(cycled)
        input_ids, labels, response_mask, raw_rewards_b, advantages_b, old_log_probs = (
            batch
        )

        if advantages_b.ndim == 1:
            advantages_b = advantages_b[:, None]
        input_ids = input_ids.to(train_config.device)
        labels = labels.to(train_config.device)
        response_mask = response_mask.to(train_config.device)
        raw_rewards_b = raw_rewards_b.to(train_config.device)
        advantages_b = advantages_b.to(train_config.device)
        old_log_probs = old_log_probs.to(train_config.device)

        with ctx:
            out = get_response_log_probs(
                model=model, input_ids=input_ids, labels=labels
            )
            policy_log_probs = out["log_probs"]

            loss, metadata = grpo_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=train_config.gradient_accumulation_steps,
                loss_type="grpo_clip",
                raw_rewards=raw_rewards_b,
                advantages=advantages_b,
                old_log_probs=old_log_probs,
                cliprange=train_config.cliprange,
            )

        batch_loss_accum += float(loss.detach().cpu())
        approx_kl_acc += float(metadata["approx_kl"].cpu())
        approx_kl_ppo_acc += float(metadata["approx_kl_ppo"].cpu())
        kl_denom_acc += float(metadata["kl_denom"])
        micro += 1

        del (
            input_ids,
            labels,
            response_mask,
            raw_rewards_b,
            advantages_b,
            old_log_probs,
            out,
            policy_log_probs,
        )
        clear()

        if micro % train_config.gradient_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            opt_steps += 1
            global_step_ += 1

            approx_kl_step = approx_kl_acc / max(1.0, kl_denom_acc)
            approx_kl_ppo_step = approx_kl_ppo_acc / max(1.0, kl_denom_acc)
            step_loss = batch_loss_accum  # already scaled inside microbatch step

            print(
                f"[update] opt_step={opt_steps}/{train_config.n_train_steps_per_rollout_batch} "
                f"global_step={global_step_} loss={step_loss:.6f}"
                f" approx_kl={approx_kl_step:.8f} approx_kl_ppo={approx_kl_ppo_step:.8f} "
            )

            wandb.log(
                {
                    "train/loss": step_loss,
                    "train/approx_kl": approx_kl_step,
                    "train/approx_kl_ppo": approx_kl_ppo_step,
                    "train_step": global_step_,
                }
            )

            batch_loss_accum = 0.0
            approx_kl_acc = 0.0
            approx_kl_ppo_acc = 0.0
            kl_denom_acc = 0.0

    return global_step_


# ----------------------------
# Main GRPO training loop (single GPU)
# ----------------------------
def train_grpo_single_gpu(
    train_config: TrainConfig,
    eval_config: EvaluateConfig,
    train_prompts,
    train_cot,
    train_answers,
    seed: int,
):
    wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project="cs336-alignment-grpo",
        config={"train": asdict(train_config), "eval": asdict(eval_config)},
        name=get_run_name("grpo", train_config),
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # HF model + optimizer
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=train_config.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)

    move_model_to_gpu(model, train_config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate, betas=train_config.betas
    )

    # Base question sampling loader
    base_ds = GRPODataset(train_prompts, train_cot, train_answers)
    base_dl = DataLoader(
        dataset=base_ds,
        batch_size=train_config.question_per_grpo_step,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    cycled_questions = cycle_dataloader(base_dl)

    # vLLM sampling params (group_size outputs per prompt)
    grpo_sp = SamplingParams(
        temperature=train_config.temperature,
        top_p=train_config.top_p,
        max_tokens=train_config.max_tokens,
        min_tokens=train_config.min_tokens,
        stop=train_config.stop_tokens,
        include_stop_str_in_output=train_config.include_stop_str_in_output,
        n=train_config.group_size,
        seed=train_config.vllm_seed,
    )
    eval_sp = SamplingParams(
        temperature=eval_config.temperature,
        top_p=eval_config.top_p,
        max_tokens=eval_config.max_tokens,
        stop=eval_config.stop_tokens,
        include_stop_str_in_output=eval_config.include_stop_str_in_output,
    )

    global_step = 0

    for grpo_step in range(train_config.n_grpo_steps):
        # (1) sample questions
        sample_prompts, _, sample_answers = next(cycled_questions)
        sample_prompts = list(sample_prompts)
        sample_answers = list(sample_answers)

        # ---------------------------
        # SINGLE GPU SWAP: HF -> CPU, vLLM -> GPU for rollouts
        # ---------------------------
        model.eval()
        ckpt_dir = save_model_and_tokenizer(model, tokenizer, train_config)

        move_optimizer_to_cpu(optimizer)
        move_model_to_cpu(model)

        vllm = init_vllm_from_path(
            model_path=str(ckpt_dir),
            seed=seed,
            gpu_memory_utilization=train_config.vllm_gpu_memory_utilization,
        )

        # sample group_size responses per prompt
        print(
            f"[rollout] step={grpo_step} sampling {train_config.group_size} per prompt..."
        )
        all_gens = vllm.generate(sample_prompts, grpo_sp)

        all_prompts = []
        all_responses = []
        all_answers = []
        for q, a, gens in zip(sample_prompts, sample_answers, all_gens):
            for o in gens.outputs:
                all_prompts.append(q)
                all_responses.append(o.text)
                all_answers.append(a)

        del vllm
        clear()

        # ---------------------------
        # Swap back: HF -> GPU for advantages + old_log_probs + update
        # ---------------------------
        move_model_to_gpu(model, train_config.device)
        move_optimizer_to_gpu(optimizer, train_config.device)

        # (2) compute rewards + group-normalized advantages
        advantages, raw_rewards, metadata = compute_group_normalized_rewards(
            r1_zero_reward_fn,
            rollout_responses=all_responses,
            repeated_ground_truths=all_answers,
            group_size=train_config.group_size,
            advantage_eps=train_config.advantage_eps,
            normalized_by_std=train_config.use_std_normalization,
        )

        # (3) update policy on rollout batch (pi_old fixed via old_log_probs)
        global_step = update_policy_on_rollouts(
            model=model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            train_config=train_config,
            prompts=all_prompts,
            responses=all_responses,
            raw_rewards=raw_rewards,
            advantages=advantages,
            global_step=global_step,
        )

        # Periodic eval (same swap pattern)
        if (grpo_step + 1) % train_config.eval_steps == 0:
            model.eval()
            ckpt_dir = save_model_and_tokenizer(model, tokenizer, train_config)

            move_optimizer_to_cpu(optimizer)
            move_model_to_cpu(model)

            vllm = init_vllm_from_path(
                model_path=str(ckpt_dir),
                seed=seed,
                gpu_memory_utilization=train_config.vllm_gpu_memory_utilization,
            )

            eval_prompts, _, eval_answers = load_and_format_prompts(
                eval_config.data_path, eval_config.prompt_path
            )
            results = evaluate_vllm(
                vllm_model=vllm,
                reward_fn=r1_zero_reward_fn,
                prompts=eval_prompts,
                answers=eval_answers,
                eval_sampling_params=eval_sp,
            )

            wandb.log(
                {
                    "eval/correct": results["correct"],
                    "eval/answer_wrong": results["answer_wrong"],
                    "eval/format_wrong": results["format_wrong"],
                    "eval_step": grpo_step,
                }
            )
            print(results)

            del vllm
            clear()

            # back to training
            move_model_to_gpu(model, train_config.device)
            move_optimizer_to_gpu(optimizer, train_config.device)

        # Save each step (optional; comment out if slow)
        save_model_and_tokenizer(model, tokenizer, train_config)

    save_model_and_tokenizer(model, tokenizer, train_config)
    wandb.finish()
    print("Training finished.")


def main(
    *,
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    data_path: str = "./data/gsm8k/train.jsonl",
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt",
    seed: int = 123,
):
    dotenv.load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/workspace/hf")
    os.environ["HF_HUB_CACHE"] = os.environ.get("HF_HUB_CACHE", "/workspace/hf/hub")

    # Often helps vLLM cleanup behavior on some setups
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)

    train_config = TrainConfig()
    eval_config = EvaluateConfig()

    train_config.model_name = model_name
    train_config.data_path = data_path
    train_config.prompt_path = prompt_path
    eval_config.prompt_path = prompt_path

    prompts, cot, answers = load_and_format_prompts(
        train_config.data_path, train_config.prompt_path
    )

    train_config.num_example = len(prompts)
    train_config.experiment_name = f"grpo_single_gpu_{train_config.num_example}"

    print_rich_dict(
        {"train config": asdict(train_config), "eval config": asdict(eval_config)}
    )

    train_grpo_single_gpu(
        train_config=train_config,
        eval_config=eval_config,
        train_prompts=prompts,
        train_cot=cot,
        train_answers=answers,
        seed=seed,
    )


if __name__ == "__main__":
    fire.Fire(main)
