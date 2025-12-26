from unittest.mock import patch

import torch
from vllm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from cs336_alignment.utils import print_color


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.2):
    vllm_set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.v1.worker.gpu_worker.Worker.determine_available_memory", return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            # device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def init_vllm_from_path(model_path: str, seed: int, gpu_memory_utilization: float = 0.2) -> LLM:
    # vLLM 0.11.0: do NOT pass device= ; use CUDA_VISIBLE_DEVICES outside if needed.
    vllm_set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.v1.worker.gpu_worker.Worker.determine_available_memory",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_path,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_model_into_vllm_instance(model: torch.nn.Module, llm: LLM):
    # snapshot to CPU -> then load into vLLM
    model.eval()
    model.tie_weights()
    cpu_sd = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(cpu_sd.items())
    model.train()
    torch.cuda.synchronize(torch.device("cuda:1"))
    print_color("Model weights loaded into VLLM instance.")
