# Dockerfile.verl.dev
# FROM nvcr.io/nvidia/pytorch:25.06-py3
FROM hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0

# Work in /workspace
WORKDIR /app

COPY . /app

RUN pip install fire latex2sympy2_extended math_verify

WORKDIR /app