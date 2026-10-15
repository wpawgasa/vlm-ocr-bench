# Stand-in for the vLLM base image: python + the entrypoint's deps + a `vllm` that prints its argv.
FROM python:3.11-slim
RUN pip install --no-cache-dir pydantic pyyaml
COPY huggingface_hub /usr/local/lib/python3.11/site-packages/huggingface_hub
COPY vllm /usr/local/bin/vllm
RUN chmod +x /usr/local/bin/vllm
