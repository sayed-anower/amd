# AMD AI League - Match 4: Intelligent Web Information Retrieval
#
# ROCm PyTorch is the pinned base. We never pip-install torch, so the ROCm
# build that ships in this image is preserved. llama.cpp is installed from a
# prebuilt release archive (no compilation, no llama-cpp-python).
FROM rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0

ARG LLAMA_BUILD=b11849

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LLM_BASE_URL=http://127.0.0.1:8001/v1 \
    LLM_MODEL=qwen2.5-0.5b-instruct \
    LLM_TIMEOUT=120 \
    MODEL_PATH=/app/model/qwen-2.5-0.5b.gguf \
    SEARCH_PROVIDER=tavily \
    PROVIDER_FALLBACK=local \
    PLANNER_MODE=auto \
    MAX_ROUNDS=2 \
    MAX_SNIPPETS=3 \
    EVIDENCE_CHAR_BUDGET=1000 \
    ANSWER_MAX_TOKENS=96 \
    MAX_RESPONSE_WORDS=45

WORKDIR /app

# Runtime tools for the release archive and the health check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Hard guard: the base image's ROCm PyTorch must still be the active torch.
RUN python -c "import torch; v=torch.__version__; assert 'rocm' in v.lower(), 'ROCm torch was overwritten: '+v; assert getattr(torch.version, 'hip', None), 'torch.version.hip missing'; print('[build] ROCm torch OK:', v)"

# Prebuilt llama.cpp binaries only. No compiler, no CMake, no llama-cpp-python.
RUN mkdir -p /opt/llama \
    && curl -fL --retry 3 --connect-timeout 20 \
        "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_BUILD}/llama-${LLAMA_BUILD}-bin-ubuntu-x64.tar.gz" -o /tmp/llama.tar.gz \
    && tar -xzf /tmp/llama.tar.gz --strip-components=1 -C /opt/llama \
    && rm -f /tmp/llama.tar.gz \
    && test -x /opt/llama/llama-server \
    && test -x /opt/llama/llama-cli \
    && ln -sf /opt/llama/llama-server /usr/bin/llama-server \
    && ln -sf /opt/llama/llama-cli /usr/bin/llama-cli \
    && ln -sf /opt/llama/llama-completion /usr/bin/llama-completion \
    && /usr/bin/llama-cli --version

# Application code and the local GGUF model.
COPY requirements.txt /app/requirements.txt
COPY amd_websearch /app/amd_websearch
COPY app /app/app
COPY corpus /app/corpus
COPY scripts /app/scripts
COPY system-prompt.txt planner-prompt.txt /app/
COPY model/qwen-2.5-0.5b.gguf /app/model/qwen-2.5-0.5b.gguf
COPY run.sh /app/run.sh

RUN chmod +x /app/run.sh /app/scripts/*.sh \
    && python -c "import torch; v=torch.__version__; assert 'rocm' in v.lower(), v; print('[build] final torch:', v)"

# The grader mounts /data and /output and runs with --network none.
VOLUME ["/data", "/output"]
ENTRYPOINT ["/app/run.sh"]
