FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Runtime pins exactly as validated on rounds 7 through R2 (setup_runpod.sh).
# PyPI torch 2.8.0 resolves to the +cu128 wheel; the base image's torchaudio/
# torchvision emit harmless version-conflict warnings that no round code touches.
RUN pip install --no-cache-dir torch==2.8.0 transformers==4.57.1 peft==0.21.0 \
    bitsandbytes==0.50.2 accelerate==1.15.0 safetensors==0.8.0 'huggingface_hub>=0.26'

# Base model baked into the image, revision-pinned. setup_runpod.sh symlinks
# /opt/p6_base into the round root when present, so pods skip the HF download.
RUN python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download('unsloth/Qwen3-4B-Instruct-2507-bnb-4bit',
                      revision='f12db89cd5156e090618dded9b4367f23f8f3b33',
                      local_dir='/opt/p6_base')
print('BASE_AT', p)
PY
RUN sha256sum /opt/p6_base/model.safetensors
