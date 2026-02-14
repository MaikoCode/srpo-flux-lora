# SRPO + LoRA URL API on Replicate

This Cog model runs your fixed SRPO workflow on Replicate and lets the caller provide a LoRA URL per request.

## What is fixed

- Checkpoint (SRPO):
  - `https://civitai.com/api/download/models/2220553?type=Model&format=SafeTensor&size=pruned&fp=fp16`
- VAE:
  - `https://huggingface.co/Owen777/UltraFlux-v1/resolve/main/vae/diffusion_pytorch_model.safetensors`

These are downloaded on the Replicate worker during the first prediction on each container (lazy runtime init), not on your local machine.

This avoids Replicate setup timeout issues for very large models. The first request on a cold container will be slower; warm requests on the same container reuse the downloaded files.

## Inputs exposed as API

- `prompt`
- `negative_prompt`
- `lora_url` (required)
- `lora_strength`
- `width`, `height`
- `steps`, `cfg`, `guidance`, `seed`
- optional `civitai_api_key`, `huggingface_token` for private LoRA URLs

## Deploy

1. Install Cog: https://cog.run
2. Login:

```bash
cog login
```

3. Push:

```bash
cog push r8.im/<your-username>/<model-name>
```

## Example prediction

```bash
cog predict \
  -i prompt="professional headshot of a young man" \
  -i lora_url="https://huggingface.co/<user>/<repo>/resolve/main/lora.safetensors"
```

After pushing, call the same input schema through the Replicate HTTP API.
