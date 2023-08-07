from settings import device, repo, vae
from diffusers import StableDiffusionPipeline, AutoencoderKL
import torch
from config import (
    prompt,
    negative_prompt,
    steps,
    width,
    height,
    batch_count,
    batch_size,
    scale,
    seed,
)
from schedulers import scheduler
from datetime import datetime

print("Inference on", device.upper())

torch_dtype = torch.float16 if device == "cuda" else torch.float32

if vae == "auto":
    vae = AutoencoderKL.from_pretrained(
        repo, subfolder="vae", torch_dtype=torch_dtype
    ).to(device)
else:
    vae = AutoencoderKL.from_pretrained(vae, torch_dtype=torch_dtype).to(device)
scheduler = scheduler.from_pretrained(repo, subfolder="scheduler")

pipeline = StableDiffusionPipeline.from_pretrained(
    repo, vae=vae, scheduler=scheduler, torch_dtype=torch_dtype
).to(device)

generator = torch.Generator(device=device).manual_seed(seed)
if seed == -1:
    generator = generator.manual_seed(torch.seed())

print("Seed:", generator.initial_seed())

before = datetime.now()

for i in range(batch_count):
    images = pipeline(
        prompt=[prompt] * batch_size,
        negative_prompt=[negative_prompt] * batch_size,
        num_inference_steps=steps,
        width=width,
        height=height,
        guidance_scale=scale,
        generator=generator,
    ).images

    for i, image in enumerate(images):
        filename = f"outputs/{datetime.now().strftime('%Y%m%d%H%M%S%f')}.png"
        image.save(filename)
        print("Saved", filename)

after = datetime.now()
print(f"Total generation time: {after - before}s")

print("Done!")
