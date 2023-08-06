from settings import device, vae, model
from diffusers import StableDiffusionPipeline, AutoencoderKL
import torch
from config import prompt, negative_prompt, steps, width, height, scale, seed
from datetime import datetime

print("Inference on", device.upper())

torch_dtype = torch.float16 if device == "cuda" else torch.float32
vae = AutoencoderKL.from_pretrained(vae, torch_dtype = torch_dtype).to(device)
pipeline = StableDiffusionPipeline.from_pretrained(
    model,
    vae = vae,
    torch_dtype = torch_dtype
).to(device)

generator = torch.Generator(device=device).manual_seed(seed)
if seed == -1: generator = generator.manual_seed(torch.seed())

print("Seed:", generator.initial_seed())

image = pipeline(
    prompt = prompt,
    negative_prompt = negative_prompt,
    num_inference_steps = steps,
    width = width,
    height = height,
    guidance_scale=scale,
    generator=generator
).images[0]

filename = f"outputs/{datetime.now().strftime('%Y%m%d%H%M%S%f')}.png"
image.save(filename)

print("Done:", filename)
