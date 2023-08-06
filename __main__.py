from settings import infer_on, model
from diffusers import StableDiffusionPipeline
import torch
from config import prompt, negative_prompt, steps, width, height, scale
from datetime import datetime

print("Inference on", infer_on.upper())

pipeline = StableDiffusionPipeline.from_pretrained(
    model,
    torch_dtype = torch.float16 if infer_on == "cuda" else torch.float32
).to(infer_on)

image = pipeline(
    prompt = prompt,
    negative_prompt = negative_prompt,
    num_inference_steps = steps,
    width = width,
    height = height,
    guidance_scale=scale,
).images[0]

filename = f"outputs/{datetime.now().strftime('%Y%m%d%H%M%S%f')}.png"
image.save(filename)

print("Done:", filename)
