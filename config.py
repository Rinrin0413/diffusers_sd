prompt = """1girl, tokyo, night, city, street, lights, neon,
(masterpiece:1.4),best quality,high quality,hyper quality,highres,ultra detailed,very_high_resolution,large_filesize,full color"""

negative_prompt = """nsfw, solo,
(low quality,worst quality:1.4),lowres,cropped,watermark,signature,jpeg artifacts, missing fingers,bad hands,(bad anatomy),(inaccurate limb:1.2),bad composition,inaccurate eyes,extra digit,fewer digits,(extra arms:1.2)"""

# See https://huggingface.co/docs/diffusers/v0.19.3/en/api/schedulers/overview#schedulers-summary
sampler = "euler"

steps = 12

# Must be divisible by 8
width = 688
height = 512

scale = 7.0

# -1 for random
seed = -1
