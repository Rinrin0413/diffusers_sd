import sys
from diffusers import (
    DDIMScheduler,
    DDIMInverseScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverSinglestepScheduler,
    DPMSolverMultistepScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KarrasVeScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    ScoreSdeVeScheduler,
    IPNDMScheduler,
    # ScoreSdeVpScheduler, # Score SDE-VP is under construction: https://huggingface.co/docs/diffusers/v0.19.3/en/api/schedulers/score_sde_vp
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    VQDiffusionScheduler,
    UniPCMultistepScheduler,
    RePaintScheduler,
)
from config import sampler

if sampler == "score_sde_vp":
    print(
        "Score SDE-VP is under construction: https://huggingface.co/docs/diffusers/v0.19.3/en/api/schedulers/score_sde_vp",
        file=sys.stderr,
    )
    sys.exit(1)

scheduler = {
    "ddim": DDIMScheduler,
    "ddim_inverse": DDIMInverseScheduler,
    "ddpm": DDPMScheduler,
    "deis": DEISMultistepScheduler,
    "singlestep_dpm_solver": DPMSolverSinglestepScheduler,
    "multistep_dpm_solver": DPMSolverMultistepScheduler,
    "heun": HeunDiscreteScheduler,
    "dpm_discrete": KDPM2DiscreteScheduler,
    "dpm_discrete_ancestral": KDPM2AncestralDiscreteScheduler,
    "stochastic_karras_ve": KarrasVeScheduler,
    "lms_discrete": LMSDiscreteScheduler,
    "pndm": PNDMScheduler,
    "score_sde_ve": ScoreSdeVeScheduler,
    "ipndm": IPNDMScheduler,
    # "score_sde_vp": ScoreSdeVpScheduler,
    "euler": EulerDiscreteScheduler,
    "euler_ancestral": EulerAncestralDiscreteScheduler,
    "vq_diffusion": VQDiffusionScheduler,
    "unipc": UniPCMultistepScheduler,
    "repaint": RePaintScheduler,
}[sampler]
