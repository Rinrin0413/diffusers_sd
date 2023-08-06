# Diffusers (StableDiffusion)

## Setup [my environment (linux)]

```bash
git clone https://github.com/Rinrin0413/diffusers_sd.git
cd diffusers_sd
mkdir outputs

# Create and activate virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install diffusers transformers accelerate
```

## Usage (linux)

```bash
# If not already activated
source .venv/bin/activate
```

You can configure the parameters in [`settings.py`](settings.py), [`config.py`](config.py).

```bash
# Run script
python .
```
