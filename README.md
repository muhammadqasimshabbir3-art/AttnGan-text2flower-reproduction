# AttnGAN text-to-flower (reproduction)

This repository contains a reproduction of the AttnGAN-based text-to-flower model and saved training artifacts. It also includes an (optional) Gradio-based `app.py` intended to expose a simple UI for generating flower images from text descriptions.

Contents
- `app.py` — (optional) Gradio app entry point. When present, it should load model weights from the `training_results_epoches_79/` folder and launch a Gradio interface.
- `training_results_epoches_79/` — saved model checkpoints, encoders, and example images created during training.
- `text-to-flower-attngan.ipynb` — notebook used during experimentation and reproduction.

Prerequisites
- Python 3.8 - 3.11
- A recent pip (pip>=22 recommended)
- For GPU acceleration, a matching PyTorch + CUDA build. Installing the correct PyTorch wheel for your CUDA version is recommended. See https://pytorch.org/get-started/locally/ for the appropriate install command.

Install (recommended)
1. Create and activate a virtual environment (optional but recommended):

   python3 -m venv .venv
   source .venv/bin/activate

2. Install dependencies listed in `requirements.txt`.

   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt

3. IMPORTANT: Install PyTorch separately with the command appropriate for your platform/CUDA. Example (CPU-only):

   python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

Or use the command generated on https://pytorch.org/get-started/locally/ to install a CUDA-enabled build.

Running the Gradio app (local)
1. Make sure `app.py` is implemented to load the model and expose a Gradio interface. The repository contains saved checkpoints under `training_results_epoches_79/` which `app.py` can load (for example `netG_epoch_79.pth`, `image_encoder79.pth`, `text_encoder79.pth`).

2. Run the app:

   python app.py

If `app.py` calls `gradio.Interface.launch(share=True)` you can also expose a public link; otherwise it will be available at http://localhost:7860 by default.

Deploying (Hugging Face Spaces / other hosts)
- Push the repository (including `app.py` and `requirements.txt`) to the hosting platform. For Spaces, add the correct `requirements.txt` and ensure `app.py` starts the Gradio interface.
- On Spaces, ensure large model files are either included (if small) or downloaded at runtime (recommended) because Spaces has repository size limits.

Notes about the models and data
- The `training_results_epoches_79/` directory contains model checkpoints and example images from training. If you move these files, update `app.py` or environment variables so the app can find them.
- The repository does not include a dataset; training was done externally. Use the saved checkpoints if you want to run inference only.

Troubleshooting
- If you run into CUDA / PyTorch version mismatches, reinstall PyTorch with the correct CUDA version.
- If `gradio` is not found after installing `requirements.txt`, double-check your virtual environment activation and pip target Python.

Contact / License
- This repo is a reproduction and is intended for experimentation. Check upstream licensing for any reused code/artifacts.

If you want, I can also: (a) implement a minimal Gradio `app.py` that loads the saved generator and text/image encoders and exposes a simple UI, or (b) add instructions that programmatically download model weights at first run. Tell me which and I'll implement it.

App and Hugging Face Spaces
---------------------------------
I added a ready-to-run `app.py` that implements a Gradio interface for inference. The app:
- Loads model checkpoints from `training_results_epoches_79/`:
  - `netG_epoch_79.pth` (generator)
  - `text_encoder79.pth` (text encoder)
  - `captions_fixed.pickle` (vocabulary mappings)
- Tokenizes an input text, converts it to the model's indices and runs the generator to produce an image.

To run locally:

```bash
# create and activate virtualenv (optional)
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
# install PyTorch separately for the correct CUDA/CPU
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

python app.py
```

On Hugging Face Spaces
- Commit `app.py`, `requirements.txt` and the small files to your repo. Large model files (checkpoint .pth) may exceed Spaces repo size limits. Options:
  1. Upload checkpoints as Git LFS / to a remote storage and update `app.py` to download them on first run.
  2. Host model weights in an external storage (S3, huggingface.co/hub) and let `app.py` download them at startup.

By default `app.py` looks for the checkpoints under `training_results_epoches_79/`. If you deploy to Spaces and have limited repo size, either include a small downloader in `app.py` or move the checkpoints to an accessible URL and update the paths.

If you'd like I can now:
- add automatic checkpoint download to `app.py` (useful for Spaces), or
- reduce model file size by exporting a smaller generator, or
- further pin `requirements.txt` to an exact PyTorch/CUDA combination.



