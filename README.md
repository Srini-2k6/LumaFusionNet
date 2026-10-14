# LumaFusionNet Web — Low-Light Image Lab

A FastAPI + PyTorch web app for the LumaFusionNet dual-domain low-light image enhancement research prototype. The interface includes a persistent dark/light theme toggle, drag-and-drop image upload, before/after previews, download and reset controls, plus a clickable system-model diagram with a step-by-step explanation.

## Run locally in VS Code (Windows)

1. Extract the ZIP and open the `LumaFusionNet_Web` folder in VS Code.
2. In the terminal, create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

3. Install dependencies (PowerShell activation is not required):

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

4. Start the server:

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn app:app --reload
   ```

5. Visit http://127.0.0.1:8000. API status is available at http://127.0.0.1:8000/api/status.

## Deploy to Render

This repository includes `render.yaml` and a `Procfile` for a standard Python web service.

1. Create a GitHub repository and upload the contents of this folder (the `app.py`, `model.py`, `enhancer.py`, `requirements.txt`, `static/`, and deployment files should be at the repository root). Do not upload `.venv/`, `__pycache__/`, or private datasets.
2. Sign in to Render and choose **New + → Web Service**, then connect that GitHub repository. If Render detects `render.yaml`, review the service settings it proposes.
3. Use build command `pip install -r requirements.txt` and start command `uvicorn app:app --host 0.0.0.0 --port $PORT`. The health check path is `/api/status`.
4. Deploy and open the `onrender.com` URL Render provides. Test the landing page, `/api/status`, upload, enhancement, and download.
5. If the build runs out of memory or times out while installing PyTorch, use a paid instance with more build/runtime resources or deploy from a Docker image with a CPU-only PyTorch wheel. Free-tier resource limits can change, so check the current provider limits before relying on it.

### Git commands

From the extracted project folder:

```bash
git init
git add .
git commit -m "Refresh LumaFusionNet frontend and add deployment config"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the example remote URL with your actual GitHub repository URL.

## Demo mode versus neural inference

The website remains usable without a trained checkpoint, but the backend then uses a classical gamma/contrast enhancement fallback. This is labelled as demo mode and is **not** neural inference. For real model inference, place compatible trained weights at `weights/lumafusionnet_best.pth` and redeploy/restart the service. Ensure the checkpoint matches `model.py`. A deployed service's local filesystem may be ephemeral; keep a backup of trained weights and use persistent/object storage if the provider requires it.

## Architecture

Browser upload → FastAPI → image preprocessing → spatial and FFT amplitude/phase feature blocks → feature fusion → encoder/bottleneck attention/decoder with skip connections → residual RGB reconstruction → PNG response.

The UI diagram is a conceptual architecture overview; the model implementation is the source of truth for the exact operations. Train on paired low-light/normal images and report validated metrics such as PSNR and SSIM before making research performance claims.

# LumaFusionNet