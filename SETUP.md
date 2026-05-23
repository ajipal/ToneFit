# ToneFit — Setup Guide

Step-by-step instructions for running ToneFit on a new machine.

---

## Prerequisites

Install these before starting:

| Tool | Version | Download |
|------|---------|----------|
| Python | 3.10 or 3.11 | https://www.python.org/downloads |
| Node.js | 18 or newer | https://nodejs.org |
| Git | any | https://git-scm.com |

> **Windows users:** During Python install, check **"Add Python to PATH"**.

---

## Step 1 — Clone the repository

```bash
git clone https://github.com/ajipal/ToneFit.git
cd ToneFit
```

---

## Step 2 — Place the model files

The four model files are **not included in the repository** (too large for Git).
You need to copy them manually into the `models/` folder.

```
ToneFit/
└── models/
    ├── farl_weights.pth   ← FaRL pretrained backbone (~652 MB)
    ├── farl_model.pth     ← Trained FaRL classifier head
    ├── svm_model.pkl      ← Trained SVM model
    └── scaler.pkl         ← Feature scaler for SVM
```

**Where to get them:**

| File | Source |
|------|--------|
| `farl_weights.pth` | Download from the FaRL GitHub releases (see below) |
| `farl_model.pth` | From the Kaggle/Colab training notebook output zip |
| `svm_model.pkl` | From the Kaggle/Colab training notebook output zip |
| `scaler.pkl` | From the Kaggle/Colab training notebook output zip |

**Download `farl_weights.pth` directly:**

```bash
# On Linux / macOS / Git Bash:
wget https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep64.pth -O models/farl_weights.pth
```

Or open that URL in a browser, save the file, and rename it to `farl_weights.pth` inside the `models/` folder.

---

## Step 3 — Set up the Python environment

### 3a. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

### 3b. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3c. Install the CLIP package

CLIP is not on PyPI — install it directly from GitHub:

```bash
pip install git+https://github.com/openai/CLIP.git
```

> This requires Git to be installed. If the command fails, make sure Git is on your PATH.

---

## Step 4 — Set up the frontend

```bash
cd web
npm install
cd ..
```

---

## Step 5 — First run (CLIP weight download)

The first time the ML server starts, it will automatically download the CLIP ViT-B/16 weights (~335 MB) from OpenAI. This only happens once — they are cached at `~/.cache/clip/`.

Make sure you have an internet connection for the first launch.

---

## Step 6 — Run the project

You need **two terminals** running at the same time.

### Terminal 1 — ML backend (Python / FastAPI)

From the `ToneFit/` root directory, with the virtual environment activated:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Wait until you see:

```
INFO:     Application startup complete.
```

The first startup takes 30–60 seconds while CLIP loads into memory (or downloads on first run).

### Terminal 2 — Web frontend (Next.js)

From the `ToneFit/web/` directory:

```bash
npm run dev
```

Wait until you see:

```
▲ Next.js 15.x.x
- Local: http://localhost:3000
```

---

## Step 7 — Verify everything is working

1. Open **http://localhost:3000** in your browser — you should see the ToneFit home page.

2. Check the ML backend health endpoint:
   ```
   http://localhost:8000/health
   ```
   You should get a JSON response like:
   ```json
   {
     "status": "ok",
     "models": {
       "farl": { "loaded": true, "error": null },
       "svm":  { "loaded": true, "error": null }
     }
   }
   ```
   If `"loaded": false`, check the `"error"` field for details — most likely a missing model file.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'clip'`
CLIP was not installed. Run:
```bash
pip install git+https://github.com/openai/CLIP.git
```

### `ModuleNotFoundError: No module named 'skimage'`
```bash
pip install scikit-image
```

### FaRL shows `"loaded": false` with a path error
Make sure all four files exist inside `models/`. File names are case-sensitive.

### CLIP download is stuck or fails
You can pre-download the file manually:
- URL: `https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt`
- Save it to: `C:\Users\<your-username>\.cache\clip\ViT-B-16.pt` (Windows) or `~/.cache/clip/ViT-B-16.pt` (macOS/Linux)

### Port 8000 already in use
Change the port number and update the frontend environment variable:
```bash
# Run backend on a different port
uvicorn backend.main:app --reload --port 8001

# Tell the frontend where to find it — create web/.env.local
echo "PYTHON_API_URL=http://localhost:8001" > web/.env.local
```

### PowerShell execution policy error (Windows)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### `npm install` fails
Make sure Node.js 18+ is installed: `node --version`

---

## Project URLs (when running locally)

| URL | Description |
|-----|-------------|
| http://localhost:3000 | ToneFit web app |
| http://localhost:3000/onboarding | Start color analysis |
| http://localhost:3000/compare | FaRL vs SVM model comparison |
| http://localhost:8000/health | ML backend health check |
| http://localhost:8000/docs | FastAPI auto-generated API docs |
