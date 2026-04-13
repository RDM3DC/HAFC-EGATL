# HAFC-EGATL

Hybrid Adaptive Field Computer built on an Entropy-Gated Adaptive Topological Lattice.

Live demo: [Hugging Face Space](https://huggingface.co/spaces/rdm3dc/hafc-egatl)

## Hosting on GitHub (via Docker)

Every push to `main` automatically builds a Docker image and publishes it to the
**GitHub Container Registry** (GHCR) via the included GitHub Actions workflow
(`.github/workflows/docker-publish.yml`).  No external secrets are required —
the workflow uses the built-in `GITHUB_TOKEN`.

### Pull and run the published image

```bash
docker pull ghcr.io/rdm3dc/hafc-egatl:latest
docker run -p 8501:8501 ghcr.io/rdm3dc/hafc-egatl:latest
```

Then open <http://localhost:8501> in your browser.

### One-click deploy to a cloud host

| Platform | Steps |
|---|---|
| **Render** | New → Web Service → "Deploy an existing image" → paste the GHCR URL above. |
| **Fly.io** | `fly launch --image ghcr.io/rdm3dc/hafc-egatl:latest` |
| **Railway** | New Project → Deploy from image → paste the GHCR URL. |
| **Streamlit Community Cloud** | Connect your GitHub repo; it auto-detects `app.py` + `requirements.txt` — no Docker needed. |

### Build the image locally

```bash
docker build -t hafc-egatl .
docker run -p 8501:8501 hafc-egatl
```

## Run locally (without Docker)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Sidebar Presets

The Streamlit app includes two presets in the sidebar:

- `Default`: the original baseline configuration.
- `Balanced Recovery`: a tuned configuration using `mass=-2.5`, `alpha0=4.0`, `qzw_entropy_gain=0.0`, and `alpha_pi=0.08`.

In a 20-seed sweep, `Balanced Recovery` held transfer recovery in the `0.9994-0.9995` range with zero Bott drift and zero GMRES failures.
