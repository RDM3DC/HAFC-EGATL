# HAFC-EGATL

Hybrid Adaptive Field Computer built on an Entropy-Gated Adaptive Topological Lattice.

HAFC-EGATL is an interactive Streamlit simulator for damage, recovery, and topological readout in a QWZ-inspired adaptive conductance lattice. The public app is designed to make one question easy to inspect live: after strong damage, does the lattice recover transport and topological structure without numerical drift?

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

## What To Try First

1. Click `Run Canonical Demo` at the top of the app.
2. Open the `Compare` tab and run `Default` vs `Balanced Recovery` side by side.
3. Open the `Robustness` tab and run a seed sweep on the current configuration.

## What The App Shows

- `Dynamics`: entropy, adaptive `π_a`, and end-to-end transfer `Y_eff` over time.
- `Damage & Recovery`: selectable damage scenarios including central strip, cross, block, top-edge cut, source-corner hit, and random bond dropout.
- `Topology`: Bott index, QWZ Chern, proxy Chern, spectral gap, and edge/plaquette signatures.
- `Network`: conductance snapshots with bond width proportional to `|g_re|` and color proportional to `|g_im|`.
- `Plaquettes`: spatial heat maps of loop signatures before damage, just after damage, and after settling.
- `Compare`: an overlaid view of the shipped presets using the same lattice, timing, damage, and seed.
- `Robustness`: multi-seed sweeps for the current sidebar configuration.
- `Data`: down-sampled CSV export plus a structured JSON run report with config, numerics, observables, compare results, and robustness metadata.

## Sidebar Presets

The Streamlit app includes two presets in the sidebar:

- `Default`: the original baseline configuration.
- `Balanced Recovery`: a tuned configuration using `mass=-2.5`, `alpha0=4.0`, `qzw_entropy_gain=0.0`, and `alpha_pi=0.08`.

In a 20-seed sweep, `Balanced Recovery` held transfer recovery in the `0.9994-0.9995` range with zero Bott drift and zero GMRES failures.

## Key Metrics

- `Y_eff recovery`: how closely end-to-end transfer returns after the damage event.
- `Sig boundary rec.` and `Sig top-edge rec.`: how strongly the edge-localized Hall-like structure returns.
- `Bott Δ`: post-settle Bott index minus pre-damage Bott index.
- `GMRES fails`: numerical solver failures accumulated during the run.

## Notes

- This repo is an interactive research demo, not a hardware implementation.
- The damage lab now supports multiple scenario masks instead of a single built-in hit pattern.
- The Data tab now exports a provenance-rich JSON report inspired by the structured artifacts used in AdaptiveCAD.
- The `Compare` and `Robustness` tabs are the fastest way to evaluate whether a parameter change is genuinely better or just better on one seed.
