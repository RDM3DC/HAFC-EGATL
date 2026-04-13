# HAFC-EGATL

Hybrid Adaptive Field Computer built on an Entropy-Gated Adaptive Topological Lattice.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Sidebar Presets

The Streamlit app includes two presets in the sidebar:

- `Default`: the original baseline configuration.
- `Balanced Recovery`: a tuned configuration using `mass=-2.5`, `alpha0=4.0`, `qzw_entropy_gain=0.0`, and `alpha_pi=0.08`.

In a 20-seed sweep, `Balanced Recovery` held transfer recovery in the `0.9994-0.9995` range with zero Bott drift and zero GMRES failures.
