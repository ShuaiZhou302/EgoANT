# Models

`providers.json` contains API endpoints and model IDs. Hand-reconstruction
weights are installed locally under `hand_reconstruction/` and are intentionally
excluded from Git.

Installed by `scripts/setup_models.sh`:

- `hawor/checkpoints/hawor.ckpt` (about 3.1 GB);
- `hawor/external/detector.pt` (about 52 MB);
- `hawor/model_config.yaml`.

MANO is license-gated and must be downloaded from its official website. Place
`MANO_RIGHT.pkl` in `models/hand_reconstruction/mano/` before running the
pipeline. Model weights and the optional VITRA/HaWoR reconstruction path retain
their upstream licenses; see `THIRD_PARTY.md`.
