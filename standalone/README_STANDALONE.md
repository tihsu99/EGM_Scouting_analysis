# Standalone Baseline-vs-Custom Scouting Workflow

## Files
- `standalone/datasets.yaml`: dataset and analysis configuration.
- `standalone/run_diele_cmsstyle_compare.py`: dielectron-only comparison with CMS-style plots (recommended).
- `standalone/run_scouting_tnp_compare.py`: older TnP-based workflow.

## What it does
1. Reads baseline/custom dataset definitions from YAML.
2. Resolves ROOT files from DAS (`dasgoclient`, `prod/phys03`) or XRootD directory listing.
3. Applies event selection:
   - `DST.PFScouting_DoubleEG`,
   - apply WP to electrons first,
   - then require `nElectron == 2`,
   - opposite-sign charges.
4. Compares baseline vs custom per ID WP:
   - `no-id`, `Veto/Loose/Medium/Tight`, and `Veto/Loose/Medium/Tight-noiso`.
5. Produces CMS-style plots with ratio panel (Custom/Baseline), including:
   - dielectron mass (full, J/#psi region, Z region),
   - leading/subleading `bestTrack` variables.

## Run
Stage 1 + Stage 2 (process and plot):
```bash
python3 standalone/run_diele_cmsstyle_compare.py \
  --config standalone/datasets.yaml \
  --output standalone/output
```

Stage 2 only (replot from cached arrays, fast style iteration):
```bash
python3 standalone/run_diele_cmsstyle_compare.py \
  --config standalone/datasets.yaml \
  --output standalone/output \
  --plot-only
```

## Notes
- Fill `data_baseline` and `data_custom` in `standalone/datasets.yaml` before running data comparison.
- Electron ID is imported from your shared project file: `electron_id.py`.
- The script prints timestamped progress in shell (DAS query, sample start/end, output writes, plot generation).
- Stage-1 cached arrays are written to `standalone/output/stage1_arrays/` by default (`--stage1-dir` to override).
- For private/user datasets, use DAS dataset names (`/.../USER`) and set `das_instance: prod/phys03`; these are not filesystem paths.
- If `dataset` is a `root://...` directory (or `/store/...`), file listing is done with `xrdfs ... ls` instead of `dasgoclient`.
- If you already have explicit ROOT files in YAML, you can skip DAS lookup:
```bash
python3 standalone/run_diele_cmsstyle_compare.py \
  --config standalone/datasets.yaml \
  --output standalone/output \
  --skip-das
```
