# Standalone Baseline-vs-Custom Scouting Workflow

## Files
- `standalone/datasets.yaml`: dataset and analysis configuration.
- `standalone/run_scouting_tnp_compare.py`: self-contained Coffea pipeline.

## What it does
1. Reads baseline/custom dataset definitions from YAML.
2. Resolves ROOT files from DAS (`dasgoclient`) and applies redirector `root://xrootd-cms.infn.it//`.
3. Runs Coffea processing for:
   - leading-pair invariant mass,
   - Tag-and-Probe tuples,
   - trigger pass/fail columns for efficiency.
4. Produces baseline-vs-custom plots for MC/data (if both are provided):
   - invariant mass,
   - probe electron distributions (`probe_pt`, `|probe_eta|`),
   - efficiency vs probe pT.

## Run
```bash
python3 standalone/run_scouting_tnp_compare.py \
  --config standalone/datasets.yaml \
  --output standalone/output
```

## Notes
- Fill `data_baseline` and `data_custom` in `standalone/datasets.yaml` before running data comparison.
- Electron ID is imported from your shared project file: `electron_id.py`.
- The script prints timestamped progress in shell (DAS query, sample start/end, output writes, plot generation).
- For private/user datasets, use DAS dataset names (`/.../USER`) and set `das_instance: prod/phys03`; these are not filesystem paths.
- If `dataset` is a `root://...` directory (or `/store/...`), file listing is done with `xrdfs ... ls` instead of `dasgoclient`.
- If you already have explicit ROOT files in YAML, you can skip DAS lookup:
```bash
python3 standalone/run_scouting_tnp_compare.py \
  --config standalone/datasets.yaml \
  --output standalone/output \
  --skip-das
```
