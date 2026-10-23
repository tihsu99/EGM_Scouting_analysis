# CMS EGM Scouting Analysis

Analysis and plotting scripts for CMS Run 3 electron/photon scouting performance studies, focusing on 2024 data. The repository includes cut-based electron identification (ID) studies, dielectron invariant-mass distributions, electron transverse-momentum spectra, and trigger-efficiency measurements.

## Main scripts

- `electron_id.py` — cut-based scouting electron ID working points, with and without isolation requirements.
- `tnp_ntupler.py` — processes scouting NanoAOD files and saves dielectron mass distributions and tag-and-probe variables.
- `draw_inv_mass_fit.py` — plots dielectron invariant-mass spectra for different ID working points and fits low-mass resonances.
- `trig_summary.py` — tag-and-probe trigger-efficiency studies.
- `pt_spectrum_ntupler_v2.py` and `plot_pt_spectrum_v2.py` — electron transverse-momentum spectra.
- `id_ntupler.py` — ntuple production for electron ID studies.
- `env.sh` — CERN LCG software environment setup.

## Dielectron invariant-mass plots

The invariant-mass study compares Loose, Medium, and Tight scouting electron IDs, with and without isolation. The plotting script currently produces the barrel–barrel (EBEB) spectrum, including the low-mass resonance region and the Z-boson peak.

Starting from scouting NanoAOD ROOT files:

```bash
source env.sh

python tnp_ntupler.py \
    --files /path/to/scouting_1.root /path/to/scouting_2.root \
    --store_path output \
    --index 0

python draw_inv_mass_fit.py \
    --store_path output \
    --plot_path plots
```

The first step writes `inv_mass_data_0.npz` (mass arrays) and `output_0.h5` (tag-and-probe variables). The second step reads the `inv_mass_data*.npz` files in the output directory and saves the mass-spectrum plots as PNG and PDF files, including `inv-mass-EBEB.png` and `inv-mass-EBEB.pdf`.

The scripts use Python, Coffea/Awkward Array, ROOT, and common scientific Python packages. On CERN machines, `env.sh` provides an LCG environment; additional packages may be needed depending on the setup.
