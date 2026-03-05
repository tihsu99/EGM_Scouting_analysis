#!/usr/bin/env python3
"""Standalone Coffea workflow for baseline-vs-custom scouting comparisons.

Features:
1) Read dataset config from YAML.
2) Resolve DAS files and prepend xrootd redirector.
3) Run invariant-mass reconstruction and Tag-and-Probe extraction with Coffea.
4) Build baseline-vs-custom plots for invariant mass, probe variables, and efficiency.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vector
import yaml
from coffea import processor
from coffea.nanoevents import NanoAODSchema
from coffea.processor import FuturesExecutor, Runner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from electron_id import electron_id_mask

vector.register_awkward()

WP_ORDER = ["Veto", "Loose", "Medium", "Tight"]


def log(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def wp_field_name(wp_name: str) -> str:
    return wp_name if wp_name.endswith("_id") else f"{wp_name}_id"


def normalize_redirector(redirector: str) -> str:
    if not redirector:
        return ""
    if redirector.endswith("//"):
        return redirector
    if redirector.endswith("/"):
        return redirector + "/"
    return redirector + "//"


def prepend_redirector(path: str, redirector: str) -> str:
    if path.startswith("root://"):
        return path
    redirector_norm = normalize_redirector(redirector)
    if path.startswith("/"):
        return redirector_norm + path.lstrip("/")
    return redirector_norm + path


def split_xrootd_url(xrootd_url: str) -> Tuple[str, str]:
    if not xrootd_url.startswith("root://"):
        raise ValueError(f"Not an xrootd URL: {xrootd_url}")
    remainder = xrootd_url[len("root://") :]
    host, sep, raw_path = remainder.partition("/")
    if not sep or not host:
        raise ValueError(f"Invalid xrootd URL: {xrootd_url}")
    path = "/" + raw_path.lstrip("/")
    return host, path


def query_xrootd_files(xrootd_dir: str) -> List[str]:
    host, directory = split_xrootd_url(xrootd_dir)
    log(f"List XRootD directory: {xrootd_dir}")
    cmd = ["xrdfs", host, "ls", directory]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    entries = [line.strip() for line in proc.stdout.splitlines() if line.strip()]

    files: List[str] = []
    for entry in entries:
        if not entry.endswith(".root"):
            continue
        if entry.startswith("root://"):
            files.append(entry)
        elif entry.startswith("/"):
            files.append(f"root://{host}//{entry.lstrip('/')}")
        else:
            files.append(f"root://{host}//{directory.rstrip('/')}/{entry.lstrip('./')}")
    log(f"XRootD returned {len(files)} ROOT files for directory: {xrootd_dir}")
    return files


def query_das_files(dataset: str, das_instance: str) -> List[str]:
    log(f"Query DAS for dataset: {dataset} (instance={das_instance})")
    cmd = ["dasgoclient", "-query", f"file dataset={dataset} instance={das_instance}"]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    files = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    log(f"DAS returned {len(files)} files for dataset: {dataset}")
    return files


def resolve_sample_files(
    sample_cfg: Dict, redirector: str, max_files: int | None, skip_das: bool, das_instance: str
) -> List[str]:
    files: List[str] = []

    for item in sample_cfg.get("files", []) or []:
        files.append(prepend_redirector(str(item), redirector))

    dataset = (sample_cfg.get("dataset") or "").strip()
    if dataset and not files:
        if dataset.startswith("root://"):
            files.extend(query_xrootd_files(dataset))
        elif dataset.startswith("/store/"):
            xrd_dir = prepend_redirector(dataset, redirector)
            files.extend(query_xrootd_files(xrd_dir))
        else:
            if skip_das:
                raise RuntimeError(f"DAS lookup disabled but sample has only dataset entry: {dataset}")
            das_files = query_das_files(dataset, das_instance)
            files.extend(prepend_redirector(f, redirector) for f in das_files)

    deduped = list(dict.fromkeys(files))
    if max_files is not None:
        deduped = deduped[:max_files]
    log(f"Resolved {len(deduped)} input files")
    return deduped


def col_accumulator(dtype: np.dtype) -> processor.column_accumulator:
    return processor.column_accumulator(np.array([], dtype=dtype))


class ScoutingTnPProcessor(processor.ProcessorABC):
    def __init__(self, selection_cfg: Dict, target_triggers: List[str]):
        self.selection_cfg = selection_cfg
        self.target_triggers = target_triggers

        tnp_base_columns = [
            "inv_mass",
            "tag_pt",
            "tag_eta",
            "tag_id",
            "tag_noiso_id",
            "probe_pt",
            "probe_eta",
            "probe_id",
            "probe_noiso_id",
            "opposite_charge",
        ]
        tnp_columns = tnp_base_columns + target_triggers

        self._accumulator = processor.dict_accumulator(
            {
                "pair_mass": col_accumulator(np.float32),
                "tnp": processor.dict_accumulator({name: col_accumulator(np.float32) for name in tnp_columns}),
            }
        )

    @property
    def accumulator(self):
        return self._accumulator

    @staticmethod
    def _add_column(output_dict: Dict, key: str, array_like: ak.Array | np.ndarray) -> None:
        data = np.asarray(ak.to_numpy(array_like), dtype=np.float32)
        if data.size == 0:
            return
        output_dict[key] += processor.column_accumulator(data)

    @staticmethod
    def _get_field_or(default: ak.Array, obj: ak.Array, field_name: str) -> ak.Array:
        return obj[field_name] if field_name in obj.fields else default

    @staticmethod
    def _zero_like_events(events: ak.Array) -> ak.Array:
        if len(events) == 0:
            return ak.Array(np.array([], dtype=np.float32))
        if "run" in events.fields:
            return ak.zeros_like(events.run, dtype=np.float32)
        if "event" in events.fields:
            return ak.zeros_like(events.event, dtype=np.float32)
        return ak.Array(np.zeros(len(events), dtype=np.float32))

    def _trigger_column(self, events: ak.Array, trigger_name: str) -> ak.Array:
        if "DST" in events.fields and trigger_name in events.DST.fields:
            return ak.values_astype(events.DST[trigger_name], np.float32)
        if trigger_name in events.fields:
            return ak.values_astype(events[trigger_name], np.float32)
        return self._zero_like_events(events)

    def process(self, events):
        output = self.accumulator.identity()

        if "ScoutingElectron" not in events.fields:
            return output

        electrons = events.ScoutingElectron
        selection = (electrons.pt > self.selection_cfg["electron_pt_min"]) & (
            abs(electrons.bestTrack_eta) < self.selection_cfg["electron_abs_eta_max"]
        )
        electrons = electrons[selection]

        events["good_electrons"] = electrons
        events = events[ak.num(events.good_electrons) > 1]
        if len(events) == 0:
            return output

        electrons = events.good_electrons
        rho = events.ScoutingRho.fixedGridRhoFastjetAll if "ScoutingRho" in events.fields else self._zero_like_events(events)

        for wp in WP_ORDER:
            electrons[f"{wp}_id"] = electron_id_mask(electrons, rho, wp, use_iso=True)
            electrons[f"{wp}-noiso_id"] = electron_id_mask(electrons, rho, wp, use_iso=False)

        cutbased_id = ak.zeros_like(electrons.pt, dtype=np.int8)
        cutbased_id = ak.where(electrons["Veto_id"], 1, cutbased_id)
        cutbased_id = ak.where(electrons["Loose_id"], 2, cutbased_id)
        cutbased_id = ak.where(electrons["Medium_id"], 3, cutbased_id)
        cutbased_id = ak.where(electrons["Tight_id"], 4, cutbased_id)
        electrons["cutbased_id"] = cutbased_id

        cutbased_noiso_id = ak.zeros_like(electrons.pt, dtype=np.int8)
        cutbased_noiso_id = ak.where(electrons["Veto-noiso_id"], 1, cutbased_noiso_id)
        cutbased_noiso_id = ak.where(electrons["Loose-noiso_id"], 2, cutbased_noiso_id)
        cutbased_noiso_id = ak.where(electrons["Medium-noiso_id"], 3, cutbased_noiso_id)
        cutbased_noiso_id = ak.where(electrons["Tight-noiso_id"], 4, cutbased_noiso_id)
        electrons["cutbased_noiso_id"] = cutbased_noiso_id

        electrons = electrons[ak.argsort(electrons.pt, axis=-1, ascending=False)]
        eta_for_p4 = self._get_field_or(electrons.eta, electrons, "bestTrack_etaMode")
        phi_for_p4 = self._get_field_or(electrons.phi, electrons, "bestTrack_phiMode")

        electrons["momentum"] = ak.zip(
            {
                "pt": electrons.pt,
                "eta": eta_for_p4,
                "phi": phi_for_p4,
                "mass": ak.zeros_like(electrons.pt),
            },
            with_name="Momentum4D",
        )

        lead = electrons[:, 0]
        sublead = electrons[:, 1]
        pair_mass = (lead.momentum + sublead.momentum).mass
        if self.selection_cfg["require_opposite_charge"]:
            pair_mass = pair_mass[(lead.bestTrack_charge * sublead.bestTrack_charge) < 0]
        self._add_column(output, "pair_mass", pair_mass)

        tag_id_field = wp_field_name(self.selection_cfg["tag_id"])
        probe_pre_field = wp_field_name(self.selection_cfg["probe_pre_id"])
        if tag_id_field not in electrons.fields or probe_pre_field not in electrons.fields:
            return output

        electrons["pass_tag"] = (
            (electrons.pt > self.selection_cfg["tag_pt_min"])
            & (abs(electrons.eta) < self.selection_cfg["tag_abs_eta_max"])
            & electrons[tag_id_field]
        )
        electrons["pass_probe_pre"] = electrons[probe_pre_field]

        event_selection = ((electrons[:, 0].pass_tag) & (electrons[:, 1].pass_probe_pre)) | (
            (electrons[:, 1].pass_tag) & (electrons[:, 0].pass_probe_pre)
        )

        electrons_sel = electrons[event_selection]
        events_sel = events[event_selection]

        for tag_idx, probe_idx in ((0, 1), (1, 0)):
            tag = electrons_sel[:, tag_idx]
            probe = electrons_sel[:, probe_idx]
            orientation_mask = tag.pass_tag

            tag = tag[orientation_mask]
            probe = probe[orientation_mask]
            events_oriented = events_sel[orientation_mask]
            if len(tag) == 0:
                continue

            inv_mass = (tag.momentum + probe.momentum).mass
            opposite_charge = (tag.bestTrack_charge * probe.bestTrack_charge) < 0

            self._add_column(output["tnp"], "inv_mass", inv_mass)
            self._add_column(output["tnp"], "tag_pt", tag.pt)
            self._add_column(output["tnp"], "tag_eta", tag.eta)
            self._add_column(output["tnp"], "tag_id", tag.cutbased_id)
            self._add_column(output["tnp"], "tag_noiso_id", tag.cutbased_noiso_id)
            self._add_column(output["tnp"], "probe_pt", probe.pt)
            self._add_column(output["tnp"], "probe_eta", probe.eta)
            self._add_column(output["tnp"], "probe_id", probe.cutbased_id)
            self._add_column(output["tnp"], "probe_noiso_id", probe.cutbased_noiso_id)
            self._add_column(output["tnp"], "opposite_charge", opposite_charge)

            for trigger_name in self.target_triggers:
                trigger_values = self._trigger_column(events_oriented, trigger_name)
                self._add_column(output["tnp"], trigger_name, trigger_values)

        return output

    def postprocess(self, accumulator):
        return accumulator


def build_runner(cfg: Dict) -> Runner:
    worker_cfg = cfg.get("runner", {})
    workers = int(worker_cfg.get("workers", max(1, (os.cpu_count() or 2) - 1)))
    chunksize = int(worker_cfg.get("chunksize", 300000))
    maxchunks = worker_cfg.get("maxchunks")
    show_progress = bool(worker_cfg.get("show_progress", True))

    log(f"Create Coffea runner: workers={workers}, chunksize={chunksize}, maxchunks={maxchunks}, progress={show_progress}")
    return Runner(
        executor=FuturesExecutor(compression=None, workers=workers, status=show_progress),
        schema=NanoAODSchema,
        chunksize=chunksize,
        maxchunks=maxchunks,
    )


def run_sample(sample_name: str, files: List[str], cfg: Dict) -> Tuple[np.ndarray, pd.DataFrame]:
    runner = build_runner(cfg)
    processor_instance = ScoutingTnPProcessor(cfg["selection"], cfg["tnp"]["target_triggers"])

    log(f"[{sample_name}] Start Coffea processing with {len(files)} files")
    start = perf_counter()
    output = runner({"sample": files}, treename=cfg.get("tree_name", "Events"), processor_instance=processor_instance)
    elapsed = perf_counter() - start

    pair_mass = output["pair_mass"].value
    tnp_df = pd.DataFrame({k: v.value for k, v in output["tnp"].items()})
    log(f"[{sample_name}] Coffea finished in {elapsed:.1f}s, pair_mass={pair_mass.size}, tnp_rows={len(tnp_df)}")
    return pair_mass, tnp_df


def efficiency_in_bins(
    df: pd.DataFrame,
    trigger_name: str,
    pt_bins: np.ndarray,
    eta_range: Tuple[float, float],
    mass_window: Tuple[float, float],
    require_opposite_charge: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eta_min, eta_max = eta_range
    mmin, mmax = mass_window

    mask = (
        (np.abs(df["probe_eta"]) >= eta_min)
        & (np.abs(df["probe_eta"]) < eta_max)
        & (df["inv_mass"] >= mmin)
        & (df["inv_mass"] < mmax)
    )
    if require_opposite_charge:
        mask &= df["opposite_charge"] > 0.5

    denom_pt = df.loc[mask, "probe_pt"].to_numpy()
    num_mask = mask & (df[trigger_name] > 0.5)
    num_pt = df.loc[num_mask, "probe_pt"].to_numpy()

    denom, edges = np.histogram(denom_pt, bins=pt_bins)
    numer, _ = np.histogram(num_pt, bins=pt_bins)

    eff = np.divide(numer, denom, out=np.zeros_like(numer, dtype=float), where=denom > 0)
    err = np.zeros_like(eff)
    valid = denom > 0
    err[valid] = np.sqrt(eff[valid] * (1.0 - eff[valid]) / denom[valid])

    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, eff, err, denom


def plot_hist_compare(
    baseline: np.ndarray,
    custom: np.ndarray,
    title: str,
    xlabel: str,
    output_path: Path,
    bins: int = 80,
    hist_range: Tuple[float, float] | None = None,
    density: bool = True,
) -> None:
    plt.figure(figsize=(8, 6))
    if baseline.size:
        plt.hist(baseline, bins=bins, range=hist_range, histtype="step", linewidth=2, density=density, label="baseline")
    if custom.size:
        plt.hist(custom, bins=bins, range=hist_range, histtype="step", linewidth=2, density=density, label="custom")
    plt.xlabel(xlabel)
    plt.ylabel("a.u." if density else "Entries")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_efficiency_compare(
    baseline_df: pd.DataFrame,
    custom_df: pd.DataFrame,
    trigger_name: str,
    pt_bins: np.ndarray,
    eta_range: Tuple[float, float],
    mass_window: Tuple[float, float],
    require_opposite_charge: bool,
    output_path: Path,
    title_prefix: str,
) -> None:
    b_x, b_eff, b_err, _ = efficiency_in_bins(
        baseline_df, trigger_name, pt_bins, eta_range, mass_window, require_opposite_charge
    )
    c_x, c_eff, c_err, _ = efficiency_in_bins(custom_df, trigger_name, pt_bins, eta_range, mass_window, require_opposite_charge)

    plt.figure(figsize=(8, 6))
    plt.errorbar(b_x, b_eff, yerr=b_err, fmt="o", label="baseline")
    plt.errorbar(c_x, c_eff, yerr=c_err, fmt="s", label="custom")
    plt.ylim(0.0, 1.1)
    plt.xscale("log")
    plt.xlabel("Probe electron pT [GeV]")
    plt.ylabel("Efficiency")
    eta_min, eta_max = eta_range
    plt.title(f"{title_prefix}: {trigger_name} ({eta_min} <= |eta| < {eta_max})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_sample_outputs(sample_dir: Path, pair_mass: np.ndarray, tnp_df: pd.DataFrame) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.save(sample_dir / "pair_mass.npy", pair_mass)
    tnp_df.to_csv(sample_dir / "tnp_table.csv", index=False)


def compare_kind(kind: str, by_variant: Dict[str, Dict], cfg: Dict, output_dir: Path) -> None:
    if "baseline" not in by_variant or "custom" not in by_variant:
        log(f"[{kind}] Skip comparisons: need both baseline and custom")
        return

    baseline = by_variant["baseline"]
    custom = by_variant["custom"]

    baseline_mass = baseline["pair_mass"]
    custom_mass = custom["pair_mass"]

    kind_dir = output_dir / f"compare_{kind}"
    kind_dir.mkdir(parents=True, exist_ok=True)
    log(f"[{kind}] Building comparison plots in {kind_dir}")

    plot_hist_compare(
        baseline_mass,
        custom_mass,
        title=f"{kind.upper()} leading-pair invariant mass",
        xlabel="m(ee) [GeV]",
        output_path=kind_dir / "inv_mass_leading_pair_baseline_vs_custom.png",
        bins=120,
        hist_range=(0, 200),
        density=True,
    )
    log(f"[{kind}] Wrote leading-pair invariant-mass comparison")

    mmin, mmax = cfg["selection"]["mass_window"]
    require_oc = bool(cfg["selection"]["require_opposite_charge"])

    bdf = baseline["tnp_df"].copy()
    cdf = custom["tnp_df"].copy()

    if not bdf.empty:
        bmask = (bdf["inv_mass"] >= mmin) & (bdf["inv_mass"] < mmax)
        if require_oc:
            bmask &= bdf["opposite_charge"] > 0.5
        bdf = bdf[bmask]
    if not cdf.empty:
        cmask = (cdf["inv_mass"] >= mmin) & (cdf["inv_mass"] < mmax)
        if require_oc:
            cmask &= cdf["opposite_charge"] > 0.5
        cdf = cdf[cmask]

    plot_hist_compare(
        bdf["inv_mass"].to_numpy() if not bdf.empty else np.array([]),
        cdf["inv_mass"].to_numpy() if not cdf.empty else np.array([]),
        title=f"{kind.upper()} TnP invariant mass (selected)",
        xlabel="m(tag, probe) [GeV]",
        output_path=kind_dir / "inv_mass_tnp_baseline_vs_custom.png",
        bins=80,
        hist_range=(mmin, mmax),
        density=True,
    )
    log(f"[{kind}] Wrote TnP invariant-mass comparison")

    plot_hist_compare(
        bdf["probe_pt"].to_numpy() if not bdf.empty else np.array([]),
        cdf["probe_pt"].to_numpy() if not cdf.empty else np.array([]),
        title=f"{kind.upper()} probe pT",
        xlabel="Probe pT [GeV]",
        output_path=kind_dir / "probe_pt_baseline_vs_custom.png",
        bins=60,
        hist_range=(0, 300),
        density=True,
    )

    plot_hist_compare(
        np.abs(bdf["probe_eta"].to_numpy()) if not bdf.empty else np.array([]),
        np.abs(cdf["probe_eta"].to_numpy()) if not cdf.empty else np.array([]),
        title=f"{kind.upper()} |probe eta|",
        xlabel="|probe eta|",
        output_path=kind_dir / "probe_abs_eta_baseline_vs_custom.png",
        bins=50,
        hist_range=(0, 2.5),
        density=True,
    )
    log(f"[{kind}] Wrote probe-variable distributions")

    pt_bins = np.asarray(cfg["tnp"]["pt_bins"], dtype=float)
    eta_bins = [tuple(x) for x in cfg["tnp"]["eta_bins"]]
    for trigger_name in cfg["tnp"]["target_triggers"]:
        if trigger_name not in bdf.columns or trigger_name not in cdf.columns:
            continue
        for eta_min, eta_max in eta_bins:
            eta_tag = f"eta_{eta_min:.2f}_{eta_max:.2f}".replace(".", "p")
            plot_efficiency_compare(
                bdf,
                cdf,
                trigger_name,
                pt_bins,
                (eta_min, eta_max),
                (mmin, mmax),
                require_oc,
                kind_dir / f"eff_{trigger_name}_{eta_tag}_baseline_vs_custom.png",
                title_prefix=f"{kind.upper()} efficiency",
            )
            log(f"[{kind}] Wrote efficiency plot for {trigger_name}, eta bin {eta_min}-{eta_max}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone baseline/custom scouting comparison with Coffea")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--skip-das", action="store_true", help="Disable DAS lookup (use only explicit file lists)")
    parser.add_argument("--max-files", type=int, default=None, help="Optional override for max files per sample")
    args = parser.parse_args()

    log(f"Load config: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")

    redirector = cfg.get("redirector", "")
    das_instance = cfg.get("das_instance", "prod/global")
    max_files = args.max_files if args.max_files is not None else cfg.get("max_files_per_sample")
    total_samples = len(cfg.get("samples", {}))
    log(
        f"Configured samples: {total_samples}, max_files_per_sample={max_files}, "
        f"skip_das={args.skip_das}, das_instance={das_instance}"
    )

    processed_samples: Dict[str, Dict] = {}
    summary: Dict[str, Dict] = {}

    for idx, (sample_name, sample_cfg) in enumerate(cfg.get("samples", {}).items(), start=1):
        log(f"[{idx}/{total_samples}] Prepare sample: {sample_name}")
        try:
            files = resolve_sample_files(sample_cfg, redirector, max_files, skip_das=args.skip_das, das_instance=das_instance)
        except Exception as exc:
            log(f"[{sample_name}] File resolution failed: {exc}")
            summary[sample_name] = {
                "status": "failed_file_resolution",
                "reason": str(exc),
                "n_files": 0,
            }
            continue

        if not files:
            log(f"[{sample_name}] Skip: no files")
            summary[sample_name] = {
                "status": "skipped_no_files",
                "n_files": 0,
            }
            continue

        pair_mass, tnp_df = run_sample(sample_name, files, cfg)

        sample_dir = output_dir / sample_name
        log(f"[{sample_name}] Save outputs to {sample_dir}")
        save_sample_outputs(sample_dir, pair_mass, tnp_df)

        processed_samples[sample_name] = {
            "kind": sample_cfg.get("kind", "unknown"),
            "variant": sample_cfg.get("variant", "unknown"),
            "pair_mass": pair_mass,
            "tnp_df": tnp_df,
            "n_files": len(files),
        }
        summary[sample_name] = {
            "status": "processed",
            "n_files": len(files),
            "n_pair_mass_entries": int(pair_mass.size),
            "n_tnp_rows": int(len(tnp_df)),
        }

    by_kind_variant: Dict[str, Dict[str, Dict]] = {}
    for sample_name, content in processed_samples.items():
        kind = content["kind"]
        variant = content["variant"]
        by_kind_variant.setdefault(kind, {})[variant] = content

    for kind, by_variant in by_kind_variant.items():
        compare_kind(kind, by_variant, cfg, output_dir)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    log(f"Finished. Summary written to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
