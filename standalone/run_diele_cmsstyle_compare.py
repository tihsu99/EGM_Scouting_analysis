#!/usr/bin/env python3
"""Dielectron baseline-vs-custom comparison with CMS-style plots (no Tag-and-Probe)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import awkward as ak
import numpy as np
import ROOT
import cmsstyle as CMS
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
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

ELECTRON_MASS_GEV = 0.00051099895
WP_ORDER = ["Veto", "Loose", "Medium", "Tight"]
WP_LABELS = ["no-id"] + WP_ORDER + [f"{wp}-noiso" for wp in WP_ORDER]
STAGE1_DIRNAME = "stage1_arrays"
DEFAULT_COMPARISON_TRIGGERS = {
    "doubleEG": "PFScouting_DoubleEG",
    "singlePhoton": "PFScouting_SinglePhotonEB",
}

DIELECTRON_VARIABLE_SPECS = {
    "mass_full": {"source": "mass", "bins": (160, 0.2, 160.0), "xlabel": "m_{ee} [GeV]"},
    "mass_jpsi": {"source": "mass", "bins": (80, 2.6, 3.4), "xlabel": "m_{ee} [GeV]"},
    "mass_z": {"source": "mass", "bins": (80, 70.0, 110.0), "xlabel": "m_{ee} [GeV]"},
    "lead_bestTrack_pt": {"source": "lead_bestTrack_pt", "bins": (100, 0.0, 200.0), "xlabel": "Leading bestTrack p_{T} [GeV]"},
    "sublead_bestTrack_pt": {
        "source": "sublead_bestTrack_pt",
        "bins": (100, 0.0, 200.0),
        "xlabel": "Subleading bestTrack p_{T} [GeV]",
    },
    "lead_bestTrack_etaMode": {
        "source": "lead_bestTrack_etaMode",
        "bins": (100, -2.5, 2.5),
        "xlabel": "Leading bestTrack #eta_{mode}",
    },
    "sublead_bestTrack_etaMode": {
        "source": "sublead_bestTrack_etaMode",
        "bins": (100, -2.5, 2.5),
        "xlabel": "Subleading bestTrack #eta_{mode}",
    },
    "lead_bestTrack_phiMode": {
        "source": "lead_bestTrack_phiMode",
        "bins": (128, -3.2, 3.2),
        "xlabel": "Leading bestTrack #phi_{mode}",
    },
    "sublead_bestTrack_phiMode": {
        "source": "sublead_bestTrack_phiMode",
        "bins": (128, -3.2, 3.2),
        "xlabel": "Subleading bestTrack #phi_{mode}",
    },
    "lead_bestTrack_d0": {"source": "lead_bestTrack_d0", "bins": (200, -0.2, 0.2), "xlabel": "Leading bestTrack d_{0} [cm]"},
    "sublead_bestTrack_d0": {
        "source": "sublead_bestTrack_d0",
        "bins": (200, -0.2, 0.2),
        "xlabel": "Subleading bestTrack d_{0} [cm]",
    },
    "lead_bestTrack_dz": {"source": "lead_bestTrack_dz", "bins": (120, -0.30, 0.30), "xlabel": "Leading bestTrack d_{z} [cm]"},
    "sublead_bestTrack_dz": {
        "source": "sublead_bestTrack_dz",
        "bins": (120, -0.30, 0.30),
        "xlabel": "Subleading bestTrack d_{z} [cm]",
    },
    "lead_bestTrack_chi2overndf": {
        "source": "lead_bestTrack_chi2overndf",
        "bins": (100, 0.0, 20.0),
        "xlabel": "Leading bestTrack #chi^{2}/ndf",
    },
    "sublead_bestTrack_chi2overndf": {
        "source": "sublead_bestTrack_chi2overndf",
        "bins": (100, 0.0, 20.0),
        "xlabel": "Subleading bestTrack #chi^{2}/ndf",
    },
}

SINGLE_ELECTRON_VARIABLE_SPECS = {
    "single_bestTrack_pt": {"source": "single_bestTrack_pt", "bins": (100, 0.0, 200.0), "xlabel": "Electron bestTrack p_{T} [GeV]"},
    "single_bestTrack_etaMode": {
        "source": "single_bestTrack_etaMode",
        "bins": (100, -2.5, 2.5),
        "xlabel": "Electron bestTrack #eta_{mode}",
    },
    "single_bestTrack_phiMode": {
        "source": "single_bestTrack_phiMode",
        "bins": (128, -3.2, 3.2),
        "xlabel": "Electron bestTrack #phi_{mode}",
    },
    "single_bestTrack_d0": {"source": "single_bestTrack_d0", "bins": (200, -0.2, 0.2), "xlabel": "Electron bestTrack d_{0} [cm]"},
    "single_bestTrack_dz": {"source": "single_bestTrack_dz", "bins": (120, -0.30, 0.30), "xlabel": "Electron bestTrack d_{z} [cm]"},
    "single_bestTrack_chi2overndf": {
        "source": "single_bestTrack_chi2overndf",
        "bins": (100, 0.0, 20.0),
        "xlabel": "Electron bestTrack #chi^{2}/ndf",
    },
    "single_bestTrack_charge": {"source": "single_bestTrack_charge", "bins": (3, -1.5, 1.5), "xlabel": "Electron charge"},
}

ALL_SOURCE_NAMES = sorted(
    {spec["source"] for spec in DIELECTRON_VARIABLE_SPECS.values()}
    | {spec["source"] for spec in SINGLE_ELECTRON_VARIABLE_SPECS.values()}
)

COLORS = {
    "baseline": ROOT.TColor.GetColor("#1f77b4"),
    "custom": ROOT.TColor.GetColor("#ff7f0e"),
}


def log(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def col_accumulator(dtype: np.dtype) -> processor.column_accumulator:
    return processor.column_accumulator(np.array([], dtype=dtype))


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
    proc = subprocess.run(["xrdfs", host, "ls", directory], check=True, capture_output=True, text=True)
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
    query = f"file dataset={dataset} instance={das_instance}"
    proc = subprocess.run(["dasgoclient", "-query", query], check=True, capture_output=True, text=True)
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
            files.extend(query_xrootd_files(prepend_redirector(dataset, redirector)))
        else:
            if skip_das:
                raise RuntimeError(f"DAS lookup disabled but sample has only dataset entry: {dataset}")
            das_files = query_das_files(dataset, das_instance)
            files.extend(prepend_redirector(f, redirector) for f in das_files)

    files = list(dict.fromkeys(files))
    if max_files is not None:
        files = files[:max_files]
    log(f"Resolved {len(files)} input files")
    return files


class DielectronProcessor(processor.ProcessorABC):
    def __init__(self, trigger_field: str):
        self.trigger_field = trigger_field
        self.var_names = ALL_SOURCE_NAMES
        self._accumulator = processor.dict_accumulator(
            {
                wp: processor.dict_accumulator({name: col_accumulator(np.float32) for name in self.var_names})
                for wp in WP_LABELS
            }
        )

    @property
    def accumulator(self):
        return self._accumulator

    @staticmethod
    def _add_column(out: Dict, key: str, values: ak.Array | np.ndarray) -> None:
        arr = np.asarray(ak.to_numpy(values), dtype=np.float32)
        if arr.size == 0:
            return
        out[key] += processor.column_accumulator(arr)

    @staticmethod
    def _field_or(obj: ak.Array, primary: str, fallback: str) -> ak.Array:
        if primary in obj.fields:
            return obj[primary]
        return obj[fallback]

    @staticmethod
    def _ones_like(events: ak.Array) -> ak.Array:
        if len(events) == 0:
            return ak.Array(np.array([], dtype=bool))
        if "run" in events.fields:
            return ak.ones_like(events.run, dtype=bool)
        return ak.Array(np.ones(len(events), dtype=bool))

    @staticmethod
    def _zero_like(events: ak.Array) -> ak.Array:
        if len(events) == 0:
            return ak.Array(np.array([], dtype=np.float32))
        if "run" in events.fields:
            return ak.zeros_like(events.run, dtype=np.float32)
        return ak.Array(np.zeros(len(events), dtype=np.float32))

    def process(self, events):
        out = self.accumulator.identity()
        if "ScoutingElectron" not in events.fields:
            return out

        electrons = events.ScoutingElectron

        if "DST" in events.fields and self.trigger_field in events.DST.fields:
            trigger_mask = events.DST[self.trigger_field]
        else:
            trigger_mask = self._ones_like(events)
            trigger_mask = trigger_mask & False

        events = events[trigger_mask]
        electrons = electrons[trigger_mask]
        if len(events) == 0:
            return out

        rho = events.ScoutingRho.fixedGridRhoFastjetAll if "ScoutingRho" in events.fields else self._zero_like(events)
        for wp in WP_ORDER:
            electrons[f"{wp}_id"] = electron_id_mask(electrons, rho, wp, use_iso=True)
            electrons[f"{wp}-noiso_id"] = electron_id_mask(electrons, rho, wp, use_iso=False)

        eta_mode = self._field_or(electrons, "bestTrack_etaMode", "bestTrack_eta")
        phi_mode = self._field_or(electrons, "bestTrack_phiMode", "bestTrack_phi")
        electrons["p4"] = ak.zip(
            {
                "pt": electrons.pt,
                "eta": eta_mode,
                "phi": phi_mode,
                "mass": ak.ones_like(electrons.pt) * ELECTRON_MASS_GEV,
            },
            with_name="Momentum4D",
        )

        wp_ele_masks = {"no-id": ak.ones_like(electrons.bestTrack_pt, dtype=bool)}
        for wp in WP_ORDER:
            wp_ele_masks[wp] = electrons[f"{wp}_id"]
            wp_ele_masks[f"{wp}-noiso"] = electrons[f"{wp}-noiso_id"]

        for wp, ele_mask in wp_ele_masks.items():
            electrons_wp = electrons[ele_mask]

            n1_mask = ak.num(electrons_wp) == 1
            electrons_single = electrons_wp[n1_mask]
            if len(electrons_single) > 0:
                electrons_single = electrons_single[ak.argsort(electrons_single.bestTrack_pt, axis=-1, ascending=False)]
                single = electrons_single[:, 0]
                single_var_map = {
                    "single_bestTrack_pt": single.bestTrack_pt,
                    "single_bestTrack_etaMode": single.bestTrack_etaMode,
                    "single_bestTrack_phiMode": single.bestTrack_phiMode,
                    "single_bestTrack_d0": single.bestTrack_d0,
                    "single_bestTrack_dz": single.bestTrack_dz,
                    "single_bestTrack_chi2overndf": single.bestTrack_chi2overndf,
                    "single_bestTrack_charge": single.bestTrack_charge,
                }
                for var_name, values in single_var_map.items():
                    self._add_column(out[wp], var_name, values)

            n2_mask = ak.num(electrons_wp) == 2
            electrons_wp = electrons_wp[n2_mask]
            if len(electrons_wp) == 0:
                continue

            electrons_wp = electrons_wp[ak.argsort(electrons_wp.bestTrack_pt, axis=-1, ascending=False)]
            lead = electrons_wp[:, 0]
            sublead = electrons_wp[:, 1]
            os_mask = (lead.bestTrack_charge * sublead.bestTrack_charge) < 0
            electrons_wp = electrons_wp[os_mask]
            if len(electrons_wp) == 0:
                continue

            lead = electrons_wp[:, 0]
            sublead = electrons_wp[:, 1]
            mass = (lead.p4 + sublead.p4).mass

            var_map = {
                "mass": mass,
                "lead_bestTrack_pt": lead.bestTrack_pt,
                "sublead_bestTrack_pt": sublead.bestTrack_pt,
                "lead_bestTrack_etaMode": lead.bestTrack_etaMode,
                "sublead_bestTrack_etaMode": sublead.bestTrack_etaMode,
                "lead_bestTrack_phiMode": lead.bestTrack_phiMode,
                "sublead_bestTrack_phiMode": sublead.bestTrack_phiMode,
                "lead_bestTrack_d0": lead.bestTrack_d0,
                "sublead_bestTrack_d0": sublead.bestTrack_d0,
                "lead_bestTrack_dz": lead.bestTrack_dz,
                "sublead_bestTrack_dz": sublead.bestTrack_dz,
                "lead_bestTrack_chi2overndf": lead.bestTrack_chi2overndf,
                "sublead_bestTrack_chi2overndf": sublead.bestTrack_chi2overndf,
            }
            for var_name, values in var_map.items():
                self._add_column(out[wp], var_name, values)

        return out

    def postprocess(self, accumulator):
        return accumulator


def build_runner(cfg: Dict) -> Runner:
    runner_cfg = cfg.get("runner", {})
    workers = int(runner_cfg.get("workers", max(1, (os.cpu_count() or 2) - 1)))
    chunksize = int(runner_cfg.get("chunksize", 300000))
    maxchunks = runner_cfg.get("maxchunks")
    show_progress = bool(runner_cfg.get("show_progress", True))
    log(f"Create runner: workers={workers}, chunksize={chunksize}, maxchunks={maxchunks}, progress={show_progress}")
    return Runner(
        executor=FuturesExecutor(compression=None, workers=workers, status=show_progress),
        schema=NanoAODSchema,
        chunksize=chunksize,
        maxchunks=maxchunks,
    )


def run_sample(files: List[str], cfg: Dict, sample_name: str, trigger_field: str) -> Dict[str, Dict[str, np.ndarray]]:
    runner = build_runner(cfg)
    processor_instance = DielectronProcessor(trigger_field=trigger_field)
    log(f"[{sample_name}] Start processing {len(files)} files with trigger {trigger_field}")
    output = runner({"sample": files}, treename=cfg.get("tree_name", "Events"), processor_instance=processor_instance)
    result: Dict[str, Dict[str, np.ndarray]] = {}
    for wp in WP_LABELS:
        result[wp] = {var: output[wp][var].value for var in output[wp].keys()}
    log(f"[{sample_name}] Done")
    return result


def _flatten_key(wp: str, var: str) -> str:
    return f"{wp}::{var}"


def _unflatten_key(key: str) -> Tuple[str, str]:
    wp, var = key.split("::", 1)
    return wp, var


def save_stage1_arrays(
    stage1_dir: Path, sample_name: str, trigger_tag: str, arrays: Dict[str, Dict[str, np.ndarray]]
) -> Path:
    stage1_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, np.ndarray] = {}
    for wp in WP_LABELS:
        for var in arrays.get(wp, {}):
            payload[_flatten_key(wp, var)] = arrays[wp][var]
    outpath = stage1_dir / f"{sample_name}__{trigger_tag}.npz"
    np.savez_compressed(outpath, **payload)
    log(f"[{sample_name}/{trigger_tag}] Saved stage-1 arrays: {outpath}")
    return outpath


def load_stage1_arrays(stage1_dir: Path, sample_name: str, trigger_tag: str) -> Dict[str, Dict[str, np.ndarray]]:
    inpath = stage1_dir / f"{sample_name}__{trigger_tag}.npz"
    if not inpath.exists():
        raise FileNotFoundError(f"Missing stage-1 cache: {inpath}")

    arrays: Dict[str, Dict[str, np.ndarray]] = {wp: {} for wp in WP_LABELS}
    with np.load(inpath, allow_pickle=False) as data:
        for key in data.files:
            wp, var = _unflatten_key(key)
            arrays.setdefault(wp, {})[var] = np.asarray(data[key], dtype=np.float32)

    for wp in WP_LABELS:
        for source in ALL_SOURCE_NAMES:
            arrays[wp].setdefault(source, np.array([], dtype=np.float32))

    log(f"[{sample_name}/{trigger_tag}] Loaded stage-1 arrays: {inpath}")
    return arrays


def build_hist(name: str, values: np.ndarray, bins: Tuple[int, float, float]) -> ROOT.TH1F:
    nbins, xmin, xmax = bins
    hist = ROOT.TH1F(name, "", nbins, xmin, xmax)
    hist.Sumw2()
    for value in values:
        hist.Fill(float(value))
    return hist


def ratio_hist(numerator: ROOT.TH1F, denominator: ROOT.TH1F, name: str) -> ROOT.TH1F:
    ratio = numerator.Clone(name)
    ratio.Divide(denominator)
    # for ibin in range(1, numerator.GetNbinsX() + 1):
    #     n = numerator.GetBinContent(ibin)
    #     d = denominator.GetBinContent(ibin)
    #     en = numerator.GetBinError(ibin)
    #     ed = denominator.GetBinError(ibin)
    #     if d <= 0:
    #         ratio.SetBinContent(ibin, 0.0)
    #         ratio.SetBinError(ibin, 0.0)
    #         continue
    #     r = n / d
    #     rel_n = (en / n) if n > 0 else 0.0
    #     rel_d = ed / d
    #     er = r * np.sqrt(rel_n * rel_n + rel_d * rel_d)
    #     ratio.SetBinContent(ibin, r)
    #     ratio.SetBinError(ibin, er)
    return ratio


def draw_comparison(
    baseline: np.ndarray,
    custom: np.ndarray,
    bins: Tuple[int, float, float],
    xlabel: str,
    title_text: str,
    selection_text: str,
    outpath: Path,
    ratio_min: float,
    ratio_max: float,
) -> None:
    h_base = build_hist(f"h_base_{outpath.stem}", baseline, bins)
    h_cust = build_hist(f"h_cust_{outpath.stem}", custom, bins)
    h_ratio = ratio_hist(h_cust, h_base, f"h_ratio_{outpath.stem}")
    h_base.SetStats(0)
    h_cust.SetStats(0)
    h_ratio.SetStats(0)

    h_base.SetLineColor(COLORS["baseline"])
    h_base.SetMarkerColor(COLORS["baseline"])
    h_base.SetMarkerStyle(20)
    h_base.SetMarkerSize(0.7)
    h_base.SetLineWidth(2)

    h_cust.SetLineColor(COLORS["custom"])
    h_cust.SetMarkerColor(COLORS["custom"])
    h_cust.SetMarkerStyle(24)
    h_cust.SetMarkerSize(0.7)
    h_cust.SetLineWidth(2)

    y_max = max(h_base.GetMaximum(), h_cust.GetMaximum())
    if y_max <= 0:
        y_max = 1.0
    h_base.SetMaximum(y_max * 1.35)
    h_base.SetMinimum(0.0)

    c = ROOT.TCanvas(f"c_{outpath.stem}", "", 800, 800)
    pad1 = ROOT.TPad("pad1", "", 0.0, 0.30, 1.0, 1.0)
    pad2 = ROOT.TPad("pad2", "", 0.0, 0.0, 1.0, 0.30)
    pad1.SetBottomMargin(0.02)
    pad2.SetTopMargin(0.04)
    pad2.SetBottomMargin(0.32)
    pad1.SetLeftMargin(0.13)
    pad2.SetLeftMargin(0.13)
    pad1.SetRightMargin(0.04)
    pad2.SetRightMargin(0.04)
    pad1.Draw()
    pad2.Draw()

    pad1.cd()
    h_base.SetTitle("")
    h_base.GetYaxis().SetTitle("Counts")
    h_base.GetYaxis().SetTitleSize(0.05)
    h_base.GetYaxis().SetTitleOffset(1.2)
    h_base.GetYaxis().SetLabelSize(0.045)
    h_base.GetXaxis().SetLabelSize(0.0)
    h_base.Draw("E1")
    h_cust.Draw("E1 SAME")

    legend = CMS.cmsLeg(0.14, 0.78, 0.46, 0.90, textSize=0.038)
    legend.AddEntry(h_base, "Baseline", "lep")
    legend.AddEntry(h_cust, "Customise1", "lep")
    legend.Draw()

    cms_label = ROOT.TLatex()
    cms_label.SetNDC(True)
    cms_label.SetTextFont(62)
    cms_label.SetTextSize(0.055)
    cms_label.DrawLatex(0.14, 0.92, "CMS")
    cms_label.SetTextFont(52)
    cms_label.SetTextSize(0.045)
    cms_label.DrawLatex(0.24, 0.92, "Preliminary")
    cms_label.SetTextFont(42)
    cms_label.SetTextSize(0.04)
    cms_label.DrawLatex(0.83, 0.92, "(13.6 TeV)")

    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextFont(42)
    text.SetTextSize(0.032)
    text.DrawLatex(0.50, 0.84, title_text)
    text.DrawLatex(0.50, 0.79, selection_text)

    pad2.cd()
    h_ratio.SetTitle("")
    h_ratio.GetYaxis().SetTitle("Ratio to Baseline")
    h_ratio.GetYaxis().SetNdivisions(505)
    h_ratio.GetYaxis().SetTitleSize(0.10)
    h_ratio.GetYaxis().SetTitleOffset(0.55)
    h_ratio.GetYaxis().SetLabelSize(0.09)
    h_ratio.GetXaxis().SetTitle(xlabel)
    h_ratio.GetXaxis().SetTitleSize(0.11)
    h_ratio.GetXaxis().SetTitleOffset(1.1)
    h_ratio.GetXaxis().SetLabelSize(0.10)
    h_ratio.SetMinimum(ratio_min)
    h_ratio.SetMaximum(ratio_max)
    h_ratio.SetMarkerStyle(20)
    h_ratio.SetMarkerSize(0.6)
    h_ratio.SetLineColor(COLORS["custom"])
    h_ratio.SetMarkerColor(COLORS["custom"])
    h_ratio.Draw("E1")

    line = ROOT.TLine(h_ratio.GetXaxis().GetXmin(), 1.0, h_ratio.GetXaxis().GetXmax(), 1.0)
    line.SetLineStyle(2)
    line.SetLineColor(ROOT.kGray + 2)
    line.Draw("SAME")

    c.SaveAs(str(outpath))
    c.Close()


def run_comparisons(
    baseline_arrays: Dict[str, Dict[str, np.ndarray]],
    custom_arrays: Dict[str, Dict[str, np.ndarray]],
    output_dir: Path,
    variable_specs: Dict[str, Dict[str, object]],
    selection_text: str,
    title_suffix: str,
    ratio_min: float,
    ratio_max: float,
) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for wp in WP_LABELS:
        wp_dir = output_dir / wp.replace("-", "_")
        wp_dir.mkdir(parents=True, exist_ok=True)
        summary[wp] = {}
        for plot_name, spec in variable_specs.items():
            source = spec["source"]
            baseline = baseline_arrays[wp].get(source, np.array([], dtype=np.float32))
            custom = custom_arrays[wp].get(source, np.array([], dtype=np.float32))
            summary[wp][plot_name] = int(min(len(baseline), len(custom)))
            draw_comparison(
                baseline=baseline,
                custom=custom,
                bins=spec["bins"],
                xlabel=spec["xlabel"],
                title_text=f"{wp} {title_suffix}",
                selection_text=selection_text,
                outpath=wp_dir / f"{plot_name}.png",
                ratio_min=ratio_min,
                ratio_max=ratio_max,
            )
        log(f"[plots] Finished WP {wp}")
    return summary


def find_pairs_by_kind(samples_cfg: Dict[str, Dict]) -> Dict[str, Tuple[Tuple[str, Dict], Tuple[str, Dict]]]:
    grouped: Dict[str, Dict[str, Tuple[str, Dict]]] = {}
    for name, cfg in samples_cfg.items():
        kind = str(cfg.get("kind", "unknown")).lower()
        variant = str(cfg.get("variant", "")).lower()
        if variant not in ("baseline", "custom"):
            continue
        grouped.setdefault(kind, {})[variant] = (name, cfg)

    pairs: Dict[str, Tuple[Tuple[str, Dict], Tuple[str, Dict]]] = {}
    for kind, items in grouped.items():
        if "baseline" in items and "custom" in items:
            pairs[kind] = (items["baseline"], items["custom"])
        else:
            log(f"[kind={kind}] Skip: need both baseline and custom")

    if not pairs:
        raise RuntimeError("No complete baseline/custom pairs found by kind in config.")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="CMS-style dielectron comparison (no Tag-and-Probe)")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--plot-only", action="store_true", help="Skip Coffea processing and replot from cached stage-1 arrays")
    parser.add_argument(
        "--stage1-dir",
        default=None,
        help=f"Directory for stage-1 cache (.npz). Default: <output>/{STAGE1_DIRNAME}",
    )
    parser.add_argument("--skip-das", action="store_true", help="Disable DAS lookups")
    parser.add_argument("--max-files", type=int, default=None, help="Optional max files per sample")
    parser.add_argument("--ratio-min", type=float, default=0.9, help="Lower y-limit for ratio panel")
    parser.add_argument("--ratio-max", type=float, default=1.1, help="Upper y-limit for ratio panel")
    args = parser.parse_args()
    if args.ratio_min >= args.ratio_max:
        raise ValueError("--ratio-min must be smaller than --ratio-max")

    CMS.SetExtraText("Preliminary")
    CMS.SetEnergy("13.6")
    CMS.SetLumi("")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_dir = Path(args.stage1_dir) if args.stage1_dir else (output_dir / STAGE1_DIRNAME)

    redirector = cfg.get("redirector", "root://xrootd-cms.infn.it//")
    das_instance = cfg.get("das_instance", "prod/phys03")
    max_files = args.max_files if args.max_files is not None else cfg.get("max_files_per_sample")
    comparison_triggers = cfg.get("comparison_triggers", DEFAULT_COMPARISON_TRIGGERS)
    samples_cfg = cfg.get("samples", {})
    pairs_by_kind = find_pairs_by_kind(samples_cfg)

    summary = {}
    summary["kinds"] = {}

    for kind, (baseline_entry, custom_entry) in pairs_by_kind.items():
        log(f"[kind={kind}] Start")
        kind_summary = {"triggers": {}}
        baseline_name, _ = baseline_entry
        custom_name, _ = custom_entry

        for trigger_tag, trigger_field in comparison_triggers.items():
            log(f"[kind={kind}] [trigger={trigger_tag}] Start")
            sample_results = {}
            trig_sum = {}
            if args.plot_only:
                log(f"[kind={kind}] [trigger={trigger_tag}] Plot-only mode: load cached arrays")
                for sample_name, _sample_cfg in (baseline_entry, custom_entry):
                    sample_results[sample_name] = load_stage1_arrays(stage1_dir, sample_name, trigger_tag)
                    trig_sum[sample_name] = {
                        "stage1_cache": str(stage1_dir / f"{sample_name}__{trigger_tag}.npz"),
                        "mode": "plot-only",
                    }
            else:
                for sample_name, sample_cfg in (baseline_entry, custom_entry):
                    log(f"[{sample_name}] Resolve input files")
                    files = resolve_sample_files(sample_cfg, redirector, max_files, args.skip_das, das_instance)
                    if not files:
                        raise RuntimeError(f"No files resolved for sample {sample_name}")
                    sample_results[sample_name] = run_sample(files, cfg, sample_name, trigger_field=trigger_field)
                    save_stage1_arrays(stage1_dir, sample_name, trigger_tag, sample_results[sample_name])
                    trig_sum[sample_name] = {
                        "n_files": len(files),
                        "stage1_cache": str(stage1_dir / f"{sample_name}__{trigger_tag}.npz"),
                        "mode": "process+plot",
                    }

            selection_text = f"Pass DST.{trigger_field}, n_{{e}}(after WP)=2, OS"
            log(f"[kind={kind}] [trigger={trigger_tag}] Start CMS-style comparison plotting")
            dielectron_summary = run_comparisons(
                sample_results[baseline_name],
                sample_results[custom_name],
                output_dir / "plots" / kind / trigger_tag / "dielectron",
                variable_specs=DIELECTRON_VARIABLE_SPECS,
                selection_text=selection_text,
                title_suffix="dielectron",
                ratio_min=args.ratio_min,
                ratio_max=args.ratio_max,
            )
            single_selection_text = f"Pass DST.{trigger_field}, n_{{e}}(after WP)=1"
            single_electron_summary = run_comparisons(
                sample_results[baseline_name],
                sample_results[custom_name],
                output_dir / "plots" / kind / trigger_tag / "single_electron",
                variable_specs=SINGLE_ELECTRON_VARIABLE_SPECS,
                selection_text=single_selection_text,
                title_suffix="single electron",
                ratio_min=args.ratio_min,
                ratio_max=args.ratio_max,
            )
            trig_sum["plot_summary"] = {
                "dielectron": dielectron_summary,
                "single_electron": single_electron_summary,
            }
            kind_summary["triggers"][trigger_tag] = trig_sum
        summary["kinds"][kind] = kind_summary

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    log(f"Done. Summary written to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
