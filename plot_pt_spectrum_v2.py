#!/usr/bin/env python3
import os
import sys
import glob
import pickle
import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mplhep as hep
import cmsstyle as CMS

from coffea.processor.accumulator import AccumulatorABC

class HistAcc(AccumulatorABC):
    def __init__(self, h):
        self.h = h

    def identity(self):
        # Make an empty histogram with same axes/storage
        h0 = self.h.copy()
        try:
            h0.reset()
        except Exception:
            # Fallback: rely on arithmetic if reset() is unavailable
            h0 = self.h * 0
        return HistAcc(h0)

    def __add__(self, other):
        return HistAcc(self.h + other.h)

    def __iadd__(self, other):
        self.h += other.h
        return self

    def fill(self, **kwargs):
        self.h.fill(**kwargs)

def unwrap_hist(x):
    # For HistAcc wrapper: access .h
    return x.h if hasattr(x, "h") else x


def find_pkl_files(store_paths, pattern="*.pkl", recursive=False):
    if isinstance(store_paths, str):
        store_paths = [store_paths]

    files = []
    for d in store_paths:
        if os.path.isdir(d):
            if recursive:
                files.extend(glob.glob(os.path.join(d, "**", pattern), recursive=True))
            else:
                files.extend(glob.glob(os.path.join(d, pattern), recursive=False))
        else:
            files.extend(glob.glob(d, recursive=recursive))
    files = sorted(list(set(files)))
    return files


def _unwrap_dataset_obj(obj, dataset=None):
    # Runner output often: {"DatasetName": <dict-like with hists>}
    if isinstance(obj, dict):
        if dataset is not None and dataset in obj:
            v = obj[dataset]
            return v
        if len(obj) == 1:
            v = next(iter(obj.values()))
            if hasattr(v, "keys"):
                return v
        # If multiple datasets and user didn't specify, merge them later in load_and_merge
        return obj
    return obj


def _merge_dict_like(dst, src, keys):
    for k in keys:
        if k in src and k in dst:
            dst[k] = dst[k] + src[k]
        elif k in src and k not in dst:
            dst[k] = src[k]


def load_and_merge(pkls, dataset=None):
    # Keys produced by your ntupler for lead/sublead cutbased
    keys = [
        "h_lead_pt_abseta_cutbased",
        "h_lead_pt_abseta_cutbased_noiso",
        "h_sublead_pt_abseta_cutbased",
        "h_sublead_pt_abseta_cutbased_noiso",
    ]

    merged = None

    for p in pkls:
        with open(p, "rb") as f:
            obj_raw = pickle.load(f)

        obj = _unwrap_dataset_obj(obj_raw, dataset=dataset)

        # If obj is still a dict of datasets, combine them into one dict-like for this file
        if isinstance(obj, dict) and any(isinstance(v, dict) or hasattr(v, "keys") for v in obj.values()):
            combined = {}
            for _, v in obj.items():
                if hasattr(v, "keys"):
                    _merge_dict_like(combined, v, keys)
            obj = combined

        if merged is None:
            merged = obj
            continue

        _merge_dict_like(merged, obj, keys)

    return merged


def bin_range_from_value_edges(edges, low, high):
    edges = np.asarray(edges, dtype=float)
    nbins = len(edges) - 1

    i0 = int(np.searchsorted(edges, low, side="right") - 1)
    i1 = int(np.searchsorted(edges, high, side="left"))

    if i0 < 0:
        i0 = 0
    if i1 < 0:
        i1 = 0
    if i0 > nbins:
        i0 = nbins
    if i1 > nbins:
        i1 = nbins
    if i1 < i0:
        i1 = i0

    return i0, i1


def rebin_1d(edges, values, variances, factor):
    if factor is None or factor <= 1:
        return edges, values, variances

    n = len(values)
    m = (n // factor) * factor
    values = values[:m]
    if variances is not None:
        variances = variances[:m]

    values_rb = values.reshape(-1, factor).sum(axis=1)
    if variances is None:
        variances_rb = None
    else:
        variances_rb = variances.reshape(-1, factor).sum(axis=1)

    edges_rb = edges[::factor]
    # Ensure last edge included
    if len(edges_rb) != len(values_rb) + 1:
        edges_rb = np.append(edges_rb, edges[-1])

    return edges_rb, values_rb, variances_rb


def project_cutbased_threshold(h3, abseta_min, abseta_max, thr, inclusive=True):
    """
    h3 axes order (producer):
      pt, abseta, (cutbased_id or cutbased_noiso_id)
    """
    pt_edges = np.asarray(h3.axes["pt"].edges, dtype=float)
    ab_edges = np.asarray(h3.axes["abseta"].edges, dtype=float)

    a0, a1 = bin_range_from_value_edges(ab_edges, abseta_min, abseta_max)

    j0 = int(thr)
    j1 = 5 if inclusive else (j0 + 1)

    vals3 = h3.view(flow=False)  # (npt, nab, nid)
    spec = vals3[:, a0:a1, j0:j1].sum(axis=(1, 2))

    spec_var = None
    try:
        vars3 = h3.variances(flow=False)
        if vars3 is not None:
            spec_var = np.asarray(vars3, dtype=float)[:, a0:a1, j0:j1].sum(axis=(1, 2))
    except Exception:
        spec_var = None

    return pt_edges, np.asarray(spec, dtype=float), spec_var


def _cms_label(fig, ax):
    # Keep CMS package usage; fall back gracefully
    hep.style.use("CMS")
    hep.cms.label(ax=ax, data=True, label="Preliminary", lumi=None, rlabel="(13.6 TeV)")
def draw_pt_spectrum_cutbased_iso_noiso(
    h_iso, h_noiso,
    region_name, abseta_min, abseta_max,
    plot_path,
    tag="lead",
    inclusive=True,
    rebin=None,
    logx=False,
    logy=False,
    xmin=None,
    norm_bin_width=True,
    out_ext=("png",),
):


    h_iso = unwrap_hist(h_iso)
    h_noiso = unwrap_hist(h_noiso)

    wp_defs = [("No ID", 0), ("Veto", 1), ("Loose", 2), ("Medium", 3), ("Tight", 4)]

    hep.style.use("CMS")

    fig = plt.figure()
    ax = plt.gca()

    for wp_name, thr in wp_defs:
        pt_edges, y_iso, yv_iso = project_cutbased_threshold(
            h_iso, abseta_min, abseta_max, thr, inclusive=inclusive
        )
        _, y_no, yv_no = project_cutbased_threshold(
            h_noiso, abseta_min, abseta_max, thr, inclusive=inclusive
        )

        if norm_bin_width:
            bw = np.diff(pt_edges)
            y_iso = y_iso / bw
            y_no = y_no / bw
            if yv_iso is not None:
                yv_iso = yv_iso / (bw * bw)
            if yv_no is not None:
                yv_no = yv_no / (bw * bw)

        pt_edges_rb, y_iso_rb, yv_iso_rb = rebin_1d(pt_edges, y_iso, yv_iso, rebin)
        _, y_no_rb, yv_no_rb = rebin_1d(pt_edges, y_no, yv_no, rebin)

        line_iso = ax.step(
            pt_edges_rb[:-1], y_iso_rb, where="post",
            label=f"{wp_name} (iso)" if thr > 0 else f"{wp_name}"
        )[0]
        c = line_iso.get_color()
 
        if thr > 0:       
          ax.step(
            pt_edges_rb[:-1], y_no_rb, where="post",
            linestyle="--", color=c,
            label=f"{wp_name} (noiso)"
          )

        # Optional error bands (only if variances exist)
        if yv_iso_rb is not None:
            err = np.sqrt(np.maximum(yv_iso_rb, 0.0))
            x = pt_edges_rb[:-1]
            ax.fill_between(x, y_iso_rb - err, y_iso_rb + err, step="post", alpha=0.15, color=c)
        if yv_no_rb is not None:
            err = np.sqrt(np.maximum(yv_no_rb, 0.0))
            x = pt_edges_rb[:-1]
            ax.fill_between(x, y_no_rb - err, y_no_rb + err, step="post", alpha=0.10, color=c)

    if xmin is not None:
      ax.set_xlim(left=xmin)

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(bottom=ymin, top=ymax * 3)

    if tag == "lead" :
        ax.set_xlabel("leading electron $p_{T}$ [GeV]")
    else:
        ax.set_xlabel("subleading electron $p_{T}$ [GeV]")
    ax.set_ylabel("events/bin width [GeV]" if norm_bin_width else "N")
    
    if region_name == "EB":
        plt.text(0.06, 0.9, "Barrel", fontweight="bold", transform=ax.transAxes)
    else:
        plt.text(0.06, 0.9, "Endcap", fontweight="bold", transform=ax.transAxes)

    _cms_label(fig, ax)

    ax.legend(fontsize=12, ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    os.makedirs(plot_path, exist_ok=True)

    base = os.path.join(plot_path, f"pt_{tag}_{region_name}_iso_vs_noiso")
    for ext in out_ext:
        fig.savefig(f"{base}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store_path", nargs="+", required=True, help="Directory(ies) or glob(s) to pkl files")
    parser.add_argument("--pattern", default="*.pkl")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--dataset", default=None, help="Dataset key inside runner output dict (optional)")

    parser.add_argument("--plot_path", required=True)
    parser.add_argument("--rebin", type=int, default=None)
    parser.add_argument("--logx", action="store_true")
    parser.add_argument("--logy", action="store_true")
    parser.add_argument("--xmin", type=float, default=None)

    parser.add_argument("--exclusive", action="store_true")
    parser.add_argument("--inclusive", action="store_true")
    parser.add_argument("--no_norm_bin_width", action="store_true")

    parser.add_argument("--extra_text", default="Preliminary")
    parser.add_argument("--energy", default="13.6 TeV")
    parser.add_argument("--lumi", default="")

    parser.add_argument("--out_ext", nargs="+", default=["png"])

    args = parser.parse_args()

    os.makedirs(args.plot_path, exist_ok=True)

    # Keep CMS style configuration (no magic)
    try:
        CMS.SetExtraText(args.extra_text)
    except Exception:
        pass
    try:
        CMS.SetEnergy(args.energy)
    except Exception:
        pass
    try:
        CMS.SetLumi(args.lumi)
    except Exception:
        pass

    pkls = find_pkl_files(args.store_path, args.pattern, recursive=args.recursive)
    if len(pkls) == 0:
        print("[error] no pkl files found")
        sys.exit(1)

    print(f"Found {len(pkls)} pkl files")
    merged = load_and_merge(pkls, dataset=args.dataset)
    if merged is None:
        print("[error] failed to load pkls")
        sys.exit(1)

    # Regions
    regions = [
        ("ALL", 0.0, 2.5),
        ("EB", 0.0, 1.47),
        ("EE", 1.47, 2.5),
    ]

    # Inclusive / exclusive selection for ID threshold summation
    inclusive = True
    if args.exclusive and not args.inclusive:
        inclusive = False
    elif args.inclusive and not args.exclusive:
        inclusive = True

    norm_bin_width = not args.no_norm_bin_width

    # Grab lead/sublead iso/noiso hists
    h_lead_iso = merged.get("h_lead_pt_abseta_cutbased", None)
    h_lead_no = merged.get("h_lead_pt_abseta_cutbased_noiso", None)
    h_sub_iso = merged.get("h_sublead_pt_abseta_cutbased", None)
    h_sub_no = merged.get("h_sublead_pt_abseta_cutbased_noiso", None)

    missing = []
    for k, v in [
        ("h_lead_pt_abseta_cutbased", h_lead_iso),
        ("h_lead_pt_abseta_cutbased_noiso", h_lead_no),
        ("h_sublead_pt_abseta_cutbased", h_sub_iso),
        ("h_sublead_pt_abseta_cutbased_noiso", h_sub_no),
    ]:
        if v is None:
            missing.append(k)

    if missing:
        print("[error] missing required hist keys in merged output")
        print("Missing:", missing)
        print("Keys:", list(merged.keys()))
        sys.exit(1)

    # Draw plots: lead and sublead separated; iso/noiso overlay in each plot
    for region_name, a0, a1 in regions:
        draw_pt_spectrum_cutbased_iso_noiso(
            h_lead_iso, h_lead_no,
            region_name, a0, a1,
            args.plot_path,
            tag="lead",
            inclusive=inclusive,
            rebin=args.rebin,
            logx=args.logx,
            logy=args.logy,
            norm_bin_width=norm_bin_width,
            out_ext=tuple(args.out_ext),
            xmin=args.xmin
        )
        draw_pt_spectrum_cutbased_iso_noiso(
            h_sub_iso, h_sub_no,
            region_name, a0, a1,
            args.plot_path,
            tag="sublead",
            inclusive=inclusive,
            rebin=args.rebin,
            logx=args.logx,
            logy=args.logy,
            norm_bin_width=norm_bin_width,
            out_ext=tuple(args.out_ext),
            xmin=args.xmin
        )

    print("Done.")


if __name__ == "__main__":
    main()

