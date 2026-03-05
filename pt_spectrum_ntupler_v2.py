#!/usr/bin/env python3
import os
import pickle
import argparse

import numpy as np
import awkward as ak

from coffea import processor
from coffea.processor import Runner, FuturesExecutor
from coffea.nanoevents import NanoAODSchema

from hist import Hist

from electron_id import electron_id_mask

from coffea.processor.accumulator import AccumulatorABC




def _valid_mask(x):
    # Version-tolerant validity mask for optional data
    if hasattr(ak, "is_none"):
        return ~ak.is_none(x)
    if hasattr(ak, "is_valid"):
        return ak.is_valid(x)
    if hasattr(ak, "is_missing"):
        return ~ak.is_missing(x)
    raise RuntimeError("No validity predicate found in awkward")


class ElectronPtSpectrumProcessor(processor.ProcessorABC):
    def __init__(
        self,
        electron_pt_min=5.0,
        abseta_max=2.5,
        pt_bins=2000,
        pt_max=200.0,
        abseta_bins=50,
    ):
        self.electron_pt_min = float(electron_pt_min)
        self.abseta_max = float(abseta_max)
        self.pt_bins = int(pt_bins)
        self.pt_max = float(pt_max)
        self.abseta_bins = int(abseta_bins)

        self.WP = [
            "Veto", "Loose", "Medium", "Tight",
            "Veto-noiso", "Loose-noiso", "Medium-noiso", "Tight-noiso",
        ]
        self.RANK = ["lead", "sublead"]

        self._accumulator = processor.dict_accumulator({})

    @property
    def accumulator(self):
        return self._accumulator

    def _hist_cutbased(self, id_name):
        return Hist.new.Reg(
            self.pt_bins, 0.0, self.pt_max, name="pt", label="Electron pT [GeV]"
        ).Reg(
            self.abseta_bins, 0.0, self.abseta_max, name="abseta", label="|eta|"
        ).Reg(
            5, -0.5, 4.5, name=id_name, label=f"{id_name} (0-4)"
        ).Double()

    def _hist_wp(self):
        return Hist.new.StrCat(
            self.WP, name="wp", label="WP"
        ).Reg(
            self.pt_bins, 0.0, self.pt_max, name="pt", label="Electron pT [GeV]"
        ).Reg(
            self.abseta_bins, 0.0, self.abseta_max, name="abseta", label="|eta|"
        ).Double()

    def _hist_wp_rank(self):
        return Hist.new.StrCat(
            self.WP, name="wp", label="WP"
        ).StrCat(
            self.RANK, name="rank", label="rank"
        ).Reg(
            self.pt_bins, 0.0, self.pt_max, name="pt", label="Electron pT [GeV]"
        ).Reg(
            self.abseta_bins, 0.0, self.abseta_max, name="abseta", label="|eta|"
        ).Double()

    def _make_output(self):
        return {
            "h_all_pt_abseta_cutbased": self._hist_cutbased("cutbased_id"),
            "h_all_pt_abseta_cutbased_noiso": self._hist_cutbased("cutbased_noiso_id"),
            "h_all_pt_abseta_wp": self._hist_wp(),

            "h_lead_pt_abseta_cutbased": self._hist_cutbased("cutbased_id"),
            "h_sublead_pt_abseta_cutbased": self._hist_cutbased("cutbased_id"),

            "h_lead_pt_abseta_cutbased_noiso": self._hist_cutbased("cutbased_noiso_id"),
            "h_sublead_pt_abseta_cutbased_noiso": self._hist_cutbased("cutbased_noiso_id"),

            "h_wp_rank_pt_abseta": self._hist_wp_rank(),
        }

    def process(self, events):
        out = self._make_output()
        events = events[events.DST.PFScouting_SinglePhotonEB]
        electrons = events.ScoutingElectron

        has_besttrack = ("bestTrack_eta" in electrons.fields)
        eta = electrons.bestTrack_eta if has_besttrack else electrons.eta

        electrons = electrons[(electrons.pt > self.electron_pt_min) & (np.abs(eta) < self.abseta_max)]

        if ("ScoutingRho" in events.fields) and ("fixedGridRhoFastjetAll" in events.ScoutingRho.fields):
            rho = events.ScoutingRho.fixedGridRhoFastjetAll
        else:
            rho = ak.zeros_like(events.run, dtype=np.float32) if ("run" in events.fields) else 0.0

        for wp in self.WP:
            base_wp = wp.replace("-noiso", "")
            use_iso = ("noiso" not in wp)
            electrons[f"{wp}_id"] = electron_id_mask(electrons, rho, base_wp, use_iso=use_iso)

        cutbased_id = ak.zeros_like(electrons.pt, dtype=np.int8)
        cutbased_id = ak.where(electrons["Veto_id"], 1, cutbased_id)
        cutbased_id = ak.where(electrons["Loose_id"], 2, cutbased_id)
        cutbased_id = ak.where(electrons["Medium_id"], 3, cutbased_id)
        cutbased_id = ak.where(electrons["Tight_id"], 4, cutbased_id)

        cutbased_noiso_id = ak.zeros_like(electrons.pt, dtype=np.int8)
        cutbased_noiso_id = ak.where(electrons["Veto-noiso_id"], 1, cutbased_noiso_id)
        cutbased_noiso_id = ak.where(electrons["Loose-noiso_id"], 2, cutbased_noiso_id)
        cutbased_noiso_id = ak.where(electrons["Medium-noiso_id"], 3, cutbased_noiso_id)
        cutbased_noiso_id = ak.where(electrons["Tight-noiso_id"], 4, cutbased_noiso_id)

        electrons["cutbased_id"] = cutbased_id
        electrons["cutbased_noiso_id"] = cutbased_noiso_id

        eta2 = electrons.bestTrack_eta if has_besttrack else electrons.eta

        pt_flat = ak.to_numpy(ak.flatten(electrons.pt, axis=None))
        abseta_flat = ak.to_numpy(ak.flatten(np.abs(eta2), axis=None))
        id_flat = ak.to_numpy(ak.flatten(electrons["cutbased_id"], axis=None))
        id_noiso_flat = ak.to_numpy(ak.flatten(electrons["cutbased_noiso_id"], axis=None))

        out["h_all_pt_abseta_cutbased"].fill(pt=pt_flat, abseta=abseta_flat, cutbased_id=id_flat)
        out["h_all_pt_abseta_cutbased_noiso"].fill(pt=pt_flat, abseta=abseta_flat, cutbased_noiso_id=id_noiso_flat)

        for wp in self.WP:
            ele_wp = electrons[electrons[f"{wp}_id"]]
            if ak.sum(ak.num(ele_wp, axis=1)) == 0:
                continue
            eta_wp = ele_wp.bestTrack_eta if has_besttrack else ele_wp.eta
            pt_wp = ak.to_numpy(ak.flatten(ele_wp.pt, axis=None))
            abseta_wp = ak.to_numpy(ak.flatten(np.abs(eta_wp), axis=None))
            if pt_wp.size:
                out["h_all_pt_abseta_wp"].fill(wp=wp, pt=pt_wp, abseta=abseta_wp)

        order = ak.argsort(electrons.pt, axis=1, ascending=False)
        ele_sorted = electrons[order]
        ele_pad2 = ak.pad_none(ele_sorted, 2, axis=1, clip=True)

        lead = ele_pad2[:, 0]
        sublead = ele_pad2[:, 1]

        lead_eta = lead.bestTrack_eta if has_besttrack else lead.eta
        sublead_eta = sublead.bestTrack_eta if has_besttrack else sublead.eta

        lead_valid = _valid_mask(lead.pt)
        sub_valid = _valid_mask(sublead.pt)

        lead_pt = ak.to_numpy(lead.pt[lead_valid])
        lead_abseta = ak.to_numpy(np.abs(lead_eta[lead_valid]))
        lead_id = ak.to_numpy(lead["cutbased_id"][lead_valid])
        lead_id_noiso = ak.to_numpy(lead["cutbased_noiso_id"][lead_valid])

        if lead_pt.size:
            out["h_lead_pt_abseta_cutbased"].fill(pt=lead_pt, abseta=lead_abseta, cutbased_id=lead_id)
            out["h_lead_pt_abseta_cutbased_noiso"].fill(pt=lead_pt, abseta=lead_abseta, cutbased_noiso_id=lead_id_noiso)

        sub_pt = ak.to_numpy(sublead.pt[sub_valid])
        sub_abseta = ak.to_numpy(np.abs(sublead_eta[sub_valid]))
        sub_id = ak.to_numpy(sublead["cutbased_id"][sub_valid])
        sub_id_noiso = ak.to_numpy(sublead["cutbased_noiso_id"][sub_valid])

        if sub_pt.size:
            out["h_sublead_pt_abseta_cutbased"].fill(pt=sub_pt, abseta=sub_abseta, cutbased_id=sub_id)
            out["h_sublead_pt_abseta_cutbased_noiso"].fill(pt=sub_pt, abseta=sub_abseta, cutbased_noiso_id=sub_id_noiso)

        for wp in self.WP:
            lead_pass = ak.fill_none(lead[f"{wp}_id"], False)
            sub_pass = ak.fill_none(sublead[f"{wp}_id"], False)

            lead_mask = lead_pass & lead_valid
            sub_mask = sub_pass & sub_valid

            lead_pt_wp = ak.to_numpy(lead.pt[lead_mask])
            lead_abseta_wp = ak.to_numpy(np.abs(lead_eta[lead_mask]))

            sub_pt_wp = ak.to_numpy(sublead.pt[sub_mask])
            sub_abseta_wp = ak.to_numpy(np.abs(sublead_eta[sub_mask]))

            if lead_pt_wp.size:
                out["h_wp_rank_pt_abseta"].fill(wp=wp, rank="lead", pt=lead_pt_wp, abseta=lead_abseta_wp)
            if sub_pt_wp.size:
                out["h_wp_rank_pt_abseta"].fill(wp=wp, rank="sublead", pt=sub_pt_wp, abseta=sub_abseta_wp)

        return out

    def postprocess(self, accumulator):
        return accumulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--store_path", required=True)
    parser.add_argument("--index", default="0")

    parser.add_argument("--electron_pt_min", type=float, default=5.0)
    parser.add_argument("--abseta_max", type=float, default=2.5)
    parser.add_argument("--pt_bins", type=int, default=2000)
    parser.add_argument("--pt_max", type=float, default=200.0)
    parser.add_argument("--abseta_bins", type=int, default=50)

    parser.add_argument("--chunksize", type=int, default=500000)
    parser.add_argument("--maxchunks", type=int, default=None)
    parser.add_argument("--workers", type=int, default=max(os.cpu_count() - 1, 1))

    args = parser.parse_args()
    os.makedirs(args.store_path, exist_ok=True)

    runner = Runner(
        executor=FuturesExecutor(compression=None, workers=args.workers),
        schema=NanoAODSchema,
        chunksize=args.chunksize,
        maxchunks=args.maxchunks,
    )

    fileset = {"ScoutingDataset": args.files}

    output = runner(
        fileset,
        treename="Events",
        processor_instance=ElectronPtSpectrumProcessor(
            electron_pt_min=args.electron_pt_min,
            abseta_max=args.abseta_max,
            pt_bins=args.pt_bins,
            pt_max=args.pt_max,
            abseta_bins=args.abseta_bins,
        ),
    )

    out_pkl = os.path.join(args.store_path, f"electron_pt_hists_{args.index}.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(output, f)

    print(f"Saved: {out_pkl}")
    for dataset, dsout in output.items():
        print(f"Dataset: {dataset}")
        for k in dsout.keys():
            print(f"  - {k}: {type(dsout[k])}")

    # Note: Hist objects are stored as HistAcc; access via dsout[key].h


if __name__ == "__main__":
    main()

