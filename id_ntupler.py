import awkward as ak
import numpy as np
import vector
import matplotlib.pyplot as plt
from coffea import processor
from coffea.processor import Runner, FuturesExecutor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from electron_id import electron_id_mask
# Register vector methods
vector.register_awkward()
from hist import Hist
import subprocess
import re
import pandas as pd
import argparse
import os

class ElectronInvariantMassProcessor(processor.ProcessorABC):
    def __init__(self, tag_selection, electron_pt_min=5):
        self.electron_pt_min = electron_pt_min
        self.tag_selection = tag_selection
        self.WP = ["Veto", "Loose", "Medium", "Tight", "Veto-noiso", "Loose-noiso", "Medium-noiso", "Tight-noiso"]
        self.DST = {
          "PFScouting_DoubleEG": [],
          "PFScouting_SingleMuon": [],
          "PFScouting_JetHT": [],
          "PFScouting_ZeroBias": [],
          "PFScouting_SinglePhotonEB": []
        }

        self._accumulator = processor.dict_accumulator({
            "filtered_events": processor.dict_accumulator({
                "tag-pt": processor.column_accumulator(np.array([])),
                "tag-eta": processor.column_accumulator(np.array([])),
                "tag-id": processor.column_accumulator(np.array([])),
                "tag-noiso-id":  processor.column_accumulator(np.array([])),
                "inv-mass":  processor.column_accumulator(np.array([])),
                "probe-pt": processor.column_accumulator(np.array([])),
                "probe-eta": processor.column_accumulator(np.array([])),
                "probe-id": processor.column_accumulator(np.array([])),
                "probe-noiso-id": processor.column_accumulator(np.array([])),
                "opposite-charge": processor.column_accumulator(np.array([])),
                **{
                    key: processor.column_accumulator(np.array(val))
                    for key, val in self.DST.items()
                }}),
            "inv-mass": processor.dict_accumulator({
                **{
                    f"inv-mass-{wp}": processor.column_accumulator(np.array([]))
                    for wp in self.WP
                },
                **{
                    f"inv-mass-EBEB-{wp}": processor.column_accumulator(np.array([]))
                    for wp in self.WP
                },

            })
        })
        self.histograms = {
            wp: Hist.new.Reg(200, 0, 200, name="mass", label="Invariant Mass [GeV]").Double()
            for wp in self.WP
        }

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()
        electrons = events.ScoutingElectron
        electrons = electrons[electrons.pt > self.electron_pt_min]
        electrons = electrons[abs(electrons.bestTrack_eta) < 2.5]
        events["tight_electrons"] = electrons
        events = events[ak.num(electrons) > 1]
        electrons = events["tight_electrons"]
        rho = events.ScoutingRho.fixedGridRhoFastjetAll if "ScoutingRho" in events.fields else 0

        for wp in self.WP:
            electrons[f"{wp}_id"] = electron_id_mask(electrons, rho, wp.replace("-noiso", ""), use_iso=(not ("noiso" in wp)))
        # Initialize with zeros (i.e., fails all WPs)
        cutbased_id = ak.zeros_like(electrons.pt, dtype=np.int8)
        # Assign values, tighter WPs overwrite looser ones
        cutbased_id = ak.where(electrons["Veto_id"], 1, cutbased_id)
        cutbased_id = ak.where(electrons["Loose_id"], 2, cutbased_id)
        cutbased_id = ak.where(electrons["Medium_id"], 3, cutbased_id)
        cutbased_id = ak.where(electrons["Tight_id"], 4, cutbased_id)

        electrons["cutbased_id"] = cutbased_id

        cutbased_noiso_id = ak.zeros_like(electrons.pt, dtype=np.int8)
        # Assign values, tighter WPs overwrite looser ones
        cutbased_noiso_id = ak.where(electrons["Veto-noiso_id"], 1, cutbased_noiso_id)
        cutbased_noiso_id = ak.where(electrons["Loose-noiso_id"], 2, cutbased_noiso_id)
        cutbased_noiso_id = ak.where(electrons["Medium-noiso_id"], 3, cutbased_noiso_id)
        cutbased_noiso_id = ak.where(electrons["Tight-noiso_id"], 4, cutbased_noiso_id)

        # Add to the electrons array
        electrons["cutbased_noiso_id"] = cutbased_noiso_id


        electrons["pass_tag"] = (electrons["pt"] > self.tag_selection["pt-min"]) & (abs(electrons["eta"]) < self.tag_selection["abseta-max"]) & (electrons[self.tag_selection["id"]])
        electrons = electrons[ak.argsort(electrons.pt, axis = -1, ascending=False)]
        electrons["momentum"] = ak.zip({
                "pt": electrons.pt,
                "eta": electrons.bestTrack_etaMode,
                "phi": electrons.bestTrack_phiMode,
                "mass": ak.zeros_like(electrons.pt),
            }, with_name="Momentum4D")

        # Plot Invariant Mass diagram for each WP
        for wp in self.WP:
            electrons_wp = electrons[electrons[f"{wp}_id"]]

            valid_pair = (ak.num(electrons_wp.pt) > 1) & (events.DST.PFScouting_DoubleEG)

            electrons_wp = electrons_wp[valid_pair]

            valid_pair = ((electrons_wp[:, 0].bestTrack_charge * electrons_wp[:, 1].bestTrack_charge) < 0)
            valid_pair_EBEB = (abs(electrons_wp[:, 0].eta) < 1.47) & (abs(electrons_wp[:, 1].eta) < 1.47)

            vecs = electrons_wp["momentum"][valid_pair]
            vecs_EBEB = electrons_wp["momentum"][valid_pair_EBEB]

            masses = (vecs[:, 0] + vecs[:, 1]).mass
            self.histograms[wp].fill(mass=ak.to_numpy(masses))
            output["inv-mass"][f"inv-mass-{wp}"] += processor.column_accumulator(ak.to_numpy(masses))

            masses_EBEB = (vecs_EBEB[:, 0] + vecs_EBEB[:, 1]).mass
            output["inv-mass"][f"inv-mass-EBEB-{wp}"] += processor.column_accumulator(ak.to_numpy(masses_EBEB))


        # TnP selection

        event_selection = ((electrons[:, 0]["pass_tag"]) & (electrons[:, 1]["Veto-noiso_id"])) | ((electrons[:, 1]["pass_tag"]) & (electrons[:, 0]["Veto-noiso_id"]))
   
        electrons_selection = electrons[event_selection]
        events_filtered = events[event_selection]
        vecs = electrons_selection.momentum


        for tag_idx, probe_idx in [(0, 1), (1,0)]:
            tag_electron = electrons_selection[:, tag_idx]
            probe_electron = electrons_selection[:, probe_idx]

            selection = tag_electron["pass_tag"]
            electrons_selection_tnp = electrons_selection[selection]
            events_filtered_tnp = events_filtered[selection]

            tag_electron = electrons_selection_tnp[:, tag_idx]
            probe_electron = electrons_selection_tnp[:, probe_idx]

            inv_mass = (tag_electron.momentum + probe_electron.momentum).mass
            opposite_charge = (tag_electron.bestTrack_charge *  probe_electron.bestTrack_charge) < 0
            
            output["filtered_events"]["tag-pt"] += processor.column_accumulator(ak.to_numpy(tag_electron.pt))
            output["filtered_events"]["tag-eta"] += processor.column_accumulator(ak.to_numpy(tag_electron.eta))
            output["filtered_events"]["tag-id"] += processor.column_accumulator(ak.to_numpy(tag_electron.cutbased_id))
            output["filtered_events"]["tag-noiso-id"] += processor.column_accumulator(ak.to_numpy(tag_electron.cutbased_noiso_id))
            output["filtered_events"]["inv-mass"] += processor.column_accumulator(ak.to_numpy(inv_mass))
            output["filtered_events"]["opposite-charge"] += processor.column_accumulator(ak.to_numpy(opposite_charge))
            output["filtered_events"]["probe-pt"] += processor.column_accumulator(ak.to_numpy(probe_electron.pt))
            output["filtered_events"]["probe-eta"] += processor.column_accumulator(ak.to_numpy(probe_electron.eta))
            output["filtered_events"]["probe-id"] += processor.column_accumulator(ak.to_numpy(probe_electron.cutbased_id))
            output["filtered_events"]["probe-noiso-id"] += processor.column_accumulator(ak.to_numpy(probe_electron.cutbased_noiso_id))
            for DST in self.DST:
                output["filtered_events"][DST] += processor.column_accumulator(ak.to_numpy(events_filtered_tnp["DST"][DST]))



        return {
            "columns": output["filtered_events"],
            "histograms": self.histograms,
            "inv-mass": output["inv-mass"]
        }

    def postprocess(self, accumulator):
        pass
# === Run and plot ===
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Explore the structure of an HDF5 file")
    parser.add_argument("--store_path", help="Path to the HDF5 file")
    parser.add_argument("--index", default = 0)
    parser.add_argument("--files", nargs = "+")
    # Parse command-line arguments
    args = parser.parse_args()


#    filename = "root://cms-xrd-global.cern.ch///store/user/asahasra/ScoutingPFRun3/Scouting_2024F_crabNano250506_TestSubmit/250506_101748/0000/scouting_nano_2.root"
#    filename = "root://cms-xrd-global.cern.ch///store/user/asahasra/ScoutingPFRun3/Scouting_2024F_crabNano250527_DiElSkim/250527_081921/0000/scouting_nano_100.root"

    tag_selection = {
        "pt-min": 20,
        "abseta-max": 1.47,
        "id": "Veto-noiso_id"
    }

    print(f"run with cpu: {os.cpu_count() - 1}")

    runner = Runner(
        executor=FuturesExecutor(compression=None, workers = os.cpu_count() - 1),
        schema=NanoAODSchema,
        chunksize=500000,
        maxchunks=None,
    )

    fileset = {
        "ScoutingDataset": args.files
    }
    output = runner(
        fileset,
        treename="Events",
        processor_instance=ElectronInvariantMassProcessor(tag_selection = tag_selection, electron_pt_min=5),
    )

    os.makedirs(args.store_path, exist_ok=True)
    # Plot histograms
    plt.figure()
    for wp, hist in output["histograms"].items():
        plt.hist(hist.axes[0].centers, weights=hist.values(), bins=hist.axes[0].edges, histtype='step', label=wp)
        plt.title(f"Invariant Mass of Leading Two Electrons ({wp} ID)")
        plt.xlabel("Mass [GeV]")
        plt.ylabel("Events")
        plt.grid()
        plt.legend()
    plt.yscale('log')  # This enables logarithmic scale on y-axis
    plt.savefig(os.path.join(args.store_path, f"invariant_mass_summary.png"))

    for k, v in output["columns"].items():
      print(k, len(v.value))

    print(output["columns"]) 
    # Save filtered events (as NumPy or print summary)
    df = pd.DataFrame({k:v.value for k,v in output["columns"].items()})
    inv_mass_array = {k:v.value for k,v in output["inv-mass"].items()}
    print(df)

    df.to_hdf(os.path.join(args.store_path, f"output_{args.index}.h5"), key="data", mode='w')
    np.savez(os.path.join(args.store_path, f"inv_mass_data_{args.index}.npz"), **inv_mass_array)

