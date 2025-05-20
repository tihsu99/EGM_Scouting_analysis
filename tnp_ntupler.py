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
                "inv-mass":  processor.column_accumulator(np.array([])),
                "probe-pt": processor.column_accumulator(np.array([])),
                "probe-eta": processor.column_accumulator(np.array([])),
                "probe-id": processor.column_accumulator(np.array([])),
                **{
                    key: processor.column_accumulator(np.array(val))
                    for key, val in self.DST.items()
                }
            })
        })
        self.histograms = {
            wp: Hist.new.Reg(200, 0, 200, name="mass", label="Invariant Mass [GeV]").Double()
            for wp in ["Veto", "Loose", "Medium", "Tight"]
        }

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()
        electrons = events.ScoutingElectron
        electrons = electrons[electrons.pt > self.electron_pt_min]
        electrons = electrons[abs(electrons.bestTrack_eta) < 2.5]
        events = events[ak.num(electrons) > 1]
        electrons = events.ScoutingElectron
        electrons = electrons[electrons.pt > self.electron_pt_min]
        electrons = electrons[abs(electrons.bestTrack_eta) < 2.5]
        rho = events.ScoutingRho.fixedGridRhoFastjetAll if "ScoutingRho" in events.fields else 0

        for wp in ["Veto", "Loose", "Medium", "Tight"]:
            electrons[f"{wp}_id"] = electron_id_mask(electrons, rho, wp)
        # Initialize with zeros (i.e., fails all WPs)
        cutbased_id = ak.zeros_like(electrons.pt, dtype=np.int8)
        # Assign values, tighter WPs overwrite looser ones
        cutbased_id = ak.where(electrons["Veto_id"], 1, cutbased_id)
        cutbased_id = ak.where(electrons["Loose_id"], 2, cutbased_id)
        cutbased_id = ak.where(electrons["Medium_id"], 3, cutbased_id)
        # Add to the electrons array
        electrons["cutbased_id"] = cutbased_id


        electrons["pass_tag"] = (electrons["pt"] > self.tag_selection["pt-min"]) & (abs(electrons["eta"]) < self.tag_selection["abseta-max"]) & (electrons[self.tag_selection["id"]])
        electrons = electrons[ak.argsort(electrons.pt, axis = -1, ascending=False)]


        # Plot Invariant Mass diagram for each WP
        for wp in ["Veto", "Loose", "Medium", "Tight"]:
            electrons_wp = electrons[electrons[f"{wp}_id"]]

            valid_pair = ak.num(electrons_wp.pt) == 2

            vecs = ak.zip({
                "pt": electrons_wp.pt,
                "eta": electrons_wp.bestTrack_etaMode,
                "phi": electrons_wp.bestTrack_phiMode,
                "mass": ak.zeros_like(electrons_wp.pt),
            }, with_name="Momentum4D")

            vecs = vecs[valid_pair]

            masses = (vecs[:, 0] + vecs[:, 1]).mass
            self.histograms[wp].fill(mass=ak.to_numpy(masses))


        # TnP selection

        event_selection = (electrons[:, 0]["pass_tag"]) & (electrons[:, 1]["Veto_id"])
   
        electrons_selection = electrons[event_selection]
        vecs = ak.zip({
            "pt": electrons_selection.pt,
            "eta": electrons_selection.bestTrack_eta,
            "phi": electrons_selection.bestTrack_phi,
            #"eta": electrons_selection.eta,
            #"phi": electrons_selection.phi,
            "mass": ak.zeros_like(electrons_selection.pt),
        }, with_name="Momentum4D")

        tag_electron = electrons_selection[:, 0]
        probe_electron = electrons_selection[:, 1]
        inv_mass = (vecs[:,0] + vecs[:,1]).mass

        output["filtered_events"]["tag-pt"] += processor.column_accumulator(ak.to_numpy(tag_electron.pt))
        output["filtered_events"]["tag-eta"] += processor.column_accumulator(ak.to_numpy(tag_electron.eta))
        output["filtered_events"]["inv-mass"] += processor.column_accumulator(ak.to_numpy(inv_mass))
        output["filtered_events"]["probe-pt"] += processor.column_accumulator(ak.to_numpy(probe_electron.pt))
        output["filtered_events"]["probe-eta"] += processor.column_accumulator(ak.to_numpy(probe_electron.eta))
        output["filtered_events"]["probe-id"] += processor.column_accumulator(ak.to_numpy(probe_electron.cutbased_id))
        for DST in self.DST:
            output["filtered_events"][DST] += processor.column_accumulator(ak.to_numpy(events[event_selection]["DST"][DST]))



        return {
            "columns": output["filtered_events"],
            "histograms": self.histograms
        }

    def postprocess(self, accumulator):
        pass
# === Run and plot ===
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Explore the structure of an HDF5 file")
    parser.add_argument("--store_path", help="Path to the HDF5 file")
    # Parse command-line arguments
    args = parser.parse_args()


    filename = "root://cms-xrd-global.cern.ch///store/user/asahasra/ScoutingPFRun3/Scouting_2024F_crabNano250506_TestSubmit/250506_101748/0000/scouting_nano_2.root"


    tag_selection = {
        "pt-min": 35,
        "abseta-max": 1.47,
        "id": "Tight_id"
    }

    runner = Runner(
        executor=FuturesExecutor(compression=None, workers = 20),
        schema=NanoAODSchema,
        chunksize=200000,
        maxchunks=None,
    )

    fileset = {
        "ScoutingDataset": [
            f"root://cms-xrd-global.cern.ch///store/user/asahasra/ScoutingPFRun3/Scouting_2024F_crabNano250506_TestSubmit/250506_101748/0000/scouting_nano_{i}.root" for i in range(1, 201)
        ]
    }
    output = runner(
        fileset,
        treename="Events",
        processor_instance=ElectronInvariantMassProcessor(tag_selection = tag_selection, electron_pt_min=5),
    )


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
    plt.savefig(f"invariant_mass_{wp}.png")

    for k, v in output["columns"].items():
      print(k, len(v.value))

    print(output["columns"]) 
    # Save filtered events (as NumPy or print summary)
    df = pd.DataFrame({k:v.value for k,v in output["columns"].items()})
    print(df)

    os.makedirs(args.store_path, exist_ok=True)
    df.to_hdf(os.path.join(args.store_path, "output.h5"), key="data", mode='w')

