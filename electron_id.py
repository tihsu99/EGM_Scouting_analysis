import awkward as ak
import numpy as np

def get_ea(abs_eta):
    return ak.where(
        abs_eta < 1.0, 0.1243,
        ak.where(abs_eta < 1.479, 0.1458,
        ak.where(abs_eta < 2.0, 0.0992,
        ak.where(abs_eta < 2.2, 0.0794,
        ak.where(abs_eta < 2.3, 0.0762,
        ak.where(abs_eta < 2.4, 0.0766,
        ak.where(abs_eta < 2.5, 0.1003,
                 0.1003)))))))


def electron_id_mask(electrons,rho, working_point="Loose"):
    """
    Apply 122X-tuned ID cuts to electrons in an awkward array.
    
    Args:
        electrons: Awkward Array with electron fields.
        working_point: One of "Veto", "Loose", "Medium", "Tight".
        
    Returns:
        A boolean mask with the same structure as `electrons`, True where electrons pass the ID.
    """
    # Define thresholds from the table
    thresholds = {
        "barrel": {
            "Veto":    {"sieie": 0.0117,
                        "dEta": 0.0071,
                        "dPhi": 0.208,
                        "H/E": lambda E, p: 0.05 + 1.28/E + 0.0422*p/E,
                        "iso": lambda pt: 0.406 + 0.535/pt,
                        "ooEmooP": 0.178,
                        "missHits": 2
                       },
            "Loose":   {"sieie": 0.0107,
                        "dEta": 0.00691,
                        "dPhi": 0.175,
                        "H/E": lambda E,p: 0.05 + 1.28/E + 0.0422*p/E,
                        "iso": lambda pt: 0.194 + 0.535/pt,
                        "ooEmooP": 0.138,
                        "missHits": 1
                       },
            "Medium":  {"sieie": 0.0103,
                        "dEta": 0.00481,
                        "dPhi": 0.127,
                        "H/E": lambda E,p: 0.0241 + 1.28/E + 0.0422*p/E,
                        "iso": lambda pt: 0.0837 + 0.535/pt, 
                        "ooEmooP": 0.0966,
                        "missHits": 1
                       },
            "Tight":   {"sieie": 0.0101,
                        "dEta": 0.00411,
                        "dPhi": 0.116,
                        "H/E": lambda E,p: 0.02 + 1.16/E + 0.0422*p/E,
                        "iso": lambda pt: 0.0388 + 0.535/pt,
                        "ooEmooP": 0.023,
                        "missHits": 1},
        },
        "endcap": {
            "Veto":    {"sieie": 0.0298,
                        "dEta": 0.0173,
                        "dPhi": 0.234,
                        "H/E": lambda E, p: 0.05 + 2.3/E + 0.262*p/E,
                        "iso": lambda pt: 0.342 + 0.519/pt,
                        "ooEmooP": 0.137,
                        "missHits": 3
                       },
            "Loose":   {"sieie": 0.0275,
                        "dEta": 0.0121, 
                        "dPhi": 0.228,
                        "H/E": lambda E, p: 0.05 + 2.3/E + 0.262*p/E,
                        "iso": lambda pt: 0.184 + 0.519/pt,
                        "ooEmooP": 0.127,
                        "missHits": 1
                       },
            "Medium":  {"sieie": 0.0272,
                        "dEta": 0.00951,
                        "dPhi": 0.221,
                        "H/E": lambda E, p: 0.05 + 2.3/E + 0.262*p/E,
                        "iso": lambda pt: 0.0741 + 0.519/pt,
                        "ooEmooP": 0.111,
                        "missHits": 1
                       },
            "Tight":   {"sieie": 0.027,
                        "dEta": 0.00938,
                        "dPhi": 0.164,
                        "H/E": lambda E, p: 0.02 + 0.5/E + 0.262*p/E,
                        "iso": lambda pt: 0.0544 + 0.519/pt,
                        "ooEmooP": 0.018,
                        "missHits": 1
                       },
        }
    }

    # Compute basic variables
    abs_eta = abs(electrons.eta)
    is_barrel = abs_eta <= 1.479
    is_endcap = ~is_barrel
    pt = electrons.pt
    esc = electrons.rawEnergy
    rho = rho
    effective_area = get_ea(abs_eta)
    sieie = electrons.sigmaIetaIeta
    dEta = abs(electrons.dEtaIn)
    dPhi = abs(electrons.dPhiIn)
    hOverE = electrons.hOverE
    
    neutral_iso = electrons.ecalIso + electrons.hcalIso - rho*effective_area
    iso = (ak.where(neutral_iso > 0, neutral_iso, 0) + electrons.trackIso) / pt
    ooEmooP = abs(electrons.ooEMOop)
    missingHits = electrons.missingHits
    convVeto = True  # Placeholder: assume it’s a flag you will fill externally

    # Evaluate barrel and endcap masks separately
    barrel_cuts = thresholds["barrel"][working_point]
    endcap_cuts = thresholds["endcap"][working_point]

    barrel_mask = (
        is_barrel &
        (sieie < barrel_cuts["sieie"]) &
        (dEta < barrel_cuts["dEta"]) &
        (dPhi < barrel_cuts["dPhi"]) &
        (hOverE < barrel_cuts["H/E"](esc, rho)) &
        (iso < barrel_cuts["iso"](pt)) &
        (ooEmooP < barrel_cuts["ooEmooP"]) &
        (missingHits <= barrel_cuts["missHits"])
    )

    endcap_mask = (
        is_endcap &
        (sieie < endcap_cuts["sieie"]) &
        (dEta < endcap_cuts["dEta"]) &
        (dPhi < endcap_cuts["dPhi"]) &
        (hOverE < endcap_cuts["H/E"](esc, rho)) &
        (iso < endcap_cuts["iso"](pt)) &
        (ooEmooP < endcap_cuts["ooEmooP"]) &
        (missingHits <= endcap_cuts["missHits"])
    )

    return barrel_mask | endcap_mask

