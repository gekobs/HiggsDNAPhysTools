import glob
import argparse
import json
import awkward
import os

from yahist import Hist1D
from yahist.utils import plot_stack
import matplotlib.pyplot as plt

import mplhep as hep
plt.style.use(hep.style.CMS)

def parse_arguments():
    parser = argparse.ArgumentParser(
            description="Make tables from HiggsDNA parquet output files")

    parser.add_argument(
        "--input_dir",
        required=True,
        type=str,
        help="path to directory with HiggsDNA parquet output files")

    parser.add_argument(
        "--signals",
        required=False,
        default=None,
        type=str,
        help="csv list of signal processes (to be excluded from data/MC comparison)")

    parser.add_argument(
        "--group_procs",
        required=False,
        default=None,
        type=str,
        help="csv list of processes to group, with groups separated by '|', e.g. 'SMHiggs:ggH_M125,VBFH_M125|tt+X:ttGG,ttG,tt'")

    parser.add_argument(
        "--plots",
        required=False,
        default=None,
        type=str,
        help="json file with plot options")

    parser.add_argument(
        "--cuts",
        required=False,
        default=None,
        type=str,
        help="|-separated list of cuts, e.g. 'n_jets:[4,999]|LeadPhoton_mvaID:[-0.7,1.0]'")

    parser.add_argument(
        "--output_dir",
        required=False,
        default="output",
        type=str,
        help="output dir to store tables and plots in")

    return parser.parse_args()

def get_weight_var_yields(events, central, up, down):
    n_evts_up = awkward.sum(events["weight_central"] * (events[up] / events[central]))
    n_evts_down = awkward.sum(events["weight_central"] * (events[down] / events[central]))
    return n_evts_up, n_evts_down


def get_yield_and_unc(events, ids):
    evts = events["nominal"]
    cut = evts["process_id"] == -999
    for id in ids:
        cut = (cut) | (evts["process_id"] == id)
    evts = evts[cut]

    n_evts = awkward.sum(evts["weight_central"])
    stat_unc = awkward.sum(evts["weight_central"] * evts["weight_central"])

    syst_unc_up = 0
    syst_unc_down = 0

    weight_fields = [x for x in evts.fields if "weight" in x and "central" in x]
    weight_fields.remove("weight_central")

    for x in weight_fields:
        base = x.replace("weight_", "").replace("_central", "")
        vars_up = [y for y in evts.fields if base in y and "up" in y]
        vars_down = [y for y in evts.fields if base in y and "down" in y]
    
        for a, b in zip(vars_up, vars_down):
            n_up, n_down = get_weight_var_yields(evts, x, a, b)
            syst_unc_up += (n_evts - n_up) ** 2
            syst_unc_down += (n_evts - n_down) ** 2

    ics = [x for x in events.keys() if "up" in x]
    for x in ics:
        x_up = x
        x_down = x.replace("up", "down")

        evts_up = events[x_up]
        cut_up = evts_up["process_id"] == -999
        for id in ids:
            cut_up = (cut_up) | (evts_up["process_id"] == id)
        evts_up = evts_up[cut_up]

        evts_down = events[x_down]
        cut_down = evts_down["process_id"] == -999
        for id in ids:
            cut_down = (cut_down) | (evts_down["process_id"] == id)
        evts_down = evts_down[cut_down] 

        n_up = awkward.sum(evts_up["weight_central"])
        n_down = awkward.sum(evts_down["weight_central"])

        if n_up > 0:
            syst_unc_up += (n_evts - n_up) ** 2
        if n_down > 0:
            syst_unc_down += (n_evts - n_down) ** 2

    stat_unc = stat_unc**(0.5)
    syst_unc_up = syst_unc_up**(0.5)
    syst_unc_down = syst_unc_down**(0.5)

    return n_evts, stat_unc, syst_unc_up, syst_unc_down


def print_table(results, title):
    results["total_bkg"] = {
            "n" : 0,
            "stat_unc" : 0,
            "syst_unc_up" : 0,
            "syst_unc_down" : 0,
            "type" : "total_bkg"
    }

    for proc, yields in results.items():
        if yields["type"] == "bkg":
            results["total_bkg"]["n"] += yields["n"]
            results["total_bkg"]["stat_unc"] = (yields["stat_unc"]**2 + results["total_bkg"]["stat_unc"]**2)**(0.5)
            results["total_bkg"]["syst_unc_up"] = (yields["syst_unc_up"]**2 + results["total_bkg"]["syst_unc_up"]**2)**(0.5)
            results["total_bkg"]["syst_unc_down"] = (yields["syst_unc_down"]**2 + results["total_bkg"]["syst_unc_down"]**2)**(0.5)

    print("\\begin{center}")
    print("\\begin{tabular}{ l | r r r | r}")
    print("\\multicolumn{5}{c}{\\texttt{HiggsDNAPhysTool} : %s} \\\\ \\hline \\hline" % title) 
    print("Process & Yield & Stat. unc. & Syst. unc. & $\\mathcal F$ of bkg \\\\ \\hline")

    for proc, yields in results.items():
        if yields["type"] == "sig":
            if results["total_bkg"]["n"] > 0.:
                bkg_pct = (1. * yields["n"]) / results["total_bkg"]["n"]
            else:
                bkg_pct = 0.
            print("%s & %.3f & $\\pm \\text{%.3f}$  & $~^{+\\text{%.3f}}_{-\\text{%.3f}}$  & %.4f \\\\" % (proc.replace("_","-"), yields["n"], yields["stat_unc"], yields["syst_unc_up"], yields["syst_unc_down"], bkg_pct))
    print("\\hline")

    for proc, yields in results.items():
        if yields["type"] == "bkg":
            if results["total_bkg"]["n"] > 0.:
                bkg_pct = (yields["n"]) / results["total_bkg"]["n"]
            else:
                bkg_pct = 0.
            print("%s & %.3f & $\\pm \\text{%.3f}$  & $~^{+\\text{%.3f}}_{-\\text{%.3f}}$  & %.4f \\\\" % (proc.replace("_","-"), yields["n"], yields["stat_unc"], yields["syst_unc_up"], yields["syst_unc_down"], bkg_pct))
    print("\\hline")

    yields = results["total_bkg"]
    print("Total MC bkg & %.3f & $\\pm \\text{%.3f}$  & $~^{+\\text{%.3f}}_{-\\text{%.3f}}$  & 1 \\\\ \\hline" % (yields["n"], yields["stat_unc"], yields["syst_unc_up"], yields["syst_unc_down"]))
    

    if "Data" in results.keys():
        yields = results["Data"]
        print("Data & %.3f & $\\pm \\text{%.3f}$  & & %.4f \\\\ \\hline" % (yields["n"], yields["stat_unc"], (1. * yields["n"]) / results["total_bkg"]["n"]))

    
    print("\\end{tabular}")
    print("\\end{center}")


def events_with_ids(events, ids):
    cut = events["process_id"] == -999
    for id in ids:
        cut = (cut) | (events["process_id"] == id)
    return events[cut]


def do_cut(events, field, range):
    cut = (events[field] >= range[0]) & (events[field] <= range[1])
    return events[cut]


def make_data_mc_plot(data, bkg, sig, savename, **kwargs):
    normalize = kwargs.get("normalize", False)
    x_label = kwargs.get("x_label", None)
    y_label = kwargs.get("y_label", "Events" if not normalize else "Fraction of events")
    rat_label = kwargs.get("rat_label", "Data/MC")
    title = kwargs.get("title", None)
    y_lim = kwargs.get("y_lim", None)
    x_lim = kwargs.get("x_lim", None)
    rat_lim = kwargs.get("rat_lim", [0.0, 2.0])
    overflow = kwargs.get("overflow", False)
    log_y = kwargs.get("log_y", False)

    bins = kwargs.get("bins")  

    h_data = Hist1D(data, bins=bins, overflow=overflow, label="Data")

    h_bkg = []
    h_bkg_syst_up = []
    h_bkg_syst_down = []
    for proc, plot_data in bkg.items():
        h = Hist1D(plot_data["array"], weights = plot_data["weights"], bins = bins, overflow = overflow, label=proc)
        h_bkg.append(h) 


    h_bkg_total = None 

    for h in h_bkg:
        if h_bkg_total is None:
            h_bkg_total = h.copy()
        else:
            h_bkg_total += h        

    h_bkg_total_syst_up = None
    h_bkg_total_syst_down = None  


    h_sig = []
    for proc, plot_data in sig.items():
        h = Hist1D(plot_data["array"], weights = plot_data["weights"], bins = bins, overflow = overflow, label=proc) 
        h_sig.append(h)

    fig, (ax1,ax2) = plt.subplots(2, sharex=True, figsize=(12,9), gridspec_kw=dict(height_ratios=[3, 1]))
    plt.grid()
    h_data.plot(ax=ax1, color = "black", errors = True)
    plt.sca(ax1)
    hep.cms.label(" Preliminary",loc=0,data=True,lumi=137,fontsize=18)

    stack = sorted(h_bkg, key = lambda x : x.integral)
    plot_stack(stack, ax=ax1, histtype="stepfilled")

    for idx, h in enumerate(h_sig):
        h.plot(ax=ax1, color = "C%d" % (idx+5), errors = False, linewidth=3)

    ratio = h_data.divide(h_bkg_total, binomial = True)
    ratio._errors = h_data.errors / h_bkg_total.counts
   
    mc_stat_err_up = 1 + (0.5*(h_bkg_total.errors / h_bkg_total.counts))
    mc_stat_err_down = 1 - (0.5*(h_bkg_total.errors / h_bkg_total.counts))

    ax2.fill_between(h_data.bin_centers, mc_stat_err_down, mc_stat_err_up, color="black", alpha=0.25)
    ratio.metadata["label"] = None
    ratio.plot(ax=ax2, errors=True, color="black")

    if x_label is not None:
        ax2.set_xlabel(x_label)

    if y_label is not None:
        ax1.set_ylabel(y_label)

    if rat_label is not None:
        ax2.set_ylabel(rat_label)

    if title is not None:
        ax1.set_title(title)

    if y_lim is not None:
        ax1.set_ylim(y_lim)

    if rat_lim is not None:
        ax2.set_ylim(rat_lim)

    if x_lim is not None:
        ax1.set_xlim(x_lim)

    if log_y:
        ax1.set_yscale("log")

    plt.savefig(savename)



def main(args):
    inputs = glob.glob(args.input_dir + "/merged_*.parquet")
    
    with open(args.input_dir + "/summary.json", "r") as f_in:
        process_map = json.load(f_in)["sample_id_map"]

    for proc, id in process_map.items():
        process_map[proc] = [id]

    if args.group_procs is not None:
        groups = args.group_procs.split("|")
        for group in groups:
            new_name = group.split(":")[0]
            procs = group.split(":")[1].split(",")
            process_map[new_name] = []
            for proc in procs:
                process_map[new_name] += [process_map[proc]]
                process_map.pop(proc)

    signals = []
    if args.signals is not None:
        signals = args.signals.split(",")

    events = {}
    for input in inputs:
        name = input.split("/")[-1].replace("merged_", "").replace(".parquet","")
        evts = awkward.from_parquet(input)
        if args.cuts is not None:
            cuts = args.cuts.split("|")
            for cut in cuts:
                field = cut.split(":")[0]
                range = [float(x.replace("[","").replace("]","")) for x in cut.split(":")[1].split(",")]
                evts = do_cut(evts, field, range)
        events[name] = evts


    results = {}

    for process, id in process_map.items():
        n, stat_unc, syst_unc_up, syst_unc_down = get_yield_and_unc(events, id)
        results[process] = {
                "n" : n,
                "stat_unc" : stat_unc,
                "syst_unc_up" : syst_unc_up,
                "syst_unc_down" : syst_unc_down
        }

        if process == "Data":
            results[process]["type"] = "data"
        else:
            is_sig = process in signals
            if is_sig:
                results[process]["type"] = "sig"
            else:
                results[process]["type"] = "bkg"

    print_table(results, "Inclusive")
 
    os.system("mkdir -p %s" % args.output_dir)

    if args.plots is not None:
        with open(args.plots, "r") as f_in:
            plot_config = json.load(f_in)

        nom = events["nominal"]
        for field, info in plot_config.items():
            data = events_with_ids(nom, process_map["Data"])[field]
            bkg = {}
            sig = {}
            for proc, ids in process_map.items():
                if proc == "Data":
                    continue
                array = events_with_ids(nom, ids)[field]
                weights = events_with_ids(nom, ids)["weight_central"]
                plot_data = { "array" : array, "weights" : weights, "syst_array" : [], "syst_weights" : [] }

                syst_weight_fields = [x for x in nom.fields if "weight" in x and "central" in x]
                syst_weight_fields.remove("weight_central")
                for x in syst_weight_fields:
                    base = x.replace("weight_", "").replace("_central", "")
                    vars_up = [y for y in evts.fields if base in y and "up" in y]
                    vars_down = [y for y in evts.fields if base in y and "down" in y]

                    for up, down in zip(vars_up, vars_down):
                        weight_up = nom["weight_central"] * (nom[up] / nom[x])
                        weight_down = nom["weight_central"] * (nom[down] / nom[x])
                        plot_data["syst_weights"] += [weight_up, weight_down] 

                if proc in signals:
                    sig[proc] = plot_data 
                else:
                    bkg[proc] = plot_data

            make_data_mc_plot(data, bkg, sig, savename = "%s/%s_data_mc_HiggsDNAPhysTools.pdf" % (args.output_dir, field), **info)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

