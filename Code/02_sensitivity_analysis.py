"""
02_sensitivity_analysis.py
==========================
Step 2 of the FCM analysis pipeline for:
  "Fuzzy Cognitive Mapping as a Dynamic Decision-Support Tool for Heritage Site
   Resilience: Evidence from Integrated Flood and Tourism Management at Qalhat, Oman"

What this script does
---------------------
1. Reloads the FCM data and re-runs the baseline.
2. Runs all five management scenarios (Fb, Tb, FTb-strong, FTb-medium, FTb-weak)
   plus the baseline, plotted in a 2×3 subplot grid — saved to Appendices/.
3. Produces the main sensitivity bar chart: percentage change from baseline for
   the five scenarios across key outcome concepts — saved to Results/.
4. Repeats step 3 after activating the bridge concept C39 (Usage Activity),
   allowing comparison of outcomes with/without this structural lever — saved to Results/.
5. Exports final activation values and percentage-change tables as CSV files.

Intensity levels tested (for flood in FTb scenarios)
-----------------------------------------------------
  Strong  — 1.0
  Medium  — 0.5
  Weak    — 0.25
Tourism is held constant at 0.5 across all FTb scenarios.

Outputs (written to Results/ and Appendices/)
---------------------------------------------
  Appendices/Sensitivity analysis_with_Baseline.png / .pdf
  Results/Percentage_Change_Comparison.png / .pdf
  Results/final_activation_values_table.csv
  Results/percentage_change_table.csv
  Results/Comparison_with Usage_Baseline.png / .pdf

Data paths (Input data)
-----------------------------------------
  DATA_DIR/Aggregated participatory FCM.csv
  DATA_DIR/initial values.csv
  DATA_DIR/Concept labels.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from fcmpy import FcmSimulator

# ===========================================================================
# 0.  CONFIGURATION
# ===========================================================================
DATA_DIR    = "Data"
RESULTS_DIR = "Results"
APPEND_DIR  = "Appendices"

MATRIX_PATH = os.path.join(DATA_DIR,   "Aggregated participatory FCM.csv")
INIT_PATH   = os.path.join(DATA_DIR,   "initial values.csv")
LABELS_PATH = os.path.join(DATA_DIR, "Concept labels.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(APPEND_DIR,  exist_ok=True)

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       12,
    "axes.labelsize":  13,
    "axes.titlesize":  12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "lines.linewidth": 2,
    "lines.markersize": 5,
})

# ===========================================================================
# 1.  LOAD DATA
# ===========================================================================
fcm_data      = pd.read_csv(MATRIX_PATH, index_col=0)
concept_ids   = fcm_data.columns.tolist()
weight_matrix = fcm_data.astype(float).values

labels_df         = pd.read_csv(LABELS_PATH)
concept_label_map = dict(zip(labels_df["Concept ID"], labels_df["Label"]))
concept_labels    = [concept_label_map.get(cid, cid) for cid in concept_ids]


def get_initial_values() -> dict:
    """Return {label: initial_activation} for the baseline state vector."""
    df = pd.read_csv(INIT_PATH)
    return {
        concept_label_map.get(cid, cid):
            float(df.loc[df["Concept ID"] == cid, "Initial Value"].values[0])
            if cid in df["Concept ID"].values else 0.0
        for cid in concept_ids
    }


# ===========================================================================
# 2.  SIMULATION HELPER
# ===========================================================================
sim = FcmSimulator()


def simulate_fcm(initial_state: dict, wm: np.ndarray, iterations: int = 20) -> pd.DataFrame:
    """
    Kosko inference + sigmoid transfer.  Inactive concepts are masked before
    simulation so they do not propagate influence.

    Parameters
    ----------
    initial_state : dict
        {concept_label: activation_value}
    wm : np.ndarray
        FCM weight matrix.
    iterations : int
        Maximum update iterations (convergence threshold ε = 0.001).

    Returns
    -------
    pd.DataFrame  — activation values per iteration × concept label.
    """
    mod_matrix = wm.copy()
    inactive   = [c for c, v in initial_state.items() if float(v) == 0.0]
    for c in inactive:
        if c in concept_labels:
            idx = concept_labels.index(c)
            mod_matrix[:, idx] = 0
            mod_matrix[idx, :] = 0
    result = sim.simulate(
        initial_state=initial_state,
        weight_matrix=mod_matrix,
        transfer="sigmoid",
        inference="kosko",
        thresh=0.001,
        iterations=iterations,
        l=1,
    )
    return pd.DataFrame(result, columns=concept_labels)


# ===========================================================================
# 3.  BASELINE
# ===========================================================================
init_state  = get_initial_values()
baseline_df = simulate_fcm(init_state, weight_matrix)
last_values = {k: float(v) for k, v in baseline_df.iloc[-1].to_dict().items()}
print("Baseline simulation complete.")

# ===========================================================================
# 4.  INTERVENTION BUILDER
# ===========================================================================
def apply_intervention(name: str, base_state: dict,
                        activate_bridge: bool = False) -> dict:
    """
    Build a scenario initial-state vector.

    Parameters
    ----------
    name : str
        One of: 'Fb', 'Tb', 'FTb_strong', 'FTb_medium', 'FTb_weak'
    base_state : dict
        Steady-state from the baseline simulation.
    activate_bridge : bool
        If True, additionally set C39 (Usage Activity) = 0.25
        to test the effect of activating this bridge concept.

    Returns
    -------
    dict — initial state for the intervention scenario.
    """
    s  = base_state.copy()
    lm = concept_label_map

    if name == "Fb":
        s.update({lm["C15"]: 1.0, lm["C11"]: 1.0,
                  lm["C21"]: 0.1, lm["C27"]: 1.0, lm["C28"]: 0.1})

    elif name == "Tb":
        s.update({lm["C40"]: 1.0, lm["C21"]: 0.1, lm["C28"]: 0.1})

    elif name == "FTb_strong":
        s.update({lm["C15"]: 1.0,  lm["C11"]: 1.0,  lm["C40"]: 0.5,
                  lm["C21"]: 0.1,  lm["C27"]: 1.0,  lm["C28"]: 0.1})

    elif name == "FTb_medium":
        s.update({lm["C15"]: 0.5,  lm["C11"]: 0.5,  lm["C40"]: 0.5,
                  lm["C21"]: 0.1,  lm["C27"]: 1.0,  lm["C28"]: 0.1})

    elif name == "FTb_weak":
        s.update({lm["C15"]: 0.25, lm["C11"]: 0.25, lm["C40"]: 0.5,
                  lm["C21"]: 0.1,  lm["C27"]: 1.0,  lm["C28"]: 0.1})

    if activate_bridge:
        s[lm["C39"]] = 0.25   # activate Usage Activity bridge concept

    return s


# ===========================================================================
# 5.  APPENDIX GRID — baseline + 5 scenarios (2 × 3 subplots)
# ===========================================================================
plot_scenarios = {
    "Flood with management response (Fb)":        apply_intervention("Fb",        last_values),
    "Tourism with management response (Tb)":       apply_intervention("Tb",        last_values),
    "Flood + Tourism (FTb strong-flood)":          apply_intervention("FTb_strong",last_values),
    "Flood + Tourism (FTb medium-flood)":          apply_intervention("FTb_medium",last_values),
    "Flood + Tourism (FTb weak-flood)":            apply_intervention("FTb_weak",  last_values),
}

target_ids    = ["C1", "C2", "C3", "C7", "C15", "C40", "C8"]
target_labels = [concept_label_map[cid] for cid in target_ids]
concept_colors = {
    "C1": "gray", "C2": "#C4A484", "C3": "green",
    "C7": "purple", "C8": "red", "C15": "blue", "C40": "orange",
}

all_names = ["Baseline"] + list(plot_scenarios.keys())
n_plots   = len(all_names)
cols, rows = 3, int(np.ceil(n_plots / 3))

fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), sharex=True)
axes = axes.flatten()

# Baseline subplot
ax   = axes[0]
x_bl = range(1, len(baseline_df) + 1)
for cid, lbl in zip(target_ids, target_labels):
    ax.plot(x_bl, baseline_df[lbl], marker="o",
            color=concept_colors.get(cid, "black"), label=lbl)
ax.set_title("Baseline")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.2); ax.spines["bottom"].set_linewidth(1.2)
ax.tick_params(direction="out", length=4, width=1); ax.grid(False)

# Scenario subplots
scenario_results = {}
for i, (name, state) in enumerate(plot_scenarios.items(), start=1):
    df_res = simulate_fcm(state, weight_matrix)
    scenario_results[name] = df_res
    ax = axes[i]
    x  = range(1, len(df_res) + 1)
    for cid, lbl in zip(target_ids, target_labels):
        ax.plot(x, df_res[lbl], marker="o",
                color=concept_colors.get(cid, "black"))
    ax.set_title(name)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2); ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(direction="out", length=4, width=1); ax.grid(False)

for ax in axes[n_plots:]:
    ax.axis("off")

fig.supxlabel("Iterations", fontsize=14)
fig.supylabel("Concept Level", fontsize=14)
handles, lbls = axes[0].get_legend_handles_labels()
fig.legend(handles, lbls, bbox_to_anchor=(1.02, 0.5), loc="center left")
plt.tight_layout(rect=[0, 0, 0.85, 1])
for ext in ["png", "pdf"]:
    fig.savefig(os.path.join(APPEND_DIR,
                f"Sensitivity analysis_with_Baseline.{ext}"),
                dpi=300, bbox_inches="tight")
plt.show()
print("Appendix grid plot saved.")


# ===========================================================================
# 6.  PERCENTAGE-CHANGE BAR CHART — without bridge concept
# ===========================================================================
def _pct_change_table(scenarios_dict: dict,
                      baseline_ser: pd.Series,
                      target_lbls: list) -> pd.DataFrame:
    """
    Compute a (scenario × concept) DataFrame of % change relative to baseline.
    Infinite and NaN values are replaced with 0.
    """
    final_tab = pd.DataFrame()
    pct_tab   = pd.DataFrame()
    for name, state in scenarios_dict.items():
        df_res        = simulate_fcm(state, weight_matrix)
        final_vals    = df_res.iloc[-1][target_lbls]
        final_tab.loc[name, target_lbls] = final_vals.values
        pct           = ((final_vals - baseline_ser) / baseline_ser * 100
                         ).replace([np.inf, -np.inf], 0).fillna(0)
        pct_tab.loc[name, target_lbls] = pct.values
    return final_tab, pct_tab


# Key outcome concepts for bar chart
bar_target_ids    = ["C1", "C2", "C3", "C7", "C8"]
bar_target_labels = [concept_label_map[cid] for cid in bar_target_ids]
baseline_final    = baseline_df.iloc[-1][bar_target_labels]

bar_scenarios = {
    "Fb":               apply_intervention("Fb",         last_values),
    "Tb":               apply_intervention("Tb",         last_values),
    "FTb (Flood-strong)": apply_intervention("FTb_strong", last_values),
    "FTb (Flood-medium)": apply_intervention("FTb_medium", last_values),
    "FTb (Flood-weak)":   apply_intervention("FTb_weak",   last_values),
}

final_table, pct_table = _pct_change_table(bar_scenarios, baseline_final, bar_target_labels)

# Save tables
final_table.to_csv(os.path.join(RESULTS_DIR, "final_activation_values_table.csv"))
pct_table.to_csv(  os.path.join(RESULTS_DIR, "percentage_change_table.csv"))
print("Percentage-change tables saved.")


def lighten_color(color: str, amount: float) -> tuple:
    """Return a lightened version of `color` by mixing with white."""
    c = np.array(mcolors.to_rgb(color))
    return tuple(c + (1 - c) * amount)


scenario_colors = {
    "Fb":                 "#a8a6a3",
    "Tb":                 "#d6c165",
    "FTb (Flood-strong)": "#fa5f3c",
    "FTb (Flood-medium)": lighten_color("#fa5f3c", 0.35),
    "FTb (Flood-weak)":   lighten_color("#fa5f3c", 0.65),
}


def _bar_plot(pct_df: pd.DataFrame, colors: dict,
              target_lbls: list, filename_stem: str) -> None:
    """Render and save the grouped bar chart of % change from baseline."""
    fig, ax = plt.subplots(figsize=(14, 8))
    x           = np.arange(len(target_lbls))
    n_scen      = len(pct_df)
    total_width = 0.8
    bar_width   = total_width / n_scen
    offsets     = np.linspace(
        -total_width / 2 + bar_width / 2,
         total_width / 2 - bar_width / 2,
        n_scen
    )
    for i, scen in enumerate(pct_df.index):
        ax.bar(x + offsets[i], pct_df.loc[scen], width=bar_width,
               color=colors[scen], label=scen, edgecolor="black")

    ax.set_xticks(x); ax.set_xticklabels(target_lbls, rotation=30)
    ax.set_xlabel("Concepts"); ax.set_ylabel("% Change from Baseline")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2); ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(direction="out", length=5, width=1)
    ax.legend(title="Scenarios", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(False)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(RESULTS_DIR, f"{filename_stem}.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.show()


_bar_plot(pct_table, scenario_colors, bar_target_labels, "Percentage_Change_Comparison")
print("Main sensitivity bar chart saved.")


# ===========================================================================
# 7.  PERCENTAGE-CHANGE BAR CHART — WITH bridge concept C39 activated
# ===========================================================================
bar_scenarios_bridge = {
    "Fb":                 apply_intervention("Fb",         last_values, activate_bridge=True),
    "Tb":                 apply_intervention("Tb",         last_values, activate_bridge=True),
    "FTb (Flood-strong)": apply_intervention("FTb_strong", last_values, activate_bridge=True),
    "FTb (Flood-medium)": apply_intervention("FTb_medium", last_values, activate_bridge=True),
    "FTb (Flood-weak)":   apply_intervention("FTb_weak",   last_values, activate_bridge=True),
}

_, pct_table_bridge = _pct_change_table(
    bar_scenarios_bridge, baseline_final, bar_target_labels
)

_bar_plot(pct_table_bridge, scenario_colors, bar_target_labels,
          "Comparison_with Usage_Baseline")
print("Bridge-concept sensitivity bar chart saved.")
print("\nStep 2 complete. All outputs written to Results/ and Appendices/.")
