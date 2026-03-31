"""
01_scenario_analysis.py
=======================
Step 1 of the FCM analysis pipeline for:
  "Fuzzy Cognitive Mapping as a Dynamic Decision-Support Tool for Heritage Site
   Resilience: Evidence from Integrated Flood and Tourism Management at Qalhat, Oman"

What this script does
---------------------
1. Loads the FCM weight matrix, concept labels, and baseline initial activation values.
2. Runs the baseline simulation (Kosko inference + sigmoid transfer, λ=1, ε<0.001).
3. Plots and saves the baseline activation trajectory for all concepts.
4. Plots and saves the baseline trajectory for the five key outcome concepts only.
5. Runs five intervention scenarios (Fa, Fb, Ta, Tb, FTb) as single-shot perturbations.
6. Plots line plots per scenario group (flooding / tourism / combined).
7. Saves a comparison CSV and a summary line plot for the FTb scenario.

Outputs (written to Results/ and Appendices/)
---------------------------------------------
  Appendices/fcm_simulation_results.csv
  Appendices/fcm_simulation_plot.png / .pdf
  Appendices/fcm_baseline_simulation_results.csv
  Appendices/baseline activation levels for all concepts_plot.png / .pdf
  Results/Selected_Concept_Activation.png / .pdf
  Results/Flooding_Scenarios.png / .pdf
  Results/Flood_without_management_response_Fa.csv
  Results/Flood_with_management_response_Fb.csv
  Results/Tourism_Scenarios.png / .pdf
  Results/Tourism_without_management_response_Ta.csv
  Results/Tourism_with_management_response_Tb.csv
  Results/FTb_Scenario_Concept_Levels.png / .pdf
  Results/comparison_table.csv
  Results/The level of key system components in the FTb scenario.png / .pdf

Data paths (Input data)
-----------------------------------------
  DATA_DIR/Aggregated participatory FCM.csv   — FCM adjacency/weight matrix
  DATA_DIR/initial values.csv                — baseline activation values
  DATA_DIR/Concept labels.csv              — concept ID → label mapping
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from fcmpy import FcmSimulator

# ===========================================================================
# 0.  CONFIGURATION — edit paths here if your folder layout differs
# ===========================================================================
DATA_DIR    = "Data"
RESULTS_DIR = "Results"
APPEND_DIR  = "Appendices"

MATRIX_PATH  = os.path.join(DATA_DIR,   "Aggregated participatory FCM.csv")
INIT_PATH    = os.path.join(DATA_DIR,   "initial values.csv")
LABELS_PATH  = os.path.join(DATA_DIR, "Concept labels.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(APPEND_DIR,  exist_ok=True)

# Publication-ready matplotlib style
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        12,
    "axes.labelsize":   13,
    "axes.titlesize":   14,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  10,
    "lines.linewidth":  2,
    "lines.markersize": 6,
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


def get_initial_values():
    """Return a dict {label: initial_activation} for the baseline state."""
    df = pd.read_csv(INIT_PATH)
    if "Concept ID" not in df.columns or "Initial Value" not in df.columns:
        raise ValueError("initial values CSV must have 'Concept ID' and 'Initial Value' columns.")
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
    Run an FCM simulation using Kosko inference and a sigmoid transfer function.

    Inactive concepts (activation == 0) have their incoming and outgoing edges
    zeroed so they do not influence the dynamics.

    Parameters
    ----------
    initial_state : dict
        Mapping of concept label → initial activation value [0, 1].
    wm : np.ndarray
        FCM weight matrix (square, float).
    iterations : int
        Maximum number of update iterations (default 20).

    Returns
    -------
    pd.DataFrame
        Activation values per iteration, columns = concept labels.
    """
    mod_matrix = wm.copy()
    inactive   = [c for c, v in initial_state.items() if float(v) == 0.0]
    for c in inactive:
        if c in concept_labels:
            idx = concept_labels.index(c)
            mod_matrix[:, idx] = 0   # zero all incoming edges
            mod_matrix[idx, :] = 0   # zero all outgoing edges

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
# 3.  BASELINE SIMULATION
# ===========================================================================
init_state   = get_initial_values()
baseline_df  = simulate_fcm(init_state, weight_matrix)
last_values  = baseline_df.iloc[-1].to_dict()   # steady-state used as start for scenarios

# --- save full results CSV ---
baseline_df.to_csv(os.path.join(APPEND_DIR, "fcm_simulation_results.csv"), index=True)
baseline_df.to_csv(os.path.join(APPEND_DIR, "fcm_baseline_simulation_results.csv"), index=True)
print("Baseline simulation complete.")

# --- plot all concepts ---
fig, ax = plt.subplots(figsize=(14, 8))
for concept in concept_labels:
    ax.plot(baseline_df.index, baseline_df[concept], label=concept)
ax.set_xlabel("Simulation Steps")
ax.set_ylabel("Concept Activation")
ax.set_xticks(range(5)); ax.set_xticklabels(range(1, 6))
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=4,
          frameon=True, edgecolor="black")
plt.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ["png", "pdf"]:
    fig.savefig(os.path.join(APPEND_DIR,
                f"baseline activation levels for all concepts_plot.{ext}"),
                dpi=300, bbox_inches="tight")
plt.show()

# --- plot selected key-outcome concepts only ---
# C1=OUV  C2=Physical Fabric  C3=Landscape  C7=Budget  C8=Tourism
target_ids     = ["C1", "C2", "C3", "C7", "C8"]
target_labels  = [concept_label_map.get(c, c) for c in target_ids]
concept_colors = {
    "C1": "gray", "C2": "#C4A484", "C3": "green", "C7": "purple", "C8": "red"
}

fig, ax = plt.subplots(figsize=(10, 5))
x_vals = np.arange(1, len(baseline_df) + 1)
for cid, lbl in zip(target_ids, target_labels):
    if lbl in baseline_df.columns:
        ax.plot(x_vals, baseline_df[lbl].values, label=lbl,
                color=concept_colors.get(cid, "black"), marker="o")
ax.set_xlabel("Iterations"); ax.set_ylabel("Concept Activation Level")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.2); ax.spines["bottom"].set_linewidth(1.2)
ax.set_xticks(x_vals); ax.tick_params(direction="out", length=5, width=1)
ax.legend(loc="best", frameon=True, edgecolor="black")
ax.grid(False)
plt.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(os.path.join(RESULTS_DIR, f"Selected_Concept_Activation.{ext}"),
                dpi=300, bbox_inches="tight")
plt.show()
print("Baseline plots saved.")


# ===========================================================================
# 4.  INTERVENTION SCENARIOS
# ===========================================================================
def apply_intervention(base_state: dict, name: str) -> dict:
    """
    Return a copy of base_state with the activation levels of the
    intervention input concepts set to the scenario-specific values.

    Scenario definitions (see Table 1 in the paper):
      Fa  — severe flood, no management
      Fb  — severe flood + active management response
      Ta  — maximum tourism, no management
      Tb  — maximum tourism + visitor-management response
      FTb — combined flood (full) + tourism (partial) + full management
    """
    s = base_state.copy()
    lm = concept_label_map   # short alias

    if name == "Fa":
        s.update({lm["C15"]: 1.0, lm["C11"]: 1.0})

    elif name == "Fb":
        s.update({lm["C15"]: 1.0, lm["C11"]: 1.0,
                  lm["C21"]: 0.1, lm["C27"]: 1.0, lm["C28"]: 0.1})

    elif name == "Ta":
        s.update({lm["C40"]: 1.0})

    elif name == "Tb":
        s.update({lm["C40"]: 1.0, lm["C21"]: 0.1, lm["C28"]: 0.1})

    elif name == "FTb":
        # Tourism set to 0.5 — partial capacity during flooding
        s.update({lm["C15"]: 1.0, lm["C11"]: 1.0, lm["C40"]: 0.5,
                  lm["C21"]: 0.1, lm["C27"]: 1.0, lm["C28"]: 0.1})
    return s


# ===========================================================================
# 5.  FLOODING SCENARIOS (Fa and Fb) — line plots
# ===========================================================================
flood_scenarios = {
    "Flood without management response (Fa)": apply_intervention(last_values, "Fa"),
    "Flood with management response (Fb)":    apply_intervention(last_values, "Fb"),
}
flood_target_ids    = ["C1", "C2", "C3", "C8", "C15", "C7"]
flood_target_labels = [concept_label_map.get(c, c) for c in flood_target_ids]
flood_colors        = {**concept_colors, "C15": "blue"}

flood_dfs = []
for name, state in flood_scenarios.items():
    df = simulate_fcm(state, weight_matrix)
    safe = name.replace(" ", "_").replace("(", "").replace(")", "")
    df.to_csv(os.path.join(RESULTS_DIR, f"{safe}.csv"), index=False)
    flood_dfs.append(df)

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
for ax, df, name in zip(axes, flood_dfs, flood_scenarios.keys()):
    x = range(1, len(df) + 1)
    for cid, lbl in zip(flood_target_ids, flood_target_labels):
        if cid in df.columns:
            ax.plot(x, df[cid], label=lbl, marker="o",
                    color=flood_colors.get(cid, "black"))
    ax.set_title(name); ax.set_xlabel("Iterations")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2); ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(direction="out", length=5, width=1); ax.grid(False)
axes[0].set_ylabel("Concept Level")
handles, lbls = axes[0].get_legend_handles_labels()
fig.legend(handles, lbls, loc="upper center", ncol=len(flood_target_ids),
           frameon=True, edgecolor="black")
plt.tight_layout(rect=[0, 0, 1, 0.92])
for ext in ["png", "pdf"]:
    fig.savefig(os.path.join(RESULTS_DIR, f"Flooding_Scenarios.{ext}"),
                dpi=300, bbox_inches="tight")
plt.show()
print("Flooding scenario plots saved.")


# ===========================================================================
# 6.  TOURISM SCENARIOS (Ta and Tb) — line plots
# ===========================================================================
tourism_scenarios = {
    "Tourism without management response (Ta)": apply_intervention(last_values, "Ta"),
    "Tourism with management response (Tb)":    apply_intervention(last_values, "Tb"),
}
tourism_target_ids    = ["C1", "C2", "C3", "C40", "C7", "C8"]
tourism_target_labels = [concept_label_map.get(c, c) for c in tourism_target_ids]
tourism_colors        = {**concept_colors, "C40": "orange"}

tourism_dfs = []
for name, state in tourism_scenarios.items():
    df = simulate_fcm(state, weight_matrix)
    safe = name.replace(" ", "_").replace("(", "").replace(")", "")
    df.to_csv(os.path.join(RESULTS_DIR, f"{safe}.csv"), index=False)
    tourism_dfs.append(df)

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
for ax, df, name in zip(axes, tourism_dfs, tourism_scenarios.keys()):
    x = range(1, len(df) + 1)
    for cid, lbl in zip(tourism_target_ids, tourism_target_labels):
        if cid in df.columns:
            ax.plot(x, df[cid], label=lbl, marker="o",
                    color=tourism_colors.get(cid, "black"))
    ax.set_title(name); ax.set_xlabel("Iterations")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2); ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(direction="out", length=5, width=1); ax.grid(False)
axes[0].set_ylabel("Concept Level")
handles, lbls = axes[0].get_legend_handles_labels()
fig.legend(handles, lbls, loc="upper center", ncol=len(tourism_target_ids),
           frameon=True, edgecolor="black")
plt.tight_layout(rect=[0, 0, 1, 0.92])
for ext in ["png", "pdf"]:
    fig.savefig(os.path.join(RESULTS_DIR, f"Tourism_Scenarios.{ext}"),
                dpi=300, bbox_inches="tight")
plt.show()
print("Tourism scenario plots saved.")


# ===========================================================================
# 7.  COMBINED SCENARIO (FTb) + COMPARISON TABLE
# ===========================================================================
all_scenarios = {
    "Baseline":                            last_values,
    "Flood with Improvement (Fb)":         apply_intervention(last_values, "Fb"),
    "Tourism with Improvement (Tb)":       apply_intervention(last_values, "Tb"),
    "Flood + Tourism with Improvement (FTb)": apply_intervention(last_values, "FTb"),
}
ftb_target_ids    = ["C1", "C2", "C3", "C7", "C15", "C40", "C8"]
ftb_target_labels = [concept_label_map[cid] for cid in ftb_target_ids]
ftb_colors        = {**concept_colors, "C15": "blue", "C40": "orange"}

scenario_results = {}
for name, state in all_scenarios.items():
    scenario_results[name] = simulate_fcm(state, weight_matrix)

# Comparison CSV — final activation of target concepts across all scenarios
comparison_data = {
    scen: [df.iloc[-1][lbl] for lbl in ftb_target_labels]
    for scen, df in scenario_results.items()
}
comparison_df = pd.DataFrame(comparison_data, index=ftb_target_labels)
comparison_df.index.name = "Concept Label"
comparison_df.to_csv(os.path.join(RESULTS_DIR, "comparison_table.csv"))
print("Comparison table saved.")

# FTb line plot
ftb_name = "Flood + Tourism with Improvement (FTb)"
ftb_df   = scenario_results[ftb_name]
x_vals   = range(1, len(ftb_df) + 1)

fig, ax = plt.subplots(figsize=(12, 8))
for cid in ftb_target_ids:
    lbl = concept_label_map[cid]
    ax.plot(x_vals, ftb_df[lbl], label=lbl,
            color=ftb_colors.get(cid, "black"), marker="o")
ax.set_xlabel("Iterations"); ax.set_ylabel("Concept Level")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.2); ax.spines["bottom"].set_linewidth(1.2)
ax.set_xticks(x_vals); ax.tick_params(direction="out", length=5, width=1)
ax.legend(title="Concepts", loc="upper right", frameon=True)
ax.grid(False)
plt.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(
        os.path.join(RESULTS_DIR,
                     f"The level of key system components in the FTb scenario.{ext}"),
        dpi=300, bbox_inches="tight"
    )
plt.show()
print("FTb scenario plot saved.")
print("\nStep 1 complete. All outputs written to Results/ and Appendices/.")
