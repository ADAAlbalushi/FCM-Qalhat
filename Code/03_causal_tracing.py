"""
03_causal_tracing.py
====================
Step 3 of the FCM analysis pipeline for:
  "Fuzzy Cognitive Mapping as a Dynamic Decision-Support Tool for Heritage Site
   Resilience: Evidence from Integrated Flood and Tourism Management at Qalhat, Oman"

What this script does
---------------------
1. Loads the FCM data and re-runs the baseline simulation to obtain
   final concept activation values.
2. Builds a directed NetworkX graph from the FCM weight matrix.
3. Traces ALL simple paths from Flooding (C15) → Tourism (C40) and
   from Tourism (C40) → Flooding (C15) using nx.all_simple_paths.
4. Applies Kosko's fuzzy causal algebra:
     • Indirect effect of a path  = min{ |edge weight| } along that path
     • Total causal effect         = max{ indirect effect } across all paths
5. Identifies bridge concepts — intermediate nodes on the causal paths
   between C15 and C40.
6. Produces and saves three publication-quality network visualisations:
     a. Simple node-colour graph (active vs inactive) with spring layout
     b. Weighted-edge graph with dotted inactive-source edges
     c. Final publication figure with manually tuned node positions,
        paper colour scheme, and edge-weight labels

Outputs (written to Results/)
------------------------------
  Results/Flooding_Tourism_Network.png / .pdf  (publication figure)

Intermediate figures (written to Appendices/)
---------------------------------------------
  Appendices/Flooding_to_Tourism.png
  Appendices/Flooding_to_Tourism_weighted.png

Data paths (Input data)
-----------------------------------------
  DATA_DIR/Aggregated participatory FCM.csv
  DATA_DIR/initial values.csv
  DATA_DIR/Concept labels.csv
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    "axes.titlesize":  14,
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
    df = pd.read_csv(INIT_PATH)
    return {
        concept_label_map.get(cid, cid):
            float(df.loc[df["Concept ID"] == cid, "Initial Value"].values[0])
            if cid in df["Concept ID"].values else 0.0
        for cid in concept_ids
    }


init_state = get_initial_values()

# ===========================================================================
# 2.  MASK INACTIVE CONCEPTS IN THE WEIGHT MATRIX
# ===========================================================================
inactive = [c for c, v in init_state.items() if float(v) == 0.0]
wm_masked = weight_matrix.copy()
for c in inactive:
    idx = concept_labels.index(c)
    wm_masked[:, idx] = 0
    wm_masked[idx, :] = 0

# ===========================================================================
# 3.  BASELINE SIMULATION
# ===========================================================================
sim    = FcmSimulator()
result = sim.simulate(
    initial_state=init_state,
    weight_matrix=wm_masked,
    transfer="sigmoid",
    inference="kosko",
    thresh=0.001,
    iterations=20,
    l=1,
)
results_df  = pd.DataFrame(result, columns=concept_labels)
final_vals  = results_df.iloc[-1].to_dict()
print("Baseline simulation complete.")

# ===========================================================================
# 4.  BUILD NETWORKX GRAPH FROM FCM MATRIX
# ===========================================================================
G = nx.DiGraph()
G.add_nodes_from(concept_labels)
for i, src_id in enumerate(concept_ids):
    for j, tgt_id in enumerate(concept_ids):
        w = fcm_data.iloc[i, j]
        if w != 0:
            G.add_edge(concept_label_map[src_id], concept_label_map[tgt_id], weight=float(w))

nx.set_node_attributes(G, final_vals, "activation")

# ===========================================================================
# 5.  CAUSAL PATH TRACING
# ===========================================================================

def trace_paths(source_id: str, target_id: str) -> list:
    """
    Return all simple paths from source to target in the FCM graph.

    Parameters
    ----------
    source_id, target_id : str
        Concept IDs (e.g. 'C15', 'C40').

    Returns
    -------
    list of lists  — each inner list is a sequence of concept labels.
    """
    src = concept_label_map[source_id]
    tgt = concept_label_map[target_id]
    try:
        return list(nx.all_simple_paths(G, source=src, target=tgt))
    except nx.NetworkXNoPath:
        return []


def compute_weakest_link_per_path(paths: list) -> dict:
    """
    Kosko's indirect effect for each path = min absolute edge weight.

    Parameters
    ----------
    paths : list of lists (node sequences)

    Returns
    -------
    dict  {tuple(path): indirect_effect_value}
    """
    path_effects = {}
    for p in paths:
        weights = [abs(G[p[k]][p[k + 1]]["weight"]) for k in range(len(p) - 1)]
        path_effects[tuple(p)] = min(weights) if weights else 0.0
    return path_effects


def compute_total_effect(path_effects: dict) -> float:
    """
    Kosko's total causal effect = max over all path indirect effects.

    Parameters
    ----------
    path_effects : dict  {path_tuple: indirect_effect}

    Returns
    -------
    float
    """
    return max(path_effects.values()) if path_effects else 0.0


def identify_bridge_concepts(paths: list, source_id: str, target_id: str) -> set:
    """
    Return labels of all intermediate nodes across all causal paths
    (excludes source and target themselves).
    """
    src_lbl = concept_label_map[source_id]
    tgt_lbl = concept_label_map[target_id]
    bridges = set()
    for p in paths:
        bridges.update(node for node in p
                        if node not in (src_lbl, tgt_lbl))
    return bridges


# --- C15 → C40 (Flooding → Tourism) ---
print("\nTracing paths: Flooding (C15) → Tourism (C40) …")
paths_c15_c40    = trace_paths("C15", "C40")
effects_c15_c40  = compute_weakest_link_per_path(paths_c15_c40)
total_c15_c40    = compute_total_effect(effects_c15_c40)
bridges_c15_c40  = identify_bridge_concepts(paths_c15_c40, "C15", "C40")

print(f"  Number of paths found: {len(paths_c15_c40)}")
print(f"  Total causal effect (Kosko max–min): {total_c15_c40:.3f}")
print("\n  Weakest-link value per path:")
for p, eff in sorted(effects_c15_c40.items(), key=lambda x: -x[1]):
    print(f"    {' → '.join(p)}  |  indirect effect = {eff:.3f}")
print("\n  Bridge concepts (C15 → C40):")
for bc in sorted(bridges_c15_c40):
    print(f"    {bc}  |  final activation = {final_vals.get(bc, 'N/A'):.3f}")

# --- C40 → C15 (Tourism → Flooding) — expected: no paths ---
print("\nTracing paths: Tourism (C40) → Flooding (C15) …")
paths_c40_c15   = trace_paths("C40", "C15")
effects_c40_c15 = compute_weakest_link_per_path(paths_c40_c15)
total_c40_c15   = compute_total_effect(effects_c40_c15)
print(f"  Number of paths found: {len(paths_c40_c15)}")
print(f"  Total causal effect: {total_c40_c15:.3f}")
if not paths_c40_c15:
    print("  → No causal path exists from Tourism to Flooding (asymmetric dependency confirmed).")


# ===========================================================================
# 6.  BRIDGE CONCEPT ACTIVATION TABLE (from FTb steady state)
# ===========================================================================
# These values are taken from the FTb scenario simulation (see script 01/02).
# They are hard-coded here to allow this script to run independently.
# Update these values if you re-run the FTb scenario with different settings.
ftb_bridge_activations = {
    "LA":           0.421,
    "OUV":          0.700,
    "Abandonment":  0.000,
    "Accessibility":0.200,
    "Erosion":      0.548,
    "Landscape":    0.425,
    "Preservation": 0.250,
    "MP":           0.637,
    "Monitoring":   0.404,
    "Stray animals":0.000,
    "US":           0.000,
    "ComV":         0.200,
    "Sustainability":0.000,
    "Phy-Fab":      0.422,
    "Con & Res":    0.250,
    "Budget":       0.503,
    "Usage":        0.000,
}

# ===========================================================================
# 7.  GRAPH DATA FOR VISUALISATION
# ===========================================================================
# Weighted edge list representing the dominant causal paths Flooding → Tourism
viz_edges = [
    ("Flooding",      "Landscape",     0.65),
    ("Landscape",     "OUV",           0.75),
    ("OUV",           "Tourism",       0.833),
    ("Flooding",      "Tourism",       0.75),
    ("Flooding",      "Erosion",       0.571),
    ("Erosion",       "Phy-Fab",       0.571),
    ("Phy-Fab",       "OUV",           0.833),
    ("Flooding",      "Accessibility", 0.75),
    ("Accessibility", "Con & Res",     0.75),
    ("Con & Res",     "Preservation",  0.75),
    ("Preservation",  "Budget",        0.5),
    ("Budget",        "Sustainability",0.25),
    ("Sustainability","OUV",           0.25),
    ("Preservation",  "Landscape",     0.75),
    ("Preservation",  "Sustainability",0.25),
    ("Preservation",  "Erosion",       0.5),
    ("Preservation",  "Phy-Fab",       0.75),
    ("Accessibility", "Budget",        0.75),
    ("Flooding",      "Phy-Fab",       0.833),
    ("Flooding",      "MP",            0.583),
    ("MP",            "Con & Res",     0.583),
    ("MP",            "Preservation",  0.583),
    ("Preservation",  "OUV",           0.75),
    ("Flooding",      "Abandonment",   0.583),
    ("Abandonment",   "Usage",         0.583),
    ("Usage",         "OUV",           0.583),
    ("Usage",         "Landscape",     0.5),
    ("ComV",          "Monitoring",    0.562),
    ("Monitoring",    "Preservation",  0.562),
    ("Budget",        "Con & Res",     0.75),
    ("Con & Res",     "Accessibility", 0.417),
    ("Abandonment",   "US",            0.583),
    ("US",            "Landscape",     0.583),
    ("Abandonment",   "Stray animals", 0.25),
    ("Stray animals", "Phy-Fab",       0.25),
    ("MP",            "LA",            0.25),
    ("LA",            "Usage",         0.25),
    ("Flooding",      "ComV",          0.562),
    ("Flooding",      "Sustainability",0.25),
]

# Build visualisation graph
Gv = nx.DiGraph()
for src, tgt, w in viz_edges:
    Gv.add_edge(src, tgt, weight=w)

for node, act in ftb_bridge_activations.items():
    if node in Gv.nodes:
        Gv.nodes[node]["activation"] = act
Gv.add_node("Flooding", activation=1.0)
Gv.add_node("Tourism",  activation=0.8)

# Manually tuned positions (matches Fig. in paper)
POS = {
    "Flooding":      (-1.5,  2.5),
    "ComV":          (-2.5,  2.0),
    "Monitoring":    (-2.5,  1.5),
    "Abandonment":   (-2.5,  0.5),
    "Stray animals": (-3.0,  0.0),
    "US":            (-2.0,  0.0),
    "Usage":         (-1.5,  0.0),
    "LA":            (-1.0,  0.5),
    "MP":            (-1.0,  1.5),
    "Accessibility": ( 0.0,  2.5),
    "Budget":        ( 0.5,  2.0),
    "Con & Res":     ( 1.0,  2.5),
    "Preservation":  ( 1.5,  2.0),
    "Erosion":       ( 2.0,  1.5),
    "Phy-Fab":       ( 2.5,  1.0),
    "Landscape":     ( 2.0,  0.5),
    "OUV":           ( 1.5,  0.0),
    "Sustainability":( 0.5,  0.5),
    "Tourism":       ( 1.5, -0.5),
}


def _edge_style(G_in: nx.DiGraph):
    """Split edges into active-source and inactive-source lists."""
    active   = []
    inactive = []
    labels   = {}
    for u, v, data in G_in.edges(data=True):
        w = data.get("weight", 0)
        labels[(u, v)] = f"{w:.2f}"
        if G_in.nodes[u].get("activation", 0) == 0:
            inactive.append((u, v))
        else:
            active.append((u, v))
    return active, inactive, labels


# ===========================================================================
# 8.  FIGURE A — simple spring layout (appendix)
# ===========================================================================
pos_spring = nx.spring_layout(Gv, seed=42)
node_colors_simple = [
    "green" if Gv.nodes[n].get("activation", 0) > 0 else "red"
    for n in Gv.nodes
]
active_e, inactive_e, _ = _edge_style(Gv)

fig, ax = plt.subplots(figsize=(15, 12))
nx.draw_networkx(Gv, pos_spring, with_labels=True,
                 node_color=node_colors_simple, node_size=1500,
                 font_size=9, font_color="black",
                 arrows=True, arrowsize=15, edge_color="lightgray", ax=ax)
nx.draw_networkx_edges(Gv, pos_spring, edgelist=inactive_e,
                       edge_color="black", style="dashed", width=1.5, ax=ax)
ax.legend(handles=[
    mpatches.Patch(color="green", label="Active Node (>0)"),
    mpatches.Patch(color="red",   label="Inactive Node (=0)"),
    mpatches.Patch(edgecolor="black", facecolor="none",
                   linestyle="--", label="Weakest Path (inactive)"),
], loc="upper left")
ax.set_title("Flooding → Tourism: Causal Pathway Overview", fontsize=14)
ax.axis("off")
plt.tight_layout()
fig.savefig(os.path.join(APPEND_DIR, "Flooding_to_Tourism.png"),
            dpi=600, bbox_inches="tight")
plt.show()
print("Appendix figure A saved.")


# ===========================================================================
# 9.  FIGURE B — weighted edges, arrow placement (appendix)
# ===========================================================================
node_colors_b = [
    "green" if Gv.nodes[n].get("activation", 0) > 0 else "red"
    for n in Gv.nodes
]
active_e, inactive_e, elabels = _edge_style(Gv)

fig, ax = plt.subplots(figsize=(15, 12))
nx.draw_networkx_nodes(Gv, pos_spring, node_color=node_colors_b,
                       node_size=1500, alpha=0.9, ax=ax)
nx.draw_networkx_labels(Gv, pos_spring, font_size=9, font_color="black", ax=ax)
nx.draw_networkx_edges(Gv, pos_spring, edgelist=active_e,
                       edge_color="lightgray", arrows=True, arrowsize=15,
                       width=1.5, connectionstyle="arc3,rad=0",
                       min_source_margin=15, min_target_margin=15, ax=ax)
nx.draw_networkx_edges(Gv, pos_spring, edgelist=inactive_e,
                       edge_color="black", style="dotted", arrows=True,
                       arrowsize=15, width=1.5, connectionstyle="arc3,rad=0",
                       min_source_margin=15, min_target_margin=15, ax=ax)
nx.draw_networkx_edge_labels(Gv, pos_spring, edge_labels=elabels,
                             font_size=8, font_color="dimgray", ax=ax)
ax.legend(handles=[
    mpatches.Patch(color="green",  label="Active Node (>0)"),
    mpatches.Patch(color="red",    label="Inactive Node (=0)"),
    mpatches.Patch(edgecolor="black", facecolor="none",
                   linestyle=":", label="Inactive Source → Dotted Edge"),
], loc="upper left")
ax.set_title("Flooding → Tourism Pathways (Weighted Edges)", fontsize=14)
ax.axis("off")
plt.tight_layout()
fig.savefig(os.path.join(APPEND_DIR, "Flooding_to_Tourism_weighted.png"),
            dpi=600, bbox_inches="tight")
plt.show()
print("Appendix figure B saved.")


# ===========================================================================
# 10.  FIGURE C — publication figure (Results/)
#       Paper colour scheme, manually tuned positions, clean style
# ===========================================================================
node_colors_pub = [
    "#fa5f3c" if Gv.nodes[n].get("activation", 0) > 0 else "#a8a6a3"
    for n in Gv.nodes
]
active_e, inactive_e, elabels = _edge_style(Gv)

fig, ax = plt.subplots(figsize=(14, 10))
nx.draw_networkx_nodes(Gv, POS, node_color=node_colors_pub,
                       node_size=1400, edgecolors="black", ax=ax)
nx.draw_networkx_labels(Gv, POS, font_size=9, font_color="black", ax=ax)
nx.draw_networkx_edges(Gv, POS, edgelist=active_e,
                       edge_color="lightgray", arrows=True, arrowsize=14,
                       width=1.5, ax=ax)
nx.draw_networkx_edges(Gv, POS, edgelist=inactive_e,
                       edge_color="black", style="dotted", arrows=True,
                       arrowsize=14, width=1.2, ax=ax)
nx.draw_networkx_edge_labels(Gv, POS, edge_labels=elabels,
                             font_size=7, font_color="dimgray", ax=ax)
ax.legend(handles=[
    mpatches.Patch(color="#fa5f3c", label="Active Node (>0)"),
    mpatches.Patch(color="#a8a6a3", label="Inactive Node (=0)"),
    mpatches.Patch(edgecolor="black", facecolor="none",
                   linestyle=":", label="Inactive Connection"),
], loc="upper left")
ax.axis("off")
plt.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(os.path.join(RESULTS_DIR, f"Flooding_Tourism_Network.{ext}"),
                dpi=300, bbox_inches="tight")
plt.show()
print("Publication figure saved.")
print("\nStep 3 complete. All outputs written to Results/ and Appendices/.")
