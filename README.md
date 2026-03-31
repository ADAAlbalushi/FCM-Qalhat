# Fuzzy Cognitive Mapping for Heritage Site Resilience

### *Dynamic Decision-Support for Integrated Flood and Tourism Management at Qalhat, Oman*

[!\[Python](https://img.shields.io/badge/Python-3.8%2525252B-blue)](https://www.python.org/)
[!\[License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[!\[Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[!\[FCMpy](https://img.shields.io/badge/FCMpy-latest-lightgrey)](https://pypi.org/project/fcmpy/)

\---

## Overview

This repository contains the full reproducible code and data for the research article:

*"Fuzzy Cognitive Mapping as a Dynamic Decision-Support Tool for Heritage Site Resilience: Evidence from Integrated Flood and Tourism Management at Qalhat, Oman"*

The study uses **Fuzzy Cognitive Maps (FCMs)** to model the socio-ecological system of Qalhat — a UNESCO World Heritage site — and simulate how flood events and overtourism affect the site's Outstanding Universal Value (OUV). Five management scenarios are tested, followed by sensitivity analysis and causal tracing to identify structural bottlenecks and bridge concepts within the system.

\---

## Repository Structure

```

FCM-Qalhat/

│

├── Data/

│   ├── Aggregated participatory FCM.csv   # Adjacency matrix representing causal relationships between concepts.


│   ├── initial values.csv                 # Initial activation levels for each concept node, provided by stakeholders.

│   └── Concept labels.csv                 # Full concept names and IDs (C1–C50).
│

│

├── Code/

│   ├── 01\\\\\\\_scenario\\\\\\\_analysis.py                

│   ├── 02\\\\\\\_sensitivity\\\\\\\_analysis.py

│   └── 03\\\\\\\_causal\\\\\\\_tracing.py

│

├── Results/                             # Figures and CSV outputs referenced in the paper

├── Appendices/                          # Supplementary outputs not shown in main paper

├── requirements.txt

└── README.md
```

Note on paths: All three scripts read data from a single Data/ folder. The paths are configured in the # 0. CONFIGURATION block at the top of each script — DATA\_DIR, RESULTS\_DIR, and APPEND\_DIR.



\## Data Availability



All data required to reproduce the results are included in the `Data/` folder of this repository.

No external data sources are required.

\---

## Methodology Summary

The analysis proceeds in three steps, each corresponding to one script:

### Step 1 — Scenario Simulations (`01_scenario_analysis.py`)

Baseline FCM behaviour is first established by running the model to convergence (threshold ε < 0.001, max 20 iterations) using Kosko's inference rule with a sigmoid transfer function (λ = 1.0). Five hypothetical scenarios are then simulated as single-shot interventions:

|ID|Scenario|Key intervention inputs|
|-|-|-|
|**Fa**|Flood (no management)|Flooding = 1, Precipitation = 1|
|**Fb**|Flood + management|Above + Protection/Mitigation = 1, Conservation = 0.1, Restoration = 0.1|
|**Ta**|Overtourism (no management)|Tourism = 1|
|**Tb**|Overtourism + management|Tourism = 1 + Conservation = 0.1, Restoration = 0.1|
|**FTb**|Combined flood \& tourism + management|Flooding = 1, Tourism = 0.5 + full management response|

Each scenario is evaluated by computing the percentage change in key outcome concepts (OUV, Physical Fabric, Site Landscape, Budget) relative to the baseline.

### Step 2 — Sensitivity Analysis (`02_sensitivity_analysis.py`)

Structural robustness is tested by re-running scenarios Fb, Tb, and FTb under three intervention intensity levels:

* **Weak:** 0.25 — **Medium:** 0.50 — **Strong:** 1.0

This confirms that observed outcomes result from the causal pathways encoded in the FCM rather than simply from input magnitude. An additional comparison activates a key **bridge concept** (Usage Activity, C39) to evaluate its role as a structural lever.

### Step 3 — Causal Tracing \& Bridge Concepts (`03_causal_tracing.py`)

Using Kosko's fuzzy causal algebra, all indirect causal paths from **Flooding → Tourism** are traced. The total causal effect is computed as the maximum over all indirect path effects, where each path's strength equals its weakest edge (min–max logic):

* **Indirect effect along path *l*:** `I_l(Ci, Cj) = min{ e(Cp, Cp+1) : (p, p+1) ∈ l }`
* **Total causal effect:** `T(Ci, Cj) = max{ I_l(Ci, Cj) : 1 ≤ l ≤ m }`

Bridge concepts — intermediate nodes on dominant causal paths — are identified and visualised as a directed network. Dormant bridge concepts act as structural barriers to intervention propagation.

\---

## Getting Started

### Prerequisites

|Requirement|Version|
|-|-|
|Python|≥ 3.8|
|Operating System|Windows / macOS / Linux (tested on Windows 11 x64)|
|RAM|≥ 4 GB recommended|

### Installation

1. **Clone the repository:**

```bash
   git clone https://github.com/<ADAAlbalushi>/FCM-Qalhat.git
   cd FCM-Qalhat
   ```

2. **Create a virtual environment (recommended):**

```bash
   python -m venv venv
   source venv/bin/activate       # macOS/Linux
   venv\Scripts\activate          # Windows
   ```

3. **Install dependencies:**

```bash
   pip install -r requirements.txt
   ```

Or install the key packages manually:

```bash
   pip install fcmpy numpy pandas matplotlib networkx scikit-fuzzy tqdm openpyxl
   ```

> FCMpy requires Python ≥ 3.8 and is available on PyPI: `pip install fcmpy`

### Running the Scripts

Run the three scripts in order from your terminal:

```bash
python Code/01_scenario_analysis.py
python Code/02_sensitivity_analysis.py
python Code/03_causal_tracing.py
```

|Order|Script|Description|
|-|-|-|
|1|`01_scenario_analysis.py`|Baseline + 5 scenario simulations|
|2|`02_sensitivity_analysis.py`|Sensitivity testing (weak / medium / strong)|
|3|`03_causal_tracing.py`|Causal tracing and bridge concept visualisation|

Outputs (figures and CSV tables) are saved automatically to the `Results/` and `Appendices/` folders.

\---

## Input Data

|File|Description|
|-|-|
|`Aggregated participatory FCM.csv`|Square adjacency matrix of FCM edge weights (values in \[−1, 1]), rows = sources, columns = targets|
|`initial values.csv`|Initial activation level (value in \[0, 1]) for each concept node to excute the baseline simulation|
|`Concept labels.csv`|Mapping of concept IDs (C1, C2, …) to full concept names used in the paper|

\---

## Key Outputs

After running all scripts, the following outputs are generated:

**Results/ folder:**

* Convergence plots and baseline activation table
* Scenario comparison bar charts (% change vs. baseline)
* Line plots of concept activation trajectories per scenario
* `comparison_table.csv` — tabular summary of all scenario outcomes
* `final_activation_values_table.csv` and `percentage_change_table.csv` — sensitivity results
* `Flooding_Tourism_Network.png / .pdf` — publication figure of causal pathways

**Appendices/ folder:**

* Full baseline activation plot (all concepts)
* Sensitivity analysis grid plots (Baseline + 5 scenarios)
* Intermediate causal network figures

\---

## Key Dependencies

```
fcmpy>=1.1.3
numpy>=1.18.2
pandas>=1.0.3
matplotlib>=3.3.0
networkx>=2.5
scikit-fuzzy>=0.4.2
tqdm>=4.50.2
openpyxl>=3.0.0
```

See `requirements.txt` for the full pinned list.

\---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{ADAAlbalushi2026FCMQalhat,
  title   = title: "Fuzzy Cognitive Mapping as a Dynamic Decision-Support Tool for Socio-Ecological System Resilience: Evidence from Integrated Flood and Tourism Management at Qalhat, Oman"

authors:

&#x20;Amira D. Al-Balushi1,2\*,    Frank O. Ostermann1,   Ellen-Wien Augustijn1,    Raúl Zurita-Milla1

1 Geo-information Processing (GIP) Department, Geo-Information Science and Earth Observation (ITC) Faculty, University of Twente, Enschede, the Netherlands

2 Geography Department, The College of Arts and Social Sciences, Sultan Qaboos University, Muscat, Oman

date-released: 2026-04-01

version: "1.0.0"

doi: "https://orcid.org/0009-0002-8188-6271"

repository-code: "https://github.com/ADAAlbalushi/FCM-Qalhat"

license: "MIT"



```

\---

## References

* Kosko, B. (1986). Fuzzy cognitive maps. *International Journal of Man-Machine Studies*, 24(1), 65–75.
* Mkhitaryan, M., Giabbanelli, P. J., de Vries, N., \& Crutzen, R. (2022). FCMpy: A Python module for constructing and analyzing fuzzy cognitive maps. *PeerJ Computer Science*, 8, e1078.
* Farahani, R. Z. (2022). *Fuzzy Cognitive Maps*. Springer.
* Nápoles, G., \& Giabbanelli, P. (2024). Fuzzy cognitive maps in the era of large language models. *Applied Soft Computing*.

\---

## License

This project is licensed under the [MIT License](LICENSE.txt). You are free to use, modify, and distribute this code with attribution.

\---

## Contact

For questions about the code or methodology, please open an issue in this repository or contact the corresponding author via this email:
      \*Corresponding author E-mail: Amira D. Al-Balushi. a.d.a.albalushi@utwente.nl 

