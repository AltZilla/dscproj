# ⚡ Smart Home Energy Optimizer
### AI-Driven Grid Simulation & Privacy-Preserving Load Scheduling

> An end-to-end simulation platform that uses **Deep Learning**, **Federated Learning**, **Game Theory**, and **Evolutionary Algorithms** to minimize household electricity costs while safeguarding grid stability — all without exposing private consumer data.

---

## 📖 Table of Contents

1.  [Introduction & Motivation](#-introduction--motivation)
2.  [Key Features](#-key-features)
3.  [System Architecture](#️-system-architecture)
4.  [End-to-End Pipeline Flow](#-end-to-end-pipeline-flow)
5.  [Project Structure](#-project-structure)
6.  [Detailed Component Breakdown](#-detailed-component-breakdown)
    *   [Digital Twin Simulator](#1-digital-twin-simulator)
    *   [Data Preprocessing](#2-data-preprocessing)
    *   [VAE Feature Extraction](#3-vae-feature-extraction)
    *   [Federated GRU Load Forecasting](#4-federated-gru-load-forecasting)
    *   [Stackelberg Game Pricing](#5-stackelberg-game-pricing)
    *   [Differential Evolution Scheduling](#6-differential-evolution-scheduling)
    *   [Constraint Engine](#7-constraint-engine)
    *   [Interactive Dashboard](#8-interactive-dashboard)
7.  [Configuration](#️-configuration)
8.  [Getting Started](#-getting-started)
9.  [Usage Examples](#-usage-examples)
10. [Tech Stack](#-tech-stack)
11. [References](#-references)

---

## 📖 Introduction & Motivation

The transition to renewable energy and the rise of high-power home appliances (EV chargers, HVAC systems, water heaters) are placing unprecedented strain on electrical grids worldwide. Traditional static **Time-of-Use (TOU)** pricing is insufficient to actively manage and shave peak loads in real-time.

This project tackles the problem by building a **comprehensive, AI-driven smart grid and home energy simulation platform** that:

*   **Replaces rigid rule-based scheduling** with dynamic, model-driven optimization.
*   **Bridges utility-scale load forecasting** with individual smart home appliance scheduling.
*   **Preserves consumer privacy** via Federated Learning — raw IoT data never leaves the home.
*   **Finds economic equilibrium** via Game Theory between the grid operator and consumers.

The result: consumers save money, the grid avoids catastrophic peak overloading, and user comfort is maintained — all from a single unified pipeline.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Digital Twin Engine** | Simulates 50 smart homes with stateful appliances, Markov-chain occupancy, thermal physics, and TOU pricing at 15-minute resolution |
| **VAE Representation Learning** | Compresses noisy, high-dimensional IoT data into dense latent behavioral features |
| **Federated GRU Forecasting** | Predicts 24-hour energy demand across homes without centralizing raw data (`FedAvg` protocol) |
| **Stackelberg Game Pricing** | Computes hourly equilibrium electricity prices balancing utility revenue and grid safety |
| **DE Optimization** | Evolves appliance schedules using `DE/rand/1/bin` with 8 penalty-based fitness objectives |
| **Interactive Dashboard** | Plotly Dash UI with live floorplan, heatmap, occupancy controls, and one-click optimization |
| **Full Pipeline Orchestration** | Single `--run-all` command runs everything: simulate → preprocess → train → price → optimize |
| **YAML Configuration** | All hyperparameters, appliance specs, and constraints in one `config.yaml` file |

---

## 🏗️ System Architecture

```
                            ┌─────────────────────────────────┐
                            │         config/config.yaml      │
                            │  (all hyperparams & appliance   │
                            │   specs, centralized config)    │
                            └──────────────┬──────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE ORCHESTRATOR                            │
│                         src/pipeline.py                                 │
│                                                                         │
│  Stage 1          Stage 2           Stage 3          Stage 4            │
│  ┌──────────┐    ┌──────────────┐  ┌──────────┐    ┌───────────────┐   │
│  │ Digital  │───▶│Preprocessing │─▶│   VAE    │───▶│Federated GRU  │   │
│  │  Twin    │    │              │  │          │    │  (FedAvg)     │   │
│  └──────────┘    └──────────────┘  └──────────┘    └──────┬────────┘   │
│       │                                                    │            │
│       │           50 homes ×                   Latent       │            │
│       │           30 days ×                    Features     │ Predicted  │
│       │           96 slots/day                 (16-dim)     │ Demand     │
│       ▼                                                    ▼            │
│  data/raw/          data/processed/         outputs/models/             │
│  home_xxx.csv       train.csv               vae.pt                     │
│  all_homes.csv      val.csv                 federated_gru.pt           │
│                     test.csv                                           │
│                                                                         │
│  Stage 5                              Stage 6                           │
│  ┌──────────────────────┐            ┌──────────────────────────┐      │
│  │  Stackelberg Game    │───────────▶│  Differential Evolution  │      │
│  │  (Hourly Pricing)    │ Eq. Prices │  (Schedule Optimization) │      │
│  └──────────────────────┘            └──────────────────────────┘      │
│                                               │                         │
│                                               ▼                         │
│                                     Optimized 24h Schedule             │
│                                     + Cost Savings Report              │
└──────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
                               ┌──────────────────────┐
                               │   Interactive Dash    │
                               │     Dashboard         │
                               │  (app/dashboard.py)   │
                               └──────────────────────┘
```

---

## 🔄 End-to-End Pipeline Flow

Below is the complete data and control flow from simulation to optimized output. Each stage's output directly feeds the next.

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                                                                     │
  │  ① DIGITAL TWIN SIMULATION                                         │
  │  ─────────────────────────                                          │
  │  For each of 50 homes, for 30 simulated days:                       │
  │    • Markov Chain decides occupancy state (Home/Away/Sleeping)       │
  │    • Thermodynamic model drifts indoor temp toward outdoor           │
  │    • Each appliance rolls activation probability based on:           │
  │        - hour-of-day base schedule                                   │
  │        - occupancy state                                             │
  │        - indoor temperature (for HVAC)                               │
  │        - weekend boost (washer/dryer/dishwasher)                     │
  │    • TOU pricing tags each 15-min slot (Off-Peak/Shoulder/Peak)      │
  │                                                                     │
  │  OUTPUT → 144,000 records in data/raw/all_homes.csv                 │
  │           (50 homes × 30 days × 96 slots/day)                        │
  │                                                                     │
  └────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                                                                     │
  │  ② DATA PREPROCESSING                                              │
  │  ────────────────────                                                │
  │    • Interpolate missing values (linear + backfill/forwardfill)      │
  │    • Clip outliers using IQR method (1.5× factor)                    │
  │    • Engineer features:                                              │
  │        - Cyclical time encodings (sin/cos of hour & slot)            │
  │        - Rolling mean/std of total power (1-hour window)             │
  │        - Lag features (power at t-1 and t-4)                         │
  │    • MinMax normalize 22 feature columns                             │
  │    • Temporal train/val/test split (70/15/15) per home               │
  │                                                                     │
  │  OUTPUT → data/processed/train.csv, val.csv, test.csv               │
  │           22 normalized features per record                          │
  │                                                                     │
  └────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                                                                     │
  │  ③ VAE LATENT FEATURE EXTRACTION                                   │
  │  ──────────────────────────────                                      │
  │    • 22-dim input → 64-dim hidden → 16-dim latent space             │
  │    • Encoder: Linear→ReLU→BN→Linear→ReLU→BN → μ and σ²             │
  │    • Reparameterization trick: z = μ + ε·σ                           │
  │    • Decoder: mirrors encoder, reconstructs original features        │
  │    • Loss = MSE(x, x') + β·KL(q(z|x) ∥ p(z))   [β = 0.5]          │
  │    • Trains for 30 epochs with Adam optimizer (lr=0.001)             │
  │                                                                     │
  │  OUTPUT → 16-dim latent representations per timestep                │
  │           Saved model: outputs/models/vae.pt                        │
  │                                                                     │
  └────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                                                                     │
  │  ④ FEDERATED GRU LOAD FORECASTING                                  │
  │  ────────────────────────────────                                    │
  │    • Each home = one federated client with its own latent sequences  │
  │    • GRU architecture: 16-dim input → 64-dim hidden (2 layers)      │
  │                        → FC head → 1-dim load prediction             │
  │    • Sliding window: 16 timesteps (4 hours) → predict next load     │
  │                                                                     │
  │    • FedAvg Protocol (20 rounds):                                    │
  │        1. Server distributes global GRU model                        │
  │        2. 40% of homes selected per round as clients                 │
  │        3. Each client trains locally for 5 epochs on private data    │
  │        4. Clients upload only weight updates (NOT raw data)          │
  │        5. Server aggregates: w_global = Σ(nₖ/n_total)·wₖ            │
  │                                                                     │
  │  OUTPUT → Predicted 24h load profile for all homes                  │
  │           Metrics: MAE, RMSE, MAPE                                  │
  │           Saved model: outputs/models/federated_gru.pt              │
  │                                                                     │
  └────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                                                                     │
  │  ⑤ STACKELBERG GAME — DYNAMIC GRID PRICING                        │
  │  ──────────────────────────────────────────                          │
  │    • Leader (Grid Operator) sets price → Followers (Homes) respond   │
  │    • Runs independently for each of 24 hours                         │
  │                                                                     │
  │    Follower Best Response (per home):                                │
  │      U(c) = w·√c        ← concave utility (diminishing returns)     │
  │      c* = min((w/2p)², Cmax)   ← optimal consumption at price p     │
  │                                                                     │
  │    Leader Objective:                                                 │
  │      maximize: Revenue − Overload Penalty                            │
  │      Revenue = p · Σcᵢ*                                              │
  │      Penalty = ½·(Σcᵢ* − Capacity)²    if demand > capacity         │
  │                                                                     │
  │    Equilibrium Search:                                               │
  │      Grid search + refinement over price range [₹3 – ₹15/kWh]       │
  │      Finds p* that maximizes leader net profit                       │
  │                                                                     │
  │  OUTPUT → 24 hourly Stackelberg equilibrium prices                  │
  │           Blended SG prices fed to the optimizer                    │
  │                                                                     │
  └────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                                                                     │
  │  ⑥ DIFFERENTIAL EVOLUTION — SCHEDULE OPTIMIZATION                  │
  │  ────────────────────────────────────────────────                     │
  │    • Strategy: DE/rand/1/bin  (F=0.8, CR=0.9, pop=80–100)           │
  │    • Genome: 7 appliances × 24 hours = 168 binary values            │
  │    • Hard constraint: preserve exact ON-count per appliance          │
  │                                                                     │
  │    Model-Driven Fitness Function (8 soft penalty terms):             │
  │    ┌──────────────────────────────────────────────────────────┐      │
  │    │ 1. Energy Cost        — minimize Σ(Power × SG_Price)    │      │
  │    │ 2. Occupancy Comfort  — HVAC/Lighting ON when home      │      │
  │    │ 3. Temporal Proximity — quadratic penalty for moving     │      │
  │    │                         loads far from preferred time    │      │
  │    │ 4. Consecutiveness    — reward gapless sequential runs   │      │
  │    │ 5. TCL Duty Cycle     — penalize HVAC/WH running >2h    │      │
  │    │ 6. Peak Dispersal     — exponential spike penalty       │      │
  │    │ 7. Grid Capacity      — squared overload penalty        │      │
  │    │ 8. Forecast Alignment — nudge toward GRU predictions    │      │
  │    └──────────────────────────────────────────────────────────┘      │
  │                                                                     │
  │    Hard Constraints (enforced by src/constraints.py):               │
  │    • Washer must finish BEFORE dryer starts                          │
  │    • Min run durations (Washer: 45min, EV: 2h, HVAC: 1h, etc.)     │
  │    • Max hourly power ≤ 15kW, max daily energy ≤ 80kWh              │
  │                                                                     │
  │  OUTPUT → Optimized 7×24 binary schedule                            │
  │           Cost savings %, peak reduction %                          │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
dscproj/
│
├── main.py                        # CLI entry point (--run-all, --module, --dashboard)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── config/
│   └── config.yaml                # All hyperparameters and system configuration
│
├── src/                           # Core source modules
│   ├── __init__.py
│   ├── digital_twin.py            # Stage 1: Smart home simulation engine
│   ├── preprocessing.py           # Stage 2: Cleaning, feature engineering, splitting
│   ├── vae.py                     # Stage 3: Variational Autoencoder
│   ├── federated_gru.py           # Stage 4: Federated GRU load forecasting
│   ├── stackelberg_game.py        # Stage 5: Game-theoretic pricing
│   ├── differential_evolution.py  # Stage 6: DE schedule optimizer
│   ├── constraints.py             # Hard constraint checker for appliance schedules
│   └── pipeline.py                # End-to-end pipeline orchestrator
│
├── app/                           # Interactive dashboard
│   ├── dashboard.py               # Plotly Dash app (900+ lines)
│   └── assets/
│       └── styles.css             # Dashboard styling (dark theme)
│
├── data/
│   ├── raw/                       # Digital Twin output (50 CSVs + combined)
│   │   ├── home_000.csv ... home_049.csv
│   │   └── all_homes.csv          # ~144K records, 15MB
│   └── processed/                 # Preprocessed splits
│       ├── train.csv              # 70% (~30MB)
│       ├── val.csv                # 15% (~6.5MB)
│       └── test.csv               # 15% (~6.5MB)
│
├── outputs/
│   └── models/
│       ├── vae.pt                 # Trained VAE weights (~45KB)
│       └── federated_gru.pt       # Trained GRU weights (~174KB)
│
└── References/                    # Academic papers used
    ├── Big_Data-Intelligence_Analytics_for_Energy_Optimization.pdf
    ├── Multiagent_DRL.pdf
    ├── Spatiotemporal_Federated_Learning.pdf
    └── DSC_Project_DA1.docx / .txt
```

---

## 🔍 Detailed Component Breakdown

### 1. Digital Twin Simulator
**File:** `src/digital_twin.py` (460 lines)

The Digital Twin is the project's data engine — it synthesizes realistic smart home IoT data that the downstream ML pipeline trains on.

#### Architecture

| Class | Purpose |
|---|---|
| `Appliance` | Stateful model for each of 7 appliance types with activation probabilities, power profiles, and minimum/maximum run durations |
| `OccupancyModel` | 3-state Markov chain (`Home → Away → Sleeping`) with time-varying transition matrices for Night, Day, and Evening periods |
| `EnvironmentModel` | Thermodynamic simulator: outdoor temperature follows a sinusoidal daily curve (peak at 2 PM), indoor temperature drifts toward outdoor and is counteracted by HVAC cooling |
| `PricingModel` | Maps each hour to its TOU rate: Off-Peak (₹2/kWh, 12AM–6AM), Shoulder (₹5/kWh, 6AM–6PM), Peak (₹12/kWh, 6PM–12AM) |
| `SmartHome` | Composes all sub-models; runs one 15-min step producing a full IoT data record |
| `DigitalTwinSimulator` | Orchestrates N homes over D days and saves results to CSV |

#### Per-Step Simulation Logic
```
For each 15-minute slot:
  1. Markov Chain transitions occupancy state based on current hour
  2. Environment model updates outdoor/indoor temp and humidity
  3. For each appliance:
     a. Compute activation probability (base schedule × occupancy × temperature × weekend)
     b. Apply per-home behavioral scaling (0.7–1.3×)
     c. If activated, lock ON for random duration within [min, max] slots
     d. Record individual power consumption (kW)
  4. Compute TOU price for this hour
  5. Calculate slot cost: total_power × price × (15/60)
```

#### Simulated Appliances

| Appliance | Rated Power | Min Duration | Max Duration |
|---|---|---|---|
| HVAC | 3.5 kW | 1 hour (4 slots) | 6 hours (24 slots) |
| Washer | 0.5 kW | 45 min (3 slots) | 1.5h (6 slots) |
| Dryer | 2.0 kW | 45 min (3 slots) | 1.25h (5 slots) |
| Dishwasher | 1.8 kW | 45 min (3 slots) | 1.25h (5 slots) |
| EV Charger | 7.0 kW | 2 hours (8 slots) | 8 hours (32 slots) |
| Lighting | 0.3 kW | 15 min (1 slot) | 12 hours (48 slots) |
| Water Heater | 4.5 kW | 30 min (2 slots) | 2 hours (8 slots) |

**Output:** 50 individual CSVs (`home_000.csv` – `home_049.csv`) + combined `all_homes.csv` with ~144,000 records (each with 20+ columns).

---

### 2. Data Preprocessing
**File:** `src/preprocessing.py` (162 lines)

Transforms raw simulation data into ML-ready feature matrices.

#### Pipeline Steps

| Step | Method | Details |
|---|---|---|
| **Load** | `load_raw_data()` | Reads `all_homes.csv` with timestamp parsing |
| **Missing Values** | `handle_missing_values()` | Linear interpolation + backfill/forwardfill |
| **Outlier Clipping** | `detect_outliers()` | IQR method (1.5× factor) on all power columns |
| **Feature Engineering** | `add_features()` | Cyclical time encodings (`hour_sin`, `hour_cos`, `slot_sin`, `slot_cos`), 1-hour rolling mean/std, lag features (`t-1`, `t-4`) |
| **Normalization** | `normalize()` | MinMax scaling (configurable to Z-score) |
| **Splitting** | `temporal_split()` | 70/15/15 temporal split **per home** (preserves time ordering) |

#### Feature Vector (22 dimensions)
```
Appliance Powers : power_hvac_kw, power_washer_kw, power_dryer_kw,
                   power_dishwasher_kw, power_ev_charger_kw,
                   power_lighting_kw, power_water_heater_kw       (7)
Environment      : outdoor_temp_c, indoor_temp_c, humidity_pct     (3)
Time Encodings   : hour_sin, hour_cos                              (2)
Context          : price_inr_kwh, occupancy, is_weekend,
                   rolling_avg_power, rolling_std_power,
                   power_lag_1, power_lag_4                        (7)
Engineered       : slot_sin, slot_cos, hour_sin, hour_cos          (+3)
                                                         Total ≈ 22
```

---

### 3. VAE Feature Extraction
**File:** `src/vae.py` (201 lines)

Smart home data is noisy and heavily correlated (occupancy correlates with lighting, temperature correlates with HVAC). A **Variational Autoencoder** performs non-linear dimensionality reduction to extract core "latent behavioral features."

#### Network Architecture

```
     Input (22-dim)
         │
    ┌────▼────┐
    │ Linear  │ 22 → 64
    │  ReLU   │
    │ BatchNorm│
    │ Linear  │ 64 → 32
    │  ReLU   │
    │ BatchNorm│
    └────┬────┘
    ┌────┴────┐
    │ fc_μ    │ 32 → 16 (mean)
    │ fc_σ²   │ 32 → 16 (log-variance)
    └────┬────┘
         │ Reparameterization: z = μ + ε·σ
    ┌────▼────┐
    │ Linear  │ 16 → 32
    │  ReLU   │
    │ BatchNorm│
    │ Linear  │ 32 → 64
    │  ReLU   │
    │ BatchNorm│
    │ Linear  │ 64 → 22
    │ Sigmoid │
    └────▼────┘
    Reconstructed Input
```

#### Loss Function (β-VAE)

$$\mathcal{L}_{\text{VAE}} = \text{MSE}(x, x') + \beta \cdot D_{\text{KL}}(q(z|x) \| p(z))$$

Where KL Divergence:

$$D_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^{16} \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$

*   **β = 0.5** balances reconstruction quality with latent space regularization
*   Trained for **30 epochs** with Adam (lr = 0.001, batch = 128)
*   Output: 16-dimensional latent vectors for every timestep

---

### 4. Federated GRU Load Forecasting
**File:** `src/federated_gru.py` (314 lines)

The grid operator needs to know how much demand to expect in the next 24 hours. A **Gated Recurrent Unit (GRU)** network is trained on the VAE's latent temporal sequences — but crucially, using **Federated Learning** so raw data never leaves the home.

#### GRU Network

```
Input (16-dim latent) → GRU (2 layers, 64 hidden) → FC (64 → 32 → ReLU → 1)
Sequence length: 16 timesteps (4 hours of history)
Output: predicted total load (kW) for next timestep
```

#### Federated Averaging (FedAvg) Protocol

```
For each of 20 communication rounds:
  1. Server sends global GRU model to 40% of homes (random selection)
  2. Each selected home trains the model locally for 5 epochs
     using ONLY its own private latent sequences
  3. Homes send back ONLY their updated model weights
  4. Server aggregates weights:
     w_global = Σ (nₖ / n_total) × wₖ
     where nₖ = number of samples in client k
```

#### Key Privacy Guarantee
> **No raw appliance power, temperature, or occupancy data is ever transmitted.** Only model gradients/weights cross the network boundary. This is critical for GDPR/privacy compliance in real-world smart grid deployments.

**Metrics computed:** MAE, RMSE, MAPE on held-out test set.

---

### 5. Stackelberg Game Pricing
**File:** `src/stackelberg_game.py` (315 lines)

Instead of static TOU pricing, the grid operator (**Leader**) and smart homes (**Followers**) play a hierarchical game to establish an **equilibrium price** for each hour.

#### Game Formulation

**Followers (Smart Homes)** — maximize comfort, minimize bills:
*   Utility: $U(c) = w \cdot \sqrt{c}$ — concave, diminishing returns
*   Best response: $c^* = \min\left( \left(\frac{w}{2p}\right)^2, C_{\max} \right)$
*   Comfort weight $w = 2.5$ (higher → users demand more energy)

**Leader (Grid Operator)** — maximize revenue, avoid overload:
*   Revenue: $R = p \cdot \sum c_i^*$
*   Overload penalty (quadratic):

$$\text{Penalty} = \begin{cases} 0 & \text{if } \sum c_i^* \le \text{Capacity} \\ \frac{1}{2}\left( \sum c_i^* - \text{Capacity} \right)^2 & \text{otherwise} \end{cases}$$

**Equilibrium Search:**
1.  Coarse grid search over price range `[₹3, ₹15]`
2.  Refinement around the best price found
3.  Each evaluation: compute all follower best responses → total demand → leader profit

**Hourly Mode:** `run_hourly_stackelberg()` produces 24 independent equilibrium prices, blended with TOU prices: `SG_price = 0.5 × TOU + 0.5 × Equilibrium`.

---

### 6. Differential Evolution Scheduling
**File:** `src/differential_evolution.py` (556 lines)

The core optimizer. Given Stackelberg prices, the DE explores millions of possible schedules to find the one that minimizes a complex multi-objective fitness function.

#### DE/rand/1/bin Algorithm

```
Population: 80–100 individuals, each a flat binary vector (7 × 24 = 168 values)
Mutation:   v = x_r1 + F × (x_r2 − x_r3)      [F = 0.8]
Crossover:  binomial with CR = 0.9
Selection:  greedy — keep child if fitness(child) < fitness(parent)
Generations: 500 (pipeline) or ~100 (dashboard quick mode)
```

#### Hard Constraint Enforcement
The only hard constraint is **ON-count conservation** — the optimizer preserves the exact number of ON-hours per appliance. If a mutation produces too many/few ON-hours:
*   Over → turn off the most expensive hours
*   Under → turn on the cheapest available hours

#### Fitness Function (8 Terms)

| # | Term | What It Does | Type |
|---|---|---|---|
| 1 | **Energy Cost** | $\sum (\text{Power} \times \text{SG\_Price})$ | Minimize cost |
| 2 | **Occupancy Comfort** | Penalizes comfort appliances OFF when occupied, ON when empty | Comfort |
| 3 | **Temporal Proximity** | Quadratic penalty: $(d_{\min})^2 \times \text{Power}$ prevents moving loads far from preferred time | Anti-drift |
| 4 | **Consecutiveness** | Rewards gapless sequential runs (washer followed by dryer) | Sequential |
| 5 | **TCL Duty Cycle** | Penalizes HVAC/Water Heater running >2 consecutive hours | Realism |
| 6 | **Peak Dispersal** | Exponential penalty for running 4+ heavy appliances simultaneously | Load balancing |
| 7 | **Grid Capacity** | Squared penalty when hourly draw exceeds max kW | Safety |
| 8 | **Forecast Alignment** | Nudges schedule toward GRU-predicted aggregate demand curve | Cooperation |

---

### 7. Constraint Engine
**File:** `src/constraints.py` (215 lines)

Enforces physical and operational constraints on any candidate schedule:

| Constraint | Implementation | Penalty |
|---|---|---|
| **Energy Cap** | Max 15 kW/hour, max 80 kWh/day | Linear penalty for each violation |
| **Min Run Duration** | Each appliance type has a minimum consecutive ON duration (e.g., washer = 45 min) | Penalty per short run |
| **Sequence** | Washer must finish before dryer starts | Penalty = overlap slots |
| **Comfort** | HVAC must be ON during occupied hours; lighting ON during occupied evenings (6 PM+) | Weighted penalty per violation |

The `get_constraint_function()` factory produces a bound constraint checker used by the DE optimizer.

---

### 8. Interactive Dashboard
**File:** `app/dashboard.py` (902 lines)

A fully interactive Plotly Dash web application for real-time visualization and optimization.

#### Dashboard Features

| Feature | Description |
|---|---|
| **Schedule Grid** | 7 × 24 clickable toggle grid — click cells to turn appliances ON/OFF per hour |
| **Preset Buttons** | "All Peak", "All Off-Peak", "Spread", "Reset" for quick schedule generation |
| **Electricity Rates** | Editable Off-Peak, Shoulder, and Peak rate inputs (₹/kWh) |
| **Grid Capacity** | Configurable max kW per hour with penalty explanation |
| **Occupancy Input** | 24-hour occupancy bar (0–5 people per hour) |
| **Metric Cards** | Your Cost / Optimized Cost / Savings % / Peak Reduction % |
| **Live Floorplan** | 2D house visualization showing appliance power states for each hour, color-coded by load (green < yellow < red) |
| **Time Slider** | Scrub through 24 hours; floorplans and price display update in real-time |
| **Heatmap** | Side-by-side "Your Schedule" vs "Optimized" comparison, colored by electricity price when ON |
| **ML Integration** | If trained models exist, the dashboard uses VAE→GRU inference + Stackelberg pricing; otherwise falls back to rule-based optimization |

#### How the Dashboard Optimize Button Works

```
User clicks "🚀 Optimize Schedule":
  1. Read user's 7×24 schedule, occupancy, rates, and grid capacity
  2. Try to load cached VAE + GRU models for demand forecasting
     └─ If models exist: build feature frame → VAE encode → GRU predict → 24h forecast
     └─ If not: skip ML inference, use rules only
  3. Run hourly Stackelberg game → 24 equilibrium prices
  4. Run quick_optimize() (DE with ~100 generations)
  5. Display: cost cards, side-by-side floorplans, comparison heatmap
```

---

## ⚙️ Configuration

All parameters are centralized in `config/config.yaml`:

| Section | Key Parameters |
|---|---|
| `digital_twin` | `num_homes: 50`, `simulation_days: 30`, `time_resolution_minutes: 15`, appliance specs (power, durations), Markov transition matrices, environmental params (30°C base, ±8°C swing), TOU pricing (₹2/₹5/₹12) |
| `preprocessing` | `normalization: minmax`, `train_ratio: 0.70`, `outlier_iqr_factor: 1.5`, `rolling_window_slots: 4` |
| `vae` | `input_dim: 12` (auto-adjusted), `hidden_dim: 64`, `latent_dim: 16`, `epochs: 30`, `kl_weight: 0.5` |
| `federated_gru` | `hidden_dim: 64`, `num_layers: 2`, `sequence_length: 16`, `num_rounds: 20`, `local_epochs: 5`, `client_fraction: 0.4` |
| `stackelberg` | `num_iterations: 100`, `comfort_weight: 2.5`, `supply_capacity_kw: 500.0`, `price_range: [3.0, 15.0]` |
| `differential_evolution` | `population_size: 100`, `mutation_factor: 0.8`, `crossover_rate: 0.9`, `max_generations: 500` |
| `constraints` | `max_daily_energy_kwh: 80.0`, `max_hourly_power_kw: 15.0`, `appliance_sequences: [washer→dryer]` |

---

## 🚀 Getting Started

### Prerequisites
*   **Python 3.10+** recommended
*   **pip** package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/AltZilla/dscproj.git
cd dscproj

# Install dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline (Train Everything)

```bash
python main.py --run-all
```

This executes all 6 stages sequentially (~10–30 min depending on hardware):
1. Simulates 50 homes for 30 days → `data/raw/`
2. Preprocesses data → `data/processed/`
3. Trains VAE → `outputs/models/vae.pt`
4. Trains Federated GRU → `outputs/models/federated_gru.pt`
5. Computes Stackelberg equilibrium prices
6. Runs DE optimization → saves summary report

### Launch the Dashboard (Interactive Mode)

```bash
python main.py --dashboard
```

Or directly:

```bash
python app/dashboard.py
```

Open your browser to **`http://127.0.0.1:8050`** to interact with the optimizer.

> **Note:** If you run the dashboard without training models first, it will fall back to rule-based optimization (no ML inference). Train models first with `--run-all` for the full experience.

### Run Individual Modules

```bash
# Only simulate new data
python main.py --module digital_twin

# Only preprocess existing data
python main.py --module preprocessing

# Skip simulation, use existing CSVs
python main.py --run-all --skip-simulation

# Use GPU for training
python main.py --run-all --device cuda

# Use a custom config
python main.py --run-all --config path/to/custom_config.yaml
```

---

## 💡 Usage Examples

### Example 1: Quick Dashboard Demo
```bash
# Start dashboard with pre-trained models
python main.py --dashboard

# In your browser:
# 1. Click appliance cells to toggle ON/OFF hours
# 2. Adjust electricity rates (try ₹2 / ₹5 / ₹15)
# 3. Set occupancy per hour
# 4. Set grid capacity (try 10 kW vs 20 kW)
# 5. Click "🚀 Optimize Schedule"
# 6. Drag the time slider to compare floorplans hour-by-hour
```

### Example 2: Regenerate Data Only
```bash
# Edit config/config.yaml to change num_homes to 100
python main.py --module digital_twin
# New data saved to data/raw/
```

### Example 3: Retrain with Existing Data
```bash
python main.py --run-all --skip-simulation --device cuda
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3.10+ |
| **Deep Learning** | PyTorch (VAE, GRU networks) |
| **Data Processing** | Pandas, NumPy, scikit-learn |
| **Optimization** | SciPy (support), custom DE implementation |
| **Visualization** | Plotly, Dash, dash-bootstrap-components, Matplotlib |
| **Configuration** | PyYAML |
| **Progress** | tqdm |

### `requirements.txt`
```
# Core ML
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Visualization & Dashboard
plotly>=5.15.0
dash>=2.14.0
dash-bootstrap-components>=1.5.0
matplotlib>=3.7.0

# Utilities
pyyaml>=6.0
tqdm>=4.65.0
```

---

## 📚 References

The following academic papers informed the design of this system:

1.  **Big Data Intelligence Analytics for Energy Optimization** — foundations for data-driven grid management
2.  **Multi-Agent Deep Reinforcement Learning** — game-theoretic agent interactions in energy markets
3.  **Spatiotemporal Federated Learning** — privacy-preserving distributed model training for IoT environments

---

<div align="center">

**Built with ⚡ for the DSC Project**

*A simulation platform for AI-driven energy optimization, game-theoretic pricing, and privacy-preserving load forecasting.*

</div>
