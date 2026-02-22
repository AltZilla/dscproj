# Smart Home Energy Optimizer: AI-Driven Grid Simulation & Scheduling

## 📖 Introduction & Motivation
The transition to renewable energy sources and the increasing proliferation of power-hungry home appliances (such as EV chargers and HVAC systems) are placing unprecedented strain on global electrical grids. Traditional static Time-of-Use (TOU) pricing models are often insufficient to actively manage and shave peak loads in real-time.

This project introduces a **comprehensive, end-to-end AI-driven smart grid and home energy simulation platform**. It replaces rigid rule-based scheduling with a dynamic, game-theoretic, and privacy-preserving architecture. By bridging the gap between utility-scale load forecasting and individual smart home appliance optimization, this system finds the perfect equilibrium: saving consumers money while protecting the grid from catastrophic overloading during peak hours.

To achieve this, the project constructs a complex pipeline merging **Deep Learning (VAE, GRU)**, **Federated Learning (FedAvg)**, **Game Theory (Stackelberg Game)**, and **Evolutionary Algorithms (Differential Evolution)**.

---

## 🏗️ System Architecture & Workflow Pipeline

The entire system operates as a unified pipeline where the output of one component directly drives the intelligence of the next:

1. **Synthetic Data Generation:** A Digital Twin simulates thousands of realistic smart homes, generating 15-minute resolution metrics (appliance states, occupancy, indoor/outdoor temperatures, pricing).
2. **Latent Feature Extraction:** A Variational Autoencoder (VAE) compresses the noisy, high-dimensional smart home metrics into dense behavioral patterns.
3. **Decentralized Load Forecasting:** A Federated GRU network predicts future household energy demands using the VAE's latent features, training localized models without exposing sensitive raw consumer data to the central server.
4. **Dynamic Grid Pricing:** A Stackelberg Game computes real-time optimal electricity prices by balancing the GRU's total forecasted demand against the raw physical capacity of the grid.
5. **Appliance Optimization:** Differential Evolution (DE) algorithms inside each smart home respond to the game-theoretic prices, shifting expensive loads to optimal timeslots based on rigorous constraints (comfort, sequences, and timing).

---

## 🔍 Detailed Component Breakdown & Mathematical Formulations

### 1. Digital Twin Simulator (`src/digital_twin.py`)
AI models require vast amounts of high-quality data. The Digital Twin acts as the foundational data synthesizer, modeling real-world physics and human behavior.

*   **Appliance State Modeling:** Simulates constraints and probabilities for 7 distinct appliance categories (HVAC, Washer, Dryer, Dishwasher, EV Charger, Lighting, Water Heater).
*   **Markov Chain Occupancy:** Models human presence using a 3-state transition matrix (`Home=0`, `Away=1`, `Sleeping=2`). The probability matrices vary dynamically between `Night`, `Day`, and `Evening` hours.
*   **Thermal Environment Model:** Uses a thermodynamic decay simulator where indoor temperature slowly drifts toward an outdoor sinusoidal temperature curve, counteracted by HVAC activation cooling rates.

### 2. VAE Feature Extraction (`src/vae.py`)
Smart home data is extremely noisy and heavily correlated (e.g., occupancy directly correlates with lighting and HVAC usage). A **Variational Autoencoder (VAE)** performs non-linear dimensionality reduction to extract core "latent behavioral features" from each house.

*   **The Encoder:** Compresses environmental variables, occupancy states, and past power usage into a condensed latent space defined by a mean ($\mu$) and variance ($\sigma^2$).
*   **The Decoder:** Reconstructs the original data from the latent representation to self-supervise the learning process.
*   **The Loss Function:** 
    $$\mathcal{L}_{VAE} = \text{MSE}(x, x') + \beta \cdot D_{KL}(q(z|x) || p(z))$$
    Where the Kullback-Leibler (KL) Divergence penalty forces the latent space to follow a normal distribution, improving the downstream GRU's generalization:
    $$D_{KL} = -\frac{1}{2} \sum_{j=1}^{J} \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$

### 3. Federated GRU Load Forecasting (`src/federated_gru.py`)
To dynamically price electricity without overloading the grid, the utility provider needs to know exactly how much demand to expect in the next 24 hours. The system uses a **Gated Recurrent Unit (GRU)** network trained on the VAE's latent temporal sequences.

*   **Federated Learning (FedAvg):** To strictly preserve user privacy, the raw IoT data never leaves the smart home.
    1. The central server distributes a global GRU model to selected homes (clients).
    2. Each client trains the model locally acting on their private VAE latent sequences.
    3. Clients send only their updated model **weights** back to the server.
    4. The server aggregates the updates using a weighted average based on the local dataset sizes ($w_k = \frac{n_k}{n_{total}}$) to construct a smarter global model, keeping data completely decentralized.

### 4. Stackelberg Game Grid Pricing (`src/stackelberg_game.py`)
Instead of dumb, static Time-of-Use (TOU) pricing, the grid operator (Leader) and the smart homes (Followers) play a hierarchical game to establish an **Equilibrium Price (SG)**.

*   **The Followers (Smart Homes):** Consumers want to maximize their energy consumption for comfort, but want to minimize their electricity bills.
    *   *Utility Function (Concave Diminishing Returns):* $U(c) = w \cdot \sqrt{c}$ 
        (Where $w$ is the localized comfort weight factor, and $c$ is the consumption in kW).
    *   *Follower Best Response:* By solving $\max_c [ U(c) - p \cdot c ]$, the optimal energy consumption for a given price $p$ becomes:
        $$c^* = \min\left( \left(\frac{w}{2p}\right)^2, C_{max} \right)$$
*   **The Leader (Grid Operator):** The utility operator wants to maximize revenue off the electricity sold, while explicitly enforcing severe penalties to deter demand from exceeding physical grid safety limits.
    *   *Revenue:* $R = p \cdot \sum c_i^*$
    *   *Safety Penalty (Quadratic Overload Cost):* 
        $$Cost = \begin{cases} 0 & \text{if } \sum c_i^* \le \text{Capacity} \\ \frac{1}{2}\left( \sum c_i^* - \text{Capacity} \right)^2 & \text{if } \sum c_i^* > \text{Capacity} \end{cases}$$
*   **Equilibrium Search:** The system runs an iterative grid-search to find the exact Stackelberg Equilibrium Price `p*` that maximizes the leader's net profit while giving followers the utility they demand.

### 5. Differential Evolution (DE) Scheduling (`src/differential_evolution.py` & `src/constraints.py`)
Once the grid broadcasts the real-time game-theoretic SG blended prices, individual homes optimize their appliance schedules. It actively searches millions of possible day-ahead home schedules to minimize a complex objective fitness function using **DE/rand/1/bin**.

*   **Evolutionary Workflow:** 
    The DE maintains a constant population of binary schedules. Through mutation ($v = x_{r1} + F \cdot (x_{r2} - x_{r3})$) and binomial crossover, the population explores the search space and strictly selects schedules that score lowest on the fitness penalty function.
*   **Hard Constraints:** 
    *   *Sequential:* The Dryer must exclusively run *after* the Washer. 
    *   *Minimum Run Duration:* Washers must execute uninterrupted for at least 45 minutes once engaged.
*   **The Model-Driven Fitness Logic:**
    The DE does not use hardcoded slot replacements. It uses domain-knowledge penalty modeling.
    1.  **Energy Cost Objective:** Minimizes $\sum (\text{Power} \cdot \text{SG\_Price})$.
    2.  **Occupancy Comfort Penalty:** Severely punishes turning HVAC or Lighting off while people are actively home, or turning them on when the house is empty.
    3.  **Temporal Proximity Penalty:** Quadratic punishment designed to prevent the algorithm from moving an appliance unreasonably far from the consumer's originally requested time ($Penalty = (Distance\_in\_Hours)^2 \cdot \text{Power} \cdot 0.3$). This ensures 7kW EV chargers shift slightly to shoulder periods rather than moving 14 hours away.
    4.  **Peak Capacity Dispersal:** Exponentially punishes schedules that attempt to stack multiple heavy appliances in the very same hour.

---

## 🚀 Running the Optimization Dashboard
You can visually run, interrogate, and interact with this entire multi-layered pipeline—including real-time GRU forecasting, SG pricing interactions, and before/after schedule plotting—through the localized interactive dashboard.

### Prerequisites & Setup
Ensure you have all dependencies installed.

```bash
# Install required packages (if not already done)
pip install -r requirements.txt
```

### Launch the UI
```bash
python app/dashboard.py
```
*   This boots up a fully interactive Dash interface.
*   Open your browser to `http://127.0.0.1:8050` (or the port specified in your terminal).
*   Toggle your smart appliances, modify grid capacity limits, and click **"Optimize Schedule"** to watch the Differential Evolution algorithm seamlessly shift your devices away from Peak Time-of-Use periods while respecting all comfort and sequencing constraints!
