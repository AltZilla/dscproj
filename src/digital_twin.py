"""
Digital Twin - Smart Home Simulation Engine
============================================
Simulates N smart homes with stateful appliances, Markov-chain occupancy,
thermal environment, and Time-of-Use pricing. Produces realistic IoT energy
consumption data for downstream ML modules.
"""

import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------
def load_config(config_path: Optional[str] = None) -> dict:
    """Load config.yaml from the project root."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Appliance Model
# ---------------------------------------------------------------------------
class Appliance:
    """Models a single smart-home appliance with power profile and state."""

    # Base activation probabilities by hour (0-23) — overridden per type
    _BASE_SCHEDULES = {
        "hvac":         [0.3]*6 + [0.5]*6 + [0.7]*6 + [0.6]*6,
        "washer":       [0.0]*7 + [0.1]*2 + [0.3]*3 + [0.1]*4 + [0.2]*2 + [0.1]*6,
        "dryer":        [0.0]*8 + [0.05]*3 + [0.2]*3 + [0.1]*4 + [0.15]*2 + [0.0]*4,
        "dishwasher":   [0.0]*7 + [0.1]*1 + [0.0]*4 + [0.2]*2 + [0.0]*4 + [0.3]*2 + [0.1]*4,
        "ev_charger":   [0.3]*6 + [0.0]*12 + [0.1]*2 + [0.4]*4,
        "lighting":     [0.1]*6 + [0.3]*2 + [0.1]*4 + [0.1]*4 + [0.5]*4 + [0.7]*4,
        "water_heater": [0.1]*5 + [0.5]*2 + [0.2]*5 + [0.1]*4 + [0.3]*2 + [0.4]*2 + [0.2]*4,
    }

    def __init__(self, name: str, rated_power_kw: float,
                 min_duration_slots: int, max_duration_slots: int):
        self.name = name
        self.rated_power_kw = rated_power_kw
        self.min_duration_slots = min_duration_slots
        self.max_duration_slots = max_duration_slots

        self.is_on = False
        self.remaining_slots = 0
        self.hourly_schedule = self._BASE_SCHEDULES.get(name, [0.1] * 24)

    def get_activation_probability(self, hour: int, occupancy_state: int,
                                   indoor_temp: float, is_weekend: bool,
                                   comfort_min: float, comfort_max: float) -> float:
        """Compute probability of turning ON this slot."""
        if self.is_on:
            return 0.0  # Already on, let it finish

        base_prob = self.hourly_schedule[hour]

        # Occupancy modifier: most appliances need someone home
        if self.name not in ("ev_charger",):
            if occupancy_state == 1:    # Away
                base_prob *= 0.05
            elif occupancy_state == 2:  # Sleeping
                base_prob *= 0.2

        # Weekend boost for washer/dryer/dishwasher
        if is_weekend and self.name in ("washer", "dryer", "dishwasher"):
            base_prob *= 1.5

        # HVAC responds to temperature
        if self.name == "hvac":
            if indoor_temp > comfort_max:
                base_prob = min(1.0, base_prob + 0.4)
            elif indoor_temp < comfort_min:
                base_prob = min(1.0, base_prob + 0.3)
            else:
                base_prob *= 0.3  # Comfort zone — less likely to run

        return min(base_prob, 1.0)

    def step(self, activate: bool) -> float:
        """Advance one time step. Return power consumption (kW)."""
        if self.is_on:
            self.remaining_slots -= 1
            if self.remaining_slots <= 0:
                self.is_on = False
                self.remaining_slots = 0
            return self.rated_power_kw

        if activate:
            self.is_on = True
            self.remaining_slots = np.random.randint(
                self.min_duration_slots, self.max_duration_slots + 1
            )
            return self.rated_power_kw

        return 0.0

    def reset(self):
        self.is_on = False
        self.remaining_slots = 0


# ---------------------------------------------------------------------------
# Occupancy Model (Markov Chain)
# ---------------------------------------------------------------------------
class OccupancyModel:
    """3-state Markov chain: Home(0), Away(1), Sleeping(2)."""

    # Time-varying transition matrices (night vs day vs evening)
    _NIGHT_HOURS = set(range(0, 6))      # 12am - 6am
    _DAY_HOURS = set(range(6, 18))       # 6am - 6pm
    _EVENING_HOURS = set(range(18, 24))  # 6pm - 12am

    def __init__(self, config: dict):
        self.state = config.get("initial_state", 2)

        # Build time-dependent transition matrices
        base = config["transitions"]
        self._day_matrix = np.array([
            [0.6, 0.35, 0.05],   # Home → more likely to go out during day
            [0.25, 0.75, 0.0],   # Away → stays away during work
            [0.8, 0.15, 0.05],   # Sleeping → wakes up and likely goes out
        ])
        self._evening_matrix = np.array([
            [0.75, 0.1, 0.15],   # Home → stays home, may sleep
            [0.6, 0.3, 0.1],     # Away → comes home in evening
            [0.3, 0.0, 0.7],     # Sleeping → stays sleeping
        ])
        self._night_matrix = np.array([
            [0.2, 0.0, 0.8],     # Home → goes to sleep
            [0.5, 0.2, 0.3],     # Away → comes home and sleeps
            [0.05, 0.0, 0.95],   # Sleeping → stays sleeping
        ])

    def step(self, hour: int) -> int:
        """Transition to next state based on current hour."""
        if hour in self._NIGHT_HOURS:
            matrix = self._night_matrix
        elif hour in self._DAY_HOURS:
            matrix = self._day_matrix
        else:
            matrix = self._evening_matrix

        probs = matrix[self.state]
        self.state = np.random.choice(3, p=probs)
        return self.state

    def reset(self):
        self.state = 2  # Start sleeping

    @staticmethod
    def state_name(state: int) -> str:
        return ["Home", "Away", "Sleeping"][state]


# ---------------------------------------------------------------------------
# Environment Model
# ---------------------------------------------------------------------------
class EnvironmentModel:
    """Simulates outdoor/indoor temperature and humidity."""

    def __init__(self, config: dict):
        self.base_temp = config["base_outdoor_temp_c"]
        self.amplitude = config["temp_amplitude_c"]
        self.humidity_base = config["humidity_base_pct"]
        self.decay_rate = config["indoor_thermal_decay_rate"]
        self.hvac_cooling_rate = config["hvac_cooling_rate_c"]
        self.comfort_min = config["comfort_temp_min_c"]
        self.comfort_max = config["comfort_temp_max_c"]

        self.indoor_temp = (self.comfort_min + self.comfort_max) / 2.0
        self.outdoor_temp = self.base_temp

    def step(self, hour: int, day: int, hvac_on: bool) -> Tuple[float, float, float]:
        """
        Update environment for this timestep.
        Returns (outdoor_temp, indoor_temp, humidity).
        """
        # Outdoor: sinusoidal daily cycle (peak at 2pm = hour 14)
        hour_angle = 2 * np.pi * (hour - 14) / 24
        self.outdoor_temp = (
            self.base_temp
            + self.amplitude * np.cos(hour_angle)
            + np.random.normal(0, 1.0)  # Random noise
        )

        # Indoor: drifts toward outdoor, HVAC counteracts
        drift = self.decay_rate * (self.outdoor_temp - self.indoor_temp)
        self.indoor_temp += drift

        if hvac_on:
            if self.indoor_temp > self.comfort_max:
                self.indoor_temp -= self.hvac_cooling_rate * 0.25  # Per 15-min slot
            elif self.indoor_temp < self.comfort_min:
                self.indoor_temp += self.hvac_cooling_rate * 0.25

        # Humidity: inversely correlates with temp
        humidity = self.humidity_base - 0.5 * (self.outdoor_temp - self.base_temp) \
                   + np.random.normal(0, 3.0)
        humidity = np.clip(humidity, 30.0, 95.0)

        return self.outdoor_temp, self.indoor_temp, humidity

    def reset(self):
        self.indoor_temp = (self.comfort_min + self.comfort_max) / 2.0


# ---------------------------------------------------------------------------
# Pricing Model
# ---------------------------------------------------------------------------
class PricingModel:
    """Time-of-Use electricity pricing."""

    def __init__(self, config: dict):
        self.rate_map = {}
        for period_name in ("off_peak", "shoulder", "peak"):
            period = config[period_name]
            for h in period["hours"]:
                self.rate_map[h] = period["rate"]

    def get_rate(self, hour: int) -> float:
        return self.rate_map.get(hour, 5.0)

    def get_period_name(self, hour: int) -> str:
        rate = self.get_rate(hour)
        if rate <= 3.0:
            return "off_peak"
        elif rate <= 5.0:
            return "shoulder"
        else:
            return "peak"


# ---------------------------------------------------------------------------
# Smart Home (Digital Twin)
# ---------------------------------------------------------------------------
class SmartHome:
    """A single smart home with appliances, occupancy, environment, pricing."""

    def __init__(self, home_id: int, config: dict):
        self.home_id = home_id
        self.twin_cfg = config["digital_twin"]

        # Initialize appliances
        self.appliances: Dict[str, Appliance] = {}
        for name, spec in self.twin_cfg["appliances"].items():
            self.appliances[name] = Appliance(
                name=name,
                rated_power_kw=spec["rated_power_kw"],
                min_duration_slots=spec["min_duration_slots"],
                max_duration_slots=spec["max_duration_slots"],
            )

        # Sub-models
        self.occupancy = OccupancyModel(self.twin_cfg["occupancy"])
        self.environment = EnvironmentModel(self.twin_cfg["environment"])
        self.pricing = PricingModel(self.twin_cfg["pricing"])

        # Add slight per-home variation
        self._behavior_scale = np.random.uniform(0.7, 1.3)

    def simulate_step(self, hour: int, day: int, slot_in_day: int,
                      is_weekend: bool) -> dict:
        """Run one 15-minute timestep. Returns a dict of all readings."""
        # 1. Occupancy
        occ_state = self.occupancy.step(hour)

        # 2. Environment (need to know if HVAC was on last step)
        hvac_on = self.appliances["hvac"].is_on
        outdoor_temp, indoor_temp, humidity = self.environment.step(hour, day, hvac_on)

        comfort_min = self.twin_cfg["environment"]["comfort_temp_min_c"]
        comfort_max = self.twin_cfg["environment"]["comfort_temp_max_c"]

        # 3. Appliance activation
        appliance_powers = {}
        for name, appliance in self.appliances.items():
            prob = appliance.get_activation_probability(
                hour, occ_state, indoor_temp, is_weekend,
                comfort_min, comfort_max
            )
            prob *= self._behavior_scale
            prob = min(prob, 1.0)
            activate = np.random.random() < prob
            power = appliance.step(activate)
            appliance_powers[name] = power

        # 4. Pricing
        price = self.pricing.get_rate(hour)
        period = self.pricing.get_period_name(hour)

        # 5. Total
        total_power = sum(appliance_powers.values())
        slot_cost = total_power * price * (15 / 60)  # kWh for 15 min

        record = {
            "home_id": self.home_id,
            "day": day,
            "hour": hour,
            "slot_in_day": slot_in_day,
            "is_weekend": int(is_weekend),
            "occupancy": occ_state,
            "occupancy_name": OccupancyModel.state_name(occ_state),
            "outdoor_temp_c": round(outdoor_temp, 2),
            "indoor_temp_c": round(indoor_temp, 2),
            "humidity_pct": round(humidity, 2),
            "price_inr_kwh": price,
            "price_period": period,
            "total_power_kw": round(total_power, 3),
            "slot_cost_inr": round(slot_cost, 4),
        }
        # Individual appliance power
        for name, power in appliance_powers.items():
            record[f"power_{name}_kw"] = round(power, 3)

        return record

    def reset(self):
        for a in self.appliances.values():
            a.reset()
        self.occupancy.reset()
        self.environment.reset()


# ---------------------------------------------------------------------------
# Digital Twin Simulator (orchestrates all homes)
# ---------------------------------------------------------------------------
class DigitalTwinSimulator:
    """Runs the full simulation across N homes for D days."""

    def __init__(self, config: dict):
        self.config = config
        self.twin_cfg = config["digital_twin"]
        self.num_homes = self.twin_cfg["num_homes"]
        self.sim_days = self.twin_cfg["simulation_days"]
        self.resolution = self.twin_cfg["time_resolution_minutes"]
        self.slots_per_day = 24 * 60 // self.resolution

        np.random.seed(self.twin_cfg.get("seed", 42))

        self.homes: List[SmartHome] = [
            SmartHome(i, config) for i in range(self.num_homes)
        ]

    def run(self, verbose: bool = True) -> pd.DataFrame:
        """
        Run the full simulation.
        Returns a DataFrame with all readings for all homes and timesteps.
        """
        all_records = []
        total_steps = self.num_homes * self.sim_days * self.slots_per_day

        if verbose:
            from tqdm import tqdm
            pbar = tqdm(total=total_steps, desc="Simulating Digital Twin")

        for home in self.homes:
            home.reset()
            for day in range(self.sim_days):
                # Day-of-week: 0=Mon ... 6=Sun (start from a Monday)
                dow = day % 7
                is_weekend = dow >= 5

                for slot in range(self.slots_per_day):
                    hour = (slot * self.resolution) // 60
                    record = home.simulate_step(hour, day, slot, is_weekend)
                    record["timestamp"] = (
                        pd.Timestamp("2025-01-01")
                        + pd.Timedelta(days=day)
                        + pd.Timedelta(minutes=slot * self.resolution)
                    )
                    all_records.append(record)

                    if verbose:
                        pbar.update(1)

        if verbose:
            pbar.close()

        df = pd.DataFrame(all_records)
        return df

    def save_data(self, df: pd.DataFrame, output_dir: Optional[str] = None):
        """Save simulation data — one CSV per home + combined file."""
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent / "data" / "raw"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save per-home CSVs
        for home_id in range(self.num_homes):
            home_df = df[df["home_id"] == home_id]
            home_df.to_csv(output_dir / f"home_{home_id:03d}.csv", index=False)

        # Save combined
        df.to_csv(output_dir / "all_homes.csv", index=False)
        print(f"Saved {len(df)} records to {output_dir}")

    def get_home_snapshot(self, home_id: int, slot_index: int,
                          df: pd.DataFrame) -> dict:
        """
        Get a snapshot of a specific home at a specific time slot.
        Used by the visual dashboard.
        """
        home_data = df[(df["home_id"] == home_id)]
        if slot_index >= len(home_data):
            slot_index = len(home_data) - 1

        row = home_data.iloc[slot_index]
        snapshot = row.to_dict()

        # Extract appliance states
        snapshot["appliances"] = {}
        for name in self.config["digital_twin"]["appliances"]:
            key = f"power_{name}_kw"
            if key in snapshot:
                snapshot["appliances"][name] = {
                    "power_kw": snapshot[key],
                    "is_on": snapshot[key] > 0,
                    "rated_kw": self.config["digital_twin"]["appliances"][name]["rated_power_kw"]
                }
        return snapshot


# ---------------------------------------------------------------------------
# Entry point for standalone execution
# ---------------------------------------------------------------------------
def run_digital_twin(config_path: Optional[str] = None) -> pd.DataFrame:
    """High-level function to run the digital twin and save results."""
    config = load_config(config_path)
    simulator = DigitalTwinSimulator(config)

    print(f"Running Digital Twin: {simulator.num_homes} homes × "
          f"{simulator.sim_days} days × {simulator.slots_per_day} slots/day")

    df = simulator.run(verbose=True)

    # Print summary stats
    print(f"\nSimulation Summary:")
    print(f"  Total records:      {len(df):,}")
    print(f"  Avg power per slot: {df['total_power_kw'].mean():.2f} kW")
    print(f"  Max power per slot: {df['total_power_kw'].max():.2f} kW")
    print(f"  Total cost:         ₹{df['slot_cost_inr'].sum():,.2f}")
    print(f"  Avg daily cost/home:₹{df['slot_cost_inr'].sum() / simulator.num_homes / simulator.sim_days:.2f}")

    simulator.save_data(df)
    return df


if __name__ == "__main__":
    run_digital_twin()
