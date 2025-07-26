# ✅ Final Version: Scenario 3 with Worker "Roham"
# Roles:
# - M (Morad): rents tools only
# - K (Kikavoos): handles maintenance + cleaning (atomically)
# - R (Roham): floating worker, helps first with renting, then with maint+clean

import heapq
import math
import pandas as pd
import numpy as np
import random

# Simulation timing constants
SIMULATION_START = 0
SIMULATION_END = 600
CLOSE_TIME = 660

# Linear Congruential Generator for reproducible random numbers
class LCG:
    def __init__(self, seed=1):
        self.a = 1664525
        self.c = 1013904223
        self.m = 2 ** 32
        self.state = seed

    def random(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m

# Normal distribution using Box-Muller transform
def generate_normal_direct(mu, sigma_squared, lcg=None):
    rand = lcg.random if lcg else random.random
    R1 = rand()
    R2 = rand()
    Z = math.sqrt(-2 * math.log(R1)) * math.cos(2 * math.pi * R2)
    return max(0.1, mu + math.sqrt(sigma_squared) * Z)

# Random time generators
def next_arrival(lcg): return generate_normal_direct(30, 900, lcg)
def rental_time(person, lcg): return generate_normal_direct(14, 16, lcg) if person == "K" else generate_normal_direct(10, 25, lcg)
def usage_duration(lcg): return generate_normal_direct(60, 3600, lcg)
def maintenance_time(lcg): return generate_normal_direct(6, 16, lcg)
def cleaning_time(lcg): return generate_normal_direct(10, 36, lcg)

# Event class for discrete-event simulation
class Event:
    def __init__(self, time, type, data=None):
        self.time = time
        self.type = type
        self.data = data
    def __lt__(self, other): return self.time < other.time

# Resets all simulation state variables
def reset_state():
    return {
        "fel": [], "clock": SIMULATION_START, "queue": [], "return_queue": [],
        "to_maintain": [], "K_busy": False, "M_busy": False, "R_busy": False,
        "K_status": "idle", "M_status": "idle", "R_status": "idle",
        "active_customers": set(), "stats": {
            "wait_times": [], "queue_lengths": [],
            "rent_times_K": [], "rent_times_M": [], "rent_times_R": [],
            "maint_times": [], "clean_times": [],
            "count_rent_K": 0, "count_rent_M": 0, "count_rent_R": 0,
            "count_maint": 0, "count_clean": 0,
            "count_maint_K": 0, "count_maint_R": 0,
            "count_clean_K": 0, "count_clean_R": 0,
            "timeline": [], "wait_over_5min_count": 0, "total_customers_with_wait": 0
        },
        "customer_count": 0
    }

# Handles a customer arrival: queues them, schedules return, adds next arrival
def handle_arrival(state, lcg):
    now = state["clock"]
    cid = f"W{state['customer_count']}"
    state["customer_count"] += 1
    state["queue"].append((now, cid))
    state["active_customers"].add(cid)
    return_time = now + usage_duration(lcg)
    if return_time < CLOSE_TIME:
        heapq.heappush(state["fel"], Event(return_time, "request_return", cid))
    next_time = now + next_arrival(lcg)
    if next_time < SIMULATION_END:
        heapq.heappush(state["fel"], Event(next_time, "arrival"))
    assign_workers(state, lcg)

# Ends a rental, frees the worker
def handle_end_rent(state, who, lcg):
    state[f"{who}_busy"] = False
    state[f"{who}_status"] = "idle"
    state["stats"]["timeline"].append((state["clock"], "end_rent", who))
    assign_workers(state, lcg)

# Handles tool return request
def handle_end_return(state, cid, lcg):
    if cid in state["active_customers"]:
        state["active_customers"].remove(cid)
    state["to_maintain"].append(f"tool_from_{cid}")
    for who in ["K", "M", "R"]:
        if state[f"{who}_status"] == "returning":
            state[f"{who}_busy"] = False
            state[f"{who}_status"] = "idle"
    assign_workers(state, lcg)

# Ends maintenance and begins cleaning
def handle_end_maint(state, who, lcg):
    t_clean = cleaning_time(lcg)
    state[f"{who}_status"] = "cleaning"
    state["stats"]["count_maint"] += 1
    state["stats"][f"count_maint_{who}"] += 1
    state["stats"]["maint_times"].append(t_clean)
    heapq.heappush(state["fel"], Event(state["clock"] + t_clean, "end_clean", who))

# Ends cleaning task
def handle_end_clean(state, who, lcg):
    state[f"{who}_status"] = "idle"
    state[f"{who}_busy"] = False
    state["stats"]["count_clean"] += 1
    state["stats"][f"count_clean_{who}"] += 1
    state["stats"]["clean_times"].append(cleaning_time(lcg))
    assign_workers(state, lcg)

# Assigns workers to tasks: return -> rental -> maintenance
def assign_workers(state, lcg):
    now = state["clock"]
    if now >= CLOSE_TIME: return

    # Priority 1: handle returns
    for who in ["K", "M", "R"]:
        if not state[f"{who}_busy"] and state["return_queue"]:
            cid = state["return_queue"].pop(0)
            state[f"{who}_busy"] = True
            state[f"{who}_status"] = "returning"
            heapq.heappush(state["fel"], Event(now + 2, "return", cid))
            return

    # Priority 2: handle rentals (M and R only)
    for who in ["M", "R"]:
        if not state[f"{who}_busy"]:
            for i, (t0, cid) in enumerate(state["queue"]):
                if cid not in state["active_customers"]: continue
                wait = now - t0
                rent = rental_time(who, lcg)
                state[f"{who}_busy"] = True
                state[f"{who}_status"] = "renting"
                state["stats"]["wait_times"].append(wait)
                state["stats"][f"rent_times_{who}"].append(rent)
                state["stats"][f"count_rent_{who}"] += 1
                state["stats"]["total_customers_with_wait"] += 1
                if wait > 5:
                    state["stats"]["wait_over_5min_count"] += 1
                del state["queue"][i]
                heapq.heappush(state["fel"], Event(now + rent, "end_rent", who))
                return

    # Priority 3: handle maintenance (K and R only)
    for who in ["K", "R"]:
        if not state[f"{who}_busy"] and state["to_maintain"]:
            t_maint = maintenance_time(lcg)
            state["to_maintain"].pop(0)
            state[f"{who}_busy"] = True
            state[f"{who}_status"] = "maintaining"
            heapq.heappush(state["fel"], Event(now + t_maint, "end_maint", who))
            return

    # Log queue length
    state["stats"]["queue_lengths"].append(len(state["queue"]))

# Run one simulation with given seed
def run_simulation(seed):
    lcg = LCG(seed)
    state = reset_state()
    heapq.heappush(state["fel"], Event(SIMULATION_START, "arrival"))
    while state["fel"]:
        event = heapq.heappop(state["fel"])
        state["clock"] = event.time
        if event.type == "arrival": handle_arrival(state, lcg)
        elif event.type == "end_rent": handle_end_rent(state, event.data, lcg)
        elif event.type == "return": handle_end_return(state, event.data, lcg)
        elif event.type == "end_maint": handle_end_maint(state, event.data, lcg)
        elif event.type == "end_clean": handle_end_clean(state, event.data, lcg)
        elif event.type == "request_return":
            state["return_queue"].append(event.data)
            assign_workers(state, lcg)
    return state["stats"]

# Runs multiple simulations with different seeds
def run_full_simulation(seeds):
    results = []
    for seed in seeds:
        stats = run_simulation(seed)
        results.append({
            "Avg Wait (min)": np.mean(stats["wait_times"]),
            "Rental K (min)": np.mean(stats["rent_times_K"]) if stats["rent_times_K"] else 0,
            "Rental M (min)": np.mean(stats["rent_times_M"]) if stats["rent_times_M"] else 0,
            "Rental R (min)": np.mean(stats["rent_times_R"]) if stats["rent_times_R"] else 0,
            "Maint. (min)": np.mean(stats["maint_times"]) if stats["maint_times"] else 0,
            "Clean. (min)": np.mean(stats["clean_times"]) if stats["clean_times"] else 0,
            "Avg Queue Length": np.mean(stats["queue_lengths"]),
            "Wait >5 min (%)": (stats["wait_over_5min_count"] / stats["total_customers_with_wait"] * 100) if stats["total_customers_with_wait"] else 0,
            "Count Waits": stats["total_customers_with_wait"],
            "Count Rent K": stats["count_rent_K"],
            "Count Rent M": stats["count_rent_M"],
            "Count Rent R": stats["count_rent_R"],
            "Count Maint K": stats["count_maint_K"],
            "Count Maint R": stats["count_maint_R"],
            "Count Maint M": 0,  # Morad never does maintenance
            "Count Clean K": stats["count_clean_K"],
            "Count Clean R": stats["count_clean_R"],
            "Count Clean M": 0   # Morad never does cleaning
        })

    return pd.DataFrame(results)

# Entry point for running the full simulation
if __name__ == "__main__":
    SEEDS = [i * 17 + 123 for i in range(30)]
    df = run_full_simulation(SEEDS)
    
    # Rename for consistency
    df = df.rename(columns={
        "avg_wait": "Avg Wait (min)",
        "avg_rent_K": "Rental K (min)",
        "avg_rent_M": "Rental M (min)",
        "avg_rent_R": "Rental R (min)",
        "avg_maint": "Maint. (min)",
        "avg_clean": "Clean. (min)",
        "avg_queue_len": "Avg Queue Length",
        "wait_over_5_pct": "Wait >5 min (%)",
        "count_waits": "Count Waits",
        "count_rent_K": "Count Rent K",
        "count_rent_M": "Count Rent M",
        "count_rent_R": "Count Rent R",
        "count_maint_K": "Count Maint K",
        "count_maint_R": "Count Maint R",
        "count_clean_K": "Count Clean K",
        "count_clean_R": "Count Clean R"
    })

    # Summary row over all runs
    summary = pd.Series({
        "Avg Wait (min)": df["Avg Wait (min)"].mean(),
        "Rental K (min)": df["Rental K (min)"].mean(),
        "Rental M (min)": df["Rental M (min)"].mean(),
        "Rental R (min)": df["Rental R (min)"].mean(),
        "Maint. (min)": df["Maint. (min)"].mean(),
        "Clean. (min)": df["Clean. (min)"].mean(),
        "Avg Queue Length": df["Avg Queue Length"].mean(),
        "Wait >5 min (%)": df["Wait >5 min (%)"].mean(),
        "Count Waits": df["Count Waits"].sum(),
        "Count Rent K": df["Count Rent K"].sum(),
        "Count Rent M": df["Count Rent M"].sum(),
        "Count Rent R": df["Count Rent R"].sum(),
        "Count Maint K": df["Count Maint K"].sum(),
        "Count Maint R": df["Count Maint R"].sum(),
        "Count Maint M": df["Count Maint M"].sum(),
        "Count Clean K": df["Count Clean K"].sum(),
        "Count Clean R": df["Count Clean R"].sum(),
        "Count Clean M": df["Count Clean M"].sum()
    }, name="Average Across Runs")

    # Add missing columns if needed (for consistency with other scenarios)
    for col in [
        "Rental K (min)", "Rental M (min)", "Rental R (min)",
        "Count Rent K", "Count Rent M", "Count Rent R",
        "Count Maint K", "Count Maint R", "Count Maint M",
        "Count Clean K", "Count Clean R", "Count Clean M"
    ]:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns logically
    ordered_columns = [
        "Avg Wait (min)", "Rental K (min)", "Rental M (min)", "Rental R (min)",
        "Maint. (min)", "Clean. (min)", "Avg Queue Length", "Wait >5 min (%)",
        "Count Waits",
        "Count Rent K", "Count Rent M", "Count Rent R",
        "Count Maint M", "Count Maint K", "Count Maint R",
        "Count Clean M", "Count Clean K", "Count Clean R",
    ]
    df = df[ordered_columns]

    # Append summary to results
    df = pd.concat([df, pd.DataFrame([summary])])

    # Pretty-print settings
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.expand_frame_repr", False)

    df.to_csv("results_part3.csv", index=False)
    print("\n📊 Full Results Table (All 30 Runs + Summary):\n")
    print(df)
