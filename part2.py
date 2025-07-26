import heapq
import math
import pandas as pd
import numpy as np
import random

# Simulation configuration constants
SIMULATION_START = 0        # Start time of simulation
SIMULATION_END = 600        # End of arrival generation
CLOSE_TIME = 660            # Hard close time for all activities

# -------------------------------
# Linear Congruential Generator (LCG)
# For generating pseudo-random numbers with reproducibility
# -------------------------------
class LCG:
    def __init__(self, seed=1):
        self.a = 1664525
        self.c = 1013904223
        self.m = 2 ** 32
        self.state = seed

    def random(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m

# -------------------------------
# Random Variate Generators
# -------------------------------
def generate_normal_direct(mu, sigma_squared, lcg=None):
    rand = lcg.random if lcg else random.random()
    R1 = rand()
    R2 = rand()
    Z = math.sqrt(-2 * math.log(R1)) * math.cos(2 * math.pi * R2)
    return max(0.1, mu + math.sqrt(sigma_squared) * Z)

# Time between customer arrivals
def next_arrival(lcg):
    return generate_normal_direct(30, 900, lcg)

# Rental processing time for staff K or M
def rental_time(person, lcg):
    return generate_normal_direct(14, 16, lcg) if person == "K" else generate_normal_direct(10, 25, lcg)

# Usage duration before customer returns the item
def usage_duration(lcg):
    return generate_normal_direct(60, 3600, lcg)

# Maintenance and cleaning durations
def maintenance_time(lcg):
    return generate_normal_direct(6, 16, lcg)

def cleaning_time(lcg):
    return generate_normal_direct(10, 36, lcg)

# -------------------------
# Event class for scheduling
# -------------------------
class Event:
    def __init__(self, time, type, data=None):
        self.time = time
        self.type = type
        self.data = data

    def __lt__(self, other):
        return self.time < other.time

# -------------------------
# Initialize simulation state
# -------------------------
def reset_state():
    return {
        "fel": [],                       # Future event list
        "clock": SIMULATION_START,
        "queue": [],                     # Waiting queue for rentals
        "return_queue": [],              # Queue of returns to be processed
        "to_maintain": [],               # Items needing maintenance
        "to_clean": [],                  # Items needing cleaning
        "K_busy": False,
        "M_busy": False,
        "K_status": "idle",
        "M_status": "idle",
        "active_customers": set(),
        "stats": {
            "wait_times": [],
            "queue_lengths": [],
            "rent_times_K": [],
            "rent_times_M": [],
            "maint_times": [],
            "clean_times": [],
            "timeline": [],
            "wait_over_5min_count": 0,
            "total_customers_with_wait": 0,
            "count_rent_K": 0,
            "count_rent_M": 0,
            "count_maint": 0,
            "count_clean": 0
        },
        "customer_count": 0,
    }

# -------------------------
# Event Handlers
# -------------------------

# Handle arrival of a new customer
def handle_arrival(state, lcg):
    now = state["clock"]
    cid = f"W{state['customer_count']}"  # Customer ID
    state["customer_count"] += 1
    state["queue"].append((now, cid))
    state["active_customers"].add(cid)
    state["stats"]["timeline"].append((now, "arrival", cid))

    # Schedule return if within close time
    return_time = now + usage_duration(lcg)
    if return_time < CLOSE_TIME:
        heapq.heappush(state["fel"], Event(return_time, "request_return", cid))

    assign_workers(state, lcg)

    # Schedule next arrival
    next_time = now + next_arrival(lcg)
    if next_time < SIMULATION_END:
        heapq.heappush(state["fel"], Event(next_time, "arrival"))

# Handle end of rental processing for K or M
def handle_end_rent(state, who, lcg):
    state[f"{who}_busy"] = False
    state[f"{who}_status"] = "idle"
    state["stats"]["timeline"].append((state["clock"], "end_rent", who))
    assign_workers(state, lcg)

# Handle return processing
def handle_end_return(state, customer_id, lcg):
    if customer_id in state["active_customers"]:
        state["active_customers"].remove(customer_id)
    state["to_maintain"].append(f"tool_from_{customer_id}")
    state["stats"]["timeline"].append((state["clock"], "returned", customer_id))
    state["stats"]["timeline"].append((state["clock"], "end_return", customer_id))
    for who in ["K", "M"]:
        if state[f"{who}_status"] == "returning":
            state[f"{who}_busy"] = False
            state[f"{who}_status"] = "idle"
    assign_workers(state, lcg)

# Handle end of maintenance and schedule cleaning immediately
def handle_end_maint(state, lcg):
    t_clean = cleaning_time(lcg)
    t_maint = maintenance_time(lcg)

    state["M_status"] = "cleaning"
    state["stats"]["count_maint"] += 1
    state["stats"]["maint_times"].append(t_maint)
    state["stats"]["timeline"].append((state["clock"], "end_maint", "M"))

    heapq.heappush(state["fel"], Event(state["clock"] + t_clean, "end_clean"))
    assign_workers(state, lcg)

# Handle end of cleaning process
def handle_end_clean(state, lcg):
    state["M_status"] = "idle"
    state["M_busy"] = False
    t_clean = cleaning_time(lcg)
    state["stats"]["count_clean"] += 1
    state["stats"]["clean_times"].append(t_clean)
    state["stats"]["timeline"].append((state["clock"], "end_clean", "M"))
    assign_workers(state, lcg)

# -------------------------
# Worker assignment logic
# -------------------------
def assign_workers(state, lcg):
    now = state["clock"]
    if now >= CLOSE_TIME:
        return

    # Assign returns first
    for who in ["K", "M"]:
        if not state[f"{who}_busy"] and state["return_queue"]:
            cid = state["return_queue"].pop(0)
            state[f"{who}_busy"] = True
            state[f"{who}_status"] = "returning"
            heapq.heappush(state["fel"], Event(now + 2, "return", cid))
            return

    # Assign K to rental if available
    if not state["K_busy"]:
        for i, (t0, cid) in enumerate(state["queue"]):
            if cid not in state["active_customers"]:
                continue
            wait = now - t0
            rent = rental_time("K", lcg)
            state["K_busy"] = True
            state["K_status"] = "renting"
            state["stats"]["wait_times"].append(wait)
            state["stats"]["rent_times_K"].append(rent)
            state["stats"]["total_customers_with_wait"] += 1
            state["stats"]["count_rent_K"] += 1
            if wait > 5:
                state["stats"]["wait_over_5min_count"] += 1
            state["stats"]["timeline"].append((now, "start_rent", f"K->{cid}"))
            del state["queue"][i]
            heapq.heappush(state["fel"], Event(now + rent, "end_rent", "K"))
            return

    # Assign M to rental if available
    if not state["M_busy"] and state["M_status"] not in ["maintaining", "cleaning"]:
        for i, (t0, cid) in enumerate(state["queue"]):
            if cid not in state["active_customers"]:
                continue
            wait = now - t0
            rent = rental_time("M", lcg)
            state["M_busy"] = True
            state["M_status"] = "renting"
            state["stats"]["wait_times"].append(wait)
            state["stats"]["rent_times_M"].append(rent)
            state["stats"]["total_customers_with_wait"] += 1
            state["stats"]["count_rent_M"] += 1
            if wait > 5:
                state["stats"]["wait_over_5min_count"] += 1
            state["stats"]["timeline"].append((now, "start_rent", f"M->{cid}"))
            del state["queue"][i]
            heapq.heappush(state["fel"], Event(now + rent, "end_rent", "M"))
            return

    # Assign M to maintenance if needed
    if not state["M_busy"] and state["to_maintain"]:
        t = maintenance_time(lcg)
        if now + t <= CLOSE_TIME:
            state["to_maintain"].pop(0)
            state["M_busy"] = True
            state["M_status"] = "maintaining"
            state["stats"]["maint_times"].append(t)
            state["stats"]["timeline"].append((now, "start_maint", "M"))
            heapq.heappush(state["fel"], Event(now + t, "end_maint"))
            return

    # Record current queue length
    state["stats"]["queue_lengths"].append(len(state["queue"]))

# -------------------------
# Run a single simulation replication
# -------------------------
def run_simulation(seed):
    lcg = LCG(seed)
    state = reset_state()
    heapq.heappush(state["fel"], Event(SIMULATION_START, "arrival"))

    while state["fel"]:
        event = heapq.heappop(state["fel"])
        state["clock"] = event.time

        if event.type == "arrival":
            handle_arrival(state, lcg)
        elif event.type == "end_rent":
            handle_end_rent(state, event.data, lcg)
        elif event.type == "return":
            handle_end_return(state, event.data, lcg)
        elif event.type == "end_maint":
            handle_end_maint(state, lcg)
        elif event.type == "end_clean":
            handle_end_clean(state, lcg)
        elif event.type == "request_return":
            state["return_queue"].append(event.data)
            assign_workers(state, lcg)

    return state["stats"]

# -------------------------
# Run multiple simulation replications
# -------------------------
def run_full_simulation(seeds):
    results = []
    for seed in seeds:
        stats = run_simulation(seed)
        results.append({
            "avg_wait": np.mean(stats["wait_times"]) if stats["wait_times"] else 0,
            "avg_rent_K": np.mean(stats["rent_times_K"]) if stats["rent_times_K"] else 0,
            "avg_rent_M": np.mean(stats["rent_times_M"]) if stats["rent_times_M"] else 0,
            "avg_maint": np.mean(stats["maint_times"]) if stats["maint_times"] else 0,
            "avg_clean": np.mean(stats["clean_times"]) if stats["clean_times"] else 0,
            "avg_queue_len": np.mean(stats["queue_lengths"]) if stats["queue_lengths"] else 0,
            "wait_over_5_pct": (stats["wait_over_5min_count"] / stats["total_customers_with_wait"]) * 100 if stats["total_customers_with_wait"] > 0 else 0,
            "count_waits": stats["total_customers_with_wait"],
            "count_rent_K": stats["count_rent_K"],
            "count_rent_M": stats["count_rent_M"],
            "count_maint": stats["count_maint"],
            "count_clean": stats["count_clean"]
        })
    return results

# -------------------------
# Main program execution
# -------------------------
if __name__ == "__main__":
    SEEDS = [i * 17 + 123 for i in range(30)]
    results = run_full_simulation(SEEDS)
    df = pd.DataFrame(results)
    df.index = [f"Run {i+1}" for i in range(len(df))]

    # Summary row across all runs
    summary_row = pd.Series({
        "avg_wait": df["avg_wait"].mean(),
        "avg_rent_K": df["avg_rent_K"].mean(),
        "avg_rent_M": df["avg_rent_M"].mean(),
        "avg_maint": df["avg_maint"].mean(),
        "avg_clean": df["avg_clean"].mean(),
        "avg_queue_len": df["avg_queue_len"].mean(),
        "wait_over_5_pct": df["wait_over_5_pct"].mean(),
        "count_waits": df["count_waits"].sum(),
        "count_rent_K": df["count_rent_K"].sum(),
        "count_rent_M": df["count_rent_M"].sum(),
        "count_maint": df["count_maint"].sum(),
        "count_clean": df["count_clean"].sum()
    }, name="Average Across Runs")

    df_final = pd.concat([df, pd.DataFrame([summary_row])])

    # Rename columns for report
    df_final.rename(columns={
        "avg_wait": "Avg Wait (min)",
        "avg_rent_K": "Rental K (min)",
        "avg_rent_M": "Rental M (min)",
        "avg_maint": "Maint. (min)",
        "avg_clean": "Clean. (min)",
        "avg_queue_len": "Avg Queue Length",
        "wait_over_5_pct": "Wait >5 min (%)",
        "count_waits": "Count Waits",
        "count_rent_K": "Count Rent K",
        "count_rent_M": "Count Rent M",
        "count_maint": "Count Maint.",
        "count_clean": "Count Clean."
    }, inplace=True)

    # Ensure all expected columns are present
    for col in [
        "Rental R (min)", "Count Rent R",
        "Count Maint K", "Count Maint R",
        "Count Clean K", "Count Clean R",
        "Count Maint M", "Count Clean M"
    ]:
        if col not in df_final.columns:
            df_final[col] = 0

    # Move maintenance and cleaning counts to "M" columns and drop generic ones
    df_final["Count Maint M"] = df_final["Count Maint."]
    df_final["Count Clean M"] = df_final["Count Clean."]
    df_final.drop(columns=["Count Maint.", "Count Clean."], inplace=True)

    # Final column ordering
    ordered_columns = [
        "Avg Wait (min)", "Rental K (min)", "Rental M (min)", "Rental R (min)",
        "Maint. (min)", "Clean. (min)", "Avg Queue Length", "Wait >5 min (%)",
        "Count Waits", "Count Rent K", "Count Rent M", "Count Rent R",
        "Count Maint M", "Count Maint K", "Count Maint R",
        "Count Clean M", "Count Clean K", "Count Clean R"
    ]
    df_final = df_final[ordered_columns]

    # Pandas display settings
    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.expand_frame_repr", False)

    # Save results
    df_final.to_csv("results_part2.csv", index=False)
    print("\n\U0001F4CA Full Results Table (All 30 Runs + Summary):\n")
    print(df_final)
