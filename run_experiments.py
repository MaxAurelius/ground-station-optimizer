#!/usr/bin/env python
"""
Systematic Test Harness for the Ground Station Optimizer Paper.

This script executes the three-pillar validation framework to generate the
comparative data needed for the research paper. This definitive version
incorporates all fixes, including the Pyomo object reuse error.

Pillar 1: Pareto Frontier Analysis
Pillar 2: Scalability Analysis
Pillar 3: Budget-Constrained Analysis
"""

import logging
import datetime
import itertools
import copy
import json
import pandas as pd
from rich.console import Console
from rich.table import Table
import os

# Import all necessary components from the gsopt library
from gsopt.milp_objectives import MaxDataDownlinkObjective, MaxDataWithOCPObjective
from gsopt.milp_constraints import *
from gsopt.milp_optimizer import MilpOptimizer, get_optimizer
from gsopt.models import OptimizationWindow
from gsopt.scenarios import Scenario, ScenarioGenerator
from gsopt.utils import filter_warnings

# --- Configuration ---
CONFIG = {
    "optimizer_engine": "cbc",
    "num_trials": 1,
    "base_seed": "final-validated-results-2025",
    "output_filename": "final_results_pareto_n100_sat5.json",
    "pareto": {
        "satellite_count": 5,
        "p_base_sweep": [0, 100, 250, 500, 1000, 2500],
    },
    "scalability": {
        "satellite_counts": [1],
        "ocp_p_base": 500.0,
    },
    "budget_constrained": {
        "satellite_count": 1,
        "budget_usd_per_month": 150000.0,
        "ocp_p_base": 500.0,
    }
}

# --- Setup ---
filter_warnings()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
console = Console()


def get_fresh_constraints():
    """
    Returns a new list of the standard, baseline constraint objects.
    This set is used for unconstrained performance tests.
    """
    return [
        # --- Physical & Operational Reality Constraints ---
        MinContactDurationConstraint(min_duration=180.0),
        StationContactExclusionConstraint(),
        SatelliteContactExclusionConstraint(),
        
        # --- Constraints for a Well-Posed Problem ---
        # Prevents degenerate (zero-data) solutions.
        MinSatelliteDataDownlinkConstraint(value=1e11), # e.g., ~10 GB/day
        
        # Provides a high budget ceiling for "unconstrained" runs.
        MaxOperationalCostConstraint(value=1e9)
    ]

def run_optimization(scenario: Scenario, contacts: dict, objective_class, constraints: list) -> dict:
    """Helper function to run a single optimization instance and return key results."""
    optimizer_enum = get_optimizer(CONFIG["optimizer_engine"])
    optimizer = MilpOptimizer.from_scenario(scenario, optimizer=optimizer_enum)
    optimizer.contacts = contacts
    optimizer.set_objective(objective_class)
    optimizer.add_constraints(constraints)
    optimizer.solve()
    
    status = optimizer.solver_status.lower()
    if status != 'optimal':
        logger.warning(f"Solver returned non-optimal status: {status}")
        return { "status": status, "data_gb": 0, "cost_monthly": 0, "stations": 0, "providers": 0 }

    solution = optimizer.get_solution()
    stats = solution['statistics']
    return {
        "status": status, "data_gb": stats['data_downlinked']['total_GB'],
        "cost_monthly": stats['costs']['monthly_operational'],
        "stations": len(solution['selected_stations']),
        "providers": len(solution['selected_providers']),
    }

def run_limited_provider_benchmark(full_scenario: Scenario, all_contacts: dict) -> dict:
    """Finds the best-performing solution when restricted to only one or two providers."""
    best_result = {"data_gb": 0, "status": "untested"}
    provider_ids = [p.id for p in full_scenario.providers]

    # Single-Provider Runs
    for p_id in provider_ids:
        scenario_copy = copy.deepcopy(full_scenario)
        scenario_copy.providers = [p for p in scenario_copy.providers if p.id == p_id]
        active_provider_ids = {p.id for p in scenario_copy.providers}
        filtered_contacts = {k: v for k, v in all_contacts.items() if v.provider_id in active_provider_ids}
        result = run_optimization(scenario_copy, filtered_contacts, MaxDataDownlinkObjective(), get_fresh_constraints())
        if result["data_gb"] > best_result["data_gb"]:
            best_result = result

    # Dual-Provider Runs
    for p1_id, p2_id in itertools.combinations(provider_ids, 2):
        scenario_copy = copy.deepcopy(full_scenario)
        scenario_copy.providers = [p for p in scenario_copy.providers if p.id in [p1_id, p2_id]]
        active_provider_ids = {p.id for p in scenario_copy.providers}
        filtered_contacts = {k: v for k, v in all_contacts.items() if v.provider_id in active_provider_ids}
        result = run_optimization(scenario_copy, filtered_contacts, MaxDataDownlinkObjective(), get_fresh_constraints())
        if result["data_gb"] > best_result["data_gb"]:
            best_result = result
            
    return best_result

# --- Pillar Execution Functions ---
def execute_pareto_analysis(trial_seed: str):
    cfg = CONFIG["pareto"]
    opt_window = OptimizationWindow(datetime.datetime(2023, 1, 1), datetime.datetime(2024, 1, 1), datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 8))
    scengen = ScenarioGenerator(opt_window, seed=trial_seed)
    scengen.add_random_satellites(cfg["satellite_count"])
    scengen.add_all_providers()
    scenario = scengen.sample_scenario()
    
    optimizer = MilpOptimizer.from_scenario(scenario, optimizer=get_optimizer(CONFIG["optimizer_engine"]))
    optimizer.compute_contacts()
    contacts = optimizer.contacts
    
    trial_results = []
    for p_base in cfg["p_base_sweep"]:
        objective = MaxDataWithOCPObjective(P_base=p_base)
        result = run_optimization(scenario, contacts, objective, get_fresh_constraints())
        trial_results.append({"p_base": p_base, **result})
    return trial_results

def execute_scalability_analysis(trial_seed: str):
    cfg = CONFIG["scalability"]
    trial_results = []
    opt_window = OptimizationWindow(datetime.datetime(2023, 1, 1), datetime.datetime(2024, 1, 1), datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 8))
    
    scengen_base = ScenarioGenerator(opt_window, seed=trial_seed)
    scengen_base.add_all_providers()
    base_scenario_no_sats = scengen_base.sample_scenario()

    for sat_count in cfg["satellite_counts"]:
        scenario = copy.deepcopy(base_scenario_no_sats)
        scengen_sats = ScenarioGenerator(opt_window, seed=f'{trial_seed}-{sat_count}')
        scengen_sats.add_random_satellites(sat_count)
        scenario.satellites = scengen_sats.satellites
        
        optimizer = MilpOptimizer.from_scenario(scenario, optimizer=get_optimizer(CONFIG["optimizer_engine"]))
        optimizer.compute_contacts()
        contacts = optimizer.contacts

        res_limited = run_limited_provider_benchmark(scenario, contacts)
        res_baseline = run_optimization(scenario, contacts, MaxDataDownlinkObjective(), get_fresh_constraints())
        res_ocp = run_optimization(scenario, contacts, MaxDataWithOCPObjective(P_base=cfg["ocp_p_base"]), get_fresh_constraints())
        
        trial_results.append({
            "sat_count": sat_count, "limited": res_limited,
            "baseline": res_baseline, "ocp": res_ocp
        })
    return trial_results

def execute_budget_constrained_analysis(trial_seed: str):
    cfg = CONFIG["budget_constrained"]
    opt_window = OptimizationWindow(datetime.datetime(2023, 1, 1), datetime.datetime(2024, 1, 1), datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 8))
    scengen = ScenarioGenerator(opt_window, seed=trial_seed)
    scengen.add_random_satellites(cfg["satellite_count"])
    scengen.add_all_providers()
    scenario = scengen.sample_scenario()
    
    optimizer = MilpOptimizer.from_scenario(scenario, optimizer=get_optimizer(CONFIG["optimizer_engine"]))
    optimizer.compute_contacts()
    contacts = optimizer.contacts

    constraints = get_fresh_constraints()
    constraints.append(
        MaxOperationalCostConstraint(value=CONFIG["budget_constrained"]["budget_usd_per_month"])
        )

    res_baseline = run_optimization(scenario, contacts, MaxDataDownlinkObjective(), constraints)
    
    # Need a separate fresh list for the next run
    constraints_ocp = get_fresh_constraints()
    constraints_ocp.append(MaxOperationalCostConstraint(value=cfg["budget_usd_per_month"]))
    res_ocp = run_optimization(scenario, contacts, MaxDataWithOCPObjective(P_base=cfg["ocp_p_base"]), constraints_ocp)
    
    return {"baseline": res_baseline, "ocp": res_ocp}

# --- Post-Processing and Display Functions (omitted for brevity, no changes from previous version) ---
# --- Post-Processing and Display Functions ---
def process_and_display_results(all_raw_results: dict):
    """Processes raw data from all trials and prints summary tables for executed pillars."""
    
    # --- FIX: Check if data exists before processing each pillar ---

    # PILLAR 1: PARETO
    if all_raw_results["pareto"] and all_raw_results["pareto"][0]:
        console.print("\n[bold]Aggregated Results: Pillar 1 - Pareto Frontier[/bold]")
        df_pareto = pd.DataFrame([item for trial in all_raw_results["pareto"] for item in trial])
        df_pareto_succ = df_pareto[df_pareto['status'] == 'optimal']
        pareto_summary = df_pareto_succ.groupby('p_base').mean(numeric_only=True)
        pareto_success = (df_pareto_succ.groupby('p_base').size() / df_pareto.groupby('p_base').size() * 100).fillna(0)
        
        table = Table(title=f"Aggregated Pareto Frontier (N={CONFIG['num_trials']} trials)")
        table.add_column("P_base ($)", style="cyan")
        table.add_column("Success Rate", justify="right")
        table.add_column("Avg. Stations", justify="right")
        table.add_column("Avg. Data (GB)", justify="right", style="bold yellow")
        for p_base, row in pareto_summary.iterrows():
            table.add_row(f"{p_base}", f"{pareto_success.get(p_base, 0):.0f}%", f"{row['stations']:.1f}", f"{row['data_gb']:,.2f}")
        console.print(table)

    # PILLAR 2: SCALABILITY
    if all_raw_results["scalability"] and all_raw_results["scalability"][0]:
        console.print("\n[bold]Aggregated Results: Pillar 2 - Scalability Analysis[/bold]")
        df_scalability_raw = pd.DataFrame([item for trial in all_raw_results["scalability"] for item in trial])
        df_scalability = pd.DataFrame()
        for model in ['limited', 'baseline', 'ocp']:
            unpacked = df_scalability_raw[model].apply(pd.Series).add_prefix(f'{model}_')
            df_scalability = pd.concat([df_scalability, unpacked], axis=1)
        df_scalability['sat_count'] = df_scalability_raw['sat_count']

        table = Table(title=f"Aggregated Scalability Analysis (N={CONFIG['num_trials']} trials)")
        table.add_column("Satellites", style="cyan")
        table.add_column("Model", style="white")
        table.add_column("Success Rate", justify="right")
        table.add_column("Avg. Stations (σ)", justify="right")
        table.add_column("Avg. Data (GB) (σ)", justify="right", style="bold yellow")

        for count in sorted(df_scalability['sat_count'].unique()):
            df_group = df_scalability[df_scalability['sat_count'] == count]
            for model in ['limited', 'baseline', 'ocp']:
                df_model_succ = df_group[df_group[f'{model}_status'] == 'optimal']
                success_rate = (len(df_model_succ) / len(df_group)) * 100 if len(df_group) > 0 else 0
                mean = df_model_succ.mean(numeric_only=True)
                std = df_model_succ.std(numeric_only=True)
                table.add_row(f"{count}", model.replace('_', '-').title(), f"{success_rate:.0f}%", 
                              f"{mean.get(f'{model}_stations', 0):.1f} (σ={std.get(f'{model}_stations', 0):.1f})",
                              f"{mean.get(f'{model}_data_gb', 0):,.2f} (σ={std.get(f'{model}_data_gb', 0):.2f})",
                              end_section=(model=='ocp'))
        console.print(table)
    
    # PILLAR 3: BUDGET
    if all_raw_results["budget"] and all_raw_results["budget"][0]:
        console.print("\n[bold]Aggregated Results: Pillar 3 - Budget-Constrained Analysis[/bold]")
        df_budget_raw = pd.DataFrame(all_raw_results["budget"])
        df_budget = pd.concat([df_budget_raw['baseline'].apply(pd.Series).add_prefix('baseline_'), df_budget_raw['ocp'].apply(pd.Series).add_prefix('ocp_')], axis=1)
        
        table = Table(title=f"Aggregated Budget-Constrained Analysis (N={CONFIG['num_trials']} trials)")
        table.add_column("Model", style="cyan")
        table.add_column("Success Rate", justify="right")
        table.add_column("Avg. Stations (σ)", justify="right")
        table.add_column("Avg. Data (GB) (σ)", justify="right", style="bold yellow")
        
        for model in ['baseline', 'ocp']:
            df_model_succ = df_budget[df_budget[f'{model}_status'] == 'optimal']
            success_rate = (len(df_model_succ) / len(df_budget)) * 100 if len(df_budget) > 0 else 0
            mean = df_model_succ.mean(numeric_only=True)
            std = df_model_succ.std(numeric_only=True)
            table.add_row(model.replace('_', '-').title(), f"{success_rate:.0f}%",
                          f"{mean.get(f'{model}_stations', 0):.1f} (σ={std.get(f'{model}_stations', 0):.1f})",
                          f"{mean.get(f'{model}_data_gb', 0):,.2f} (σ={std.get(f'{model}_data_gb', 0):.2f})")
        console.print(table)    # This function remains unchanged from the previous version.
    # It correctly calculates stats based on the returned data.
    # PILLAR 1: PARETO
    df_pareto = pd.DataFrame([item for trial in all_raw_results["pareto"] for item in trial])
    df_pareto_succ = df_pareto[df_pareto['status'] == 'optimal']
    pareto_summary = df_pareto_succ.groupby('p_base').mean(numeric_only=True)
    pareto_success = (df_pareto_succ.groupby('p_base').size() / df_pareto.groupby('p_base').size() * 100).fillna(0)
    
    table = Table(title=f"Aggregated Pareto Frontier (N={CONFIG['num_trials']} trials)")
    table.add_column("P_base ($)", style="cyan")
    table.add_column("Success Rate", justify="right")
    table.add_column("Avg. Stations", justify="right")
    table.add_column("Avg. Data (GB)", justify="right", style="bold yellow")
    for p_base, row in pareto_summary.iterrows():
        table.add_row(f"{p_base}", f"{pareto_success.get(p_base, 0):.0f}%", f"{row['stations']:.1f}", f"{row['data_gb']:,.2f}")
    console.print(table)

    # PILLAR 2: SCALABILITY
    df_scalability_raw = pd.DataFrame([item for trial in all_raw_results["scalability"] for item in trial])
    df_scalability = pd.DataFrame()
    for model in ['limited', 'baseline', 'ocp']:
        unpacked = df_scalability_raw[model].apply(pd.Series).add_prefix(f'{model}_')
        df_scalability = pd.concat([df_scalability, unpacked], axis=1)
    df_scalability['sat_count'] = df_scalability_raw['sat_count']

    table = Table(title=f"Aggregated Scalability Analysis (N={CONFIG['num_trials']} trials)")
    table.add_column("Satellites", style="cyan")
    table.add_column("Model", style="white")
    table.add_column("Success Rate", justify="right")
    table.add_column("Avg. Stations (σ)", justify="right")
    table.add_column("Avg. Data (GB) (σ)", justify="right", style="bold yellow")

    for count in sorted(df_scalability['sat_count'].unique()):
        df_group = df_scalability[df_scalability['sat_count'] == count]
        for model in ['limited', 'baseline', 'ocp']:
            df_model_succ = df_group[df_group[f'{model}_status'] == 'optimal']
            success_rate = (len(df_model_succ) / len(df_group)) * 100 if len(df_group) > 0 else 0
            mean = df_model_succ.mean(numeric_only=True)
            std = df_model_succ.std(numeric_only=True)
            table.add_row(f"{count}", model.replace('_', '-').title(), f"{success_rate:.0f}%", 
                          f"{mean.get(f'{model}_stations', 0):.1f} (σ={std.get(f'{model}_stations', 0):.1f})",
                          f"{mean.get(f'{model}_data_gb', 0):,.2f} (σ={std.get(f'{model}_data_gb', 0):.2f})",
                          end_section=(model=='ocp'))
    console.print(table)
    
    # PILLAR 3: BUDGET
    df_budget_raw = pd.DataFrame(all_raw_results["budget"])
    df_budget = pd.concat([df_budget_raw['baseline'].apply(pd.Series).add_prefix('baseline_'), df_budget_raw['ocp'].apply(pd.Series).add_prefix('ocp_')], axis=1)
    
    table = Table(title=f"Aggregated Budget-Constrained Analysis (N={CONFIG['num_trials']} trials)")
    table.add_column("Model", style="cyan")
    table.add_column("Success Rate", justify="right")
    table.add_column("Avg. Stations (σ)", justify="right")
    table.add_column("Avg. Data (GB) (σ)", justify="right", style="bold yellow")
    
    for model in ['baseline', 'ocp']:
        df_model_succ = df_budget[df_budget[f'{model}_status'] == 'optimal']
        success_rate = (len(df_model_succ) / len(df_budget)) * 100 if len(df_budget) > 0 else 0
        mean = df_model_succ.mean(numeric_only=True)
        std = df_model_succ.std(numeric_only=True)
        table.add_row(model.replace('_', '-').title(), f"{success_rate:.0f}%",
                      f"{mean.get(f'{model}_stations', 0):.1f} (σ={std.get(f'{model}_stations', 0):.1f})",
                      f"{mean.get(f'{model}_data_gb', 0):,.2f} (σ={std.get(f'{model}_data_gb', 0):.2f})")
    console.print(table)

if __name__ == "__main__":
    
    # --- FIX: Load existing results to resume from checkpoint ---
    if os.path.exists(CONFIG["output_filename"]):
        try:
            console.print(f"[bold yellow]Found existing results file: {CONFIG['output_filename']}. Attempting to resume session...[/bold yellow]")
            with open(CONFIG["output_filename"], 'r') as f:
                saved_data = json.load(f)
                all_results = saved_data["raw_trial_data"]
        except (json.JSONDecodeError, KeyError):
            console.print(f"[bold red]Error: Results file is corrupted. Starting a new session.[/bold red]")
            all_results = {"pareto": [], "scalability": [], "budget": []}
    else:
        all_results = {"pareto": [], "scalability": [], "budget": []}

    # Determine how many trials have already been completed
    # We assume all pillars run in sync, so we check the length of the first one.
    completed_trials = 0
    if all_results["pareto"]:
        completed_trials = len(all_results["pareto"])
    
    console.print(f"[bold]Session Status: {completed_trials} of {CONFIG['num_trials']} trials already complete.[/bold]")

    # --- Main loop for statistical robustness, starting from the last completed trial ---
    for i in range(completed_trials, CONFIG["num_trials"]):
        trial_seed = f'{CONFIG["base_seed"]}-{i}'
        console.print(f"\n[bold green]EXECUTING TRIAL {i+1}/{CONFIG['num_trials']} (Seed: {trial_seed})...[/bold green]")
        
        try:
            all_results["pareto"].append(execute_pareto_analysis(trial_seed))
        except Exception as e:
            logger.error(f"TRIAL {i+1} FAILED in Pareto Analysis: {e}")

        # The Scalability pillar is disabled for now
        # try:
        #     all_results["scalability"].append(execute_scalability_analysis(trial_seed))
        # except Exception as e:
        #     logger.error(f"TRIAL {i+1} FAILED in Scalability Analysis: {e}")

        try:
            all_results["budget"].append(execute_budget_constrained_analysis(trial_seed))
        except Exception as e:
            logger.error(f"TRIAL {i+1} FAILED in Budget Analysis: {e}")
        
        # Save results incrementally after each trial
        try:
            with open(CONFIG["output_filename"], 'w') as f:
                json.dump({"config": CONFIG, "raw_trial_data": all_results}, f, indent=4)
            logger.info(f"Successfully saved progress for trial {i+1} to {CONFIG['output_filename']}")
        except Exception as e:
            logger.error(f"Failed to save progress for trial {i+1}. Reason: {e}")


    # --- Final Aggregation and Display ---
    console.print("\n[bold yellow]All trials complete. Aggregating and displaying final results...[/bold yellow]")
    process_and_display_results(all_results)
    console.print(f"\n[bold green]Final results saved in {CONFIG['output_filename']}[/bold green]")
