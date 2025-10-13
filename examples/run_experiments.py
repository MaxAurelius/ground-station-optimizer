#!/usr/bin/env python
"""
Systematic Test Harness for the Ground Station Optimizer Paper.

This script executes the three-pillar validation framework to generate the
comparative data needed for the research paper. This definitive version
incorporates all fixes, including statistical robustness (multi-trial runs),
data persistence (JSON output), and transparent failure handling (success rate).

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
from multiprocessing import Pool


# --- Setup ---
# Load configuration from external file
with open('config.json', 'r') as f:
    CONFIG = json.load(f)

# Set up logging to both console and a file
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(CONFIG['log_filename'])
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

console = Console()

def get_fresh_constraints():
    """Returns the foundational constraints for an experiment."""
    return [
        MinContactDurationConstraint(min_duration=180.0),
        StationContactExclusionConstraint(),
        SatelliteContactExclusionConstraint(),
        MinSatelliteDataDownlinkConstraint(value=1e11),
    ]


def run_optimization(scenario: Scenario, contacts: dict, objective_class, constraints: list) -> dict:
    """Helper function to run a single optimization instance and return key results."""
    # Correctly use the direct constructor, not the flawed class method
    optimizer_enum = get_optimizer(CONFIG["optimizer_engine"])
    optimizer = MilpOptimizer(opt_window=scenario.opt_window, optimizer=optimizer_enum)
    
    # Manually add the scenario components to the new optimizer instance
    for sat in scenario.satellites:
        optimizer.add_satellite(sat)
    for prov in scenario.providers:
        optimizer.add_provider(prov)

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

def run_limited_provider_benchmark(full_scenario: Scenario, all_contacts: dict, constraints: list) -> dict:
    """Finds the best-performing solution when restricted to only one or two providers."""
    best_result = {"data_gb": 0, "status": "untested"}
    provider_ids = [p.id for p in full_scenario.providers]

    # Single-Provider Runs
    for p_id in provider_ids:
        scenario_copy = copy.deepcopy(full_scenario)
        scenario_copy.providers = [p for p in scenario_copy.providers if p.id == p_id]
        active_provider_ids = {p.id for p in scenario_copy.providers}
        filtered_contacts = {k: v for k, v in all_contacts.items() if v.provider_id in active_provider_ids}
        result = run_optimization(scenario_copy, filtered_contacts, MaxDataDownlinkObjective(), constraints)
        if result["data_gb"] > best_result["data_gb"]:
            best_result = result

    # Dual-Provider Runs
    for p1_id, p2_id in itertools.combinations(provider_ids, 2):
        scenario_copy = copy.deepcopy(full_scenario)
        scenario_copy.providers = [p for p in scenario_copy.providers if p.id in [p1_id, p2_id]]
        active_provider_ids = {p.id for p in scenario_copy.providers}
        filtered_contacts = {k: v for k, v in all_contacts.items() if v.provider_id in active_provider_ids}
        result = run_optimization(scenario_copy, filtered_contacts, MaxDataDownlinkObjective(), constraints)
        if result["data_gb"] > best_result["data_gb"]:
            best_result = result
            
    return best_result

def execute_pareto_analysis(trial_seed: str):
    """Executes the Pareto analysis for a single trial."""
    cfg = CONFIG["pareto"]
    logger.info(f"Executing Pareto analysis with {cfg['satellite_count']} satellite(s)...")
    opt_window = OptimizationWindow(datetime.datetime(2025, 1, 1), datetime.datetime(2025, 1, 7), datetime.datetime(2025, 1, 1), datetime.datetime(2025, 1, 7))
    scengen = ScenarioGenerator(opt_window, seed=trial_seed)
    scengen.add_random_satellites(cfg["satellite_count"])
    scengen.add_all_providers()
    scenario = scengen.sample_scenario()
    
    # Use a temporary optimizer ONLY for contact computation
    temp_opt = MilpOptimizer(opt_window=scenario.opt_window)
    for sat in scenario.satellites: temp_opt.add_satellite(sat)
    for prov in scenario.providers: temp_opt.add_provider(prov)
    temp_opt.compute_contacts()
    contacts = temp_opt.contacts
    
    trial_results = []
    for p_base in cfg["p_base_sweep"]:
        logger.info(f"Running Pareto point with P_base = ${p_base}")
        objective = MaxDataWithOCPObjective(P_base=p_base)
        constraints = get_fresh_constraints()
        constraints.append(MaxOperationalCostConstraint(value=1e9)) # Unconstrained budget
        result = run_optimization(scenario, contacts, objective, constraints)
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
        
        optimizer = MilpOptimizer.from_scenario(scenario, optimizer=CONFIG["optimizer_engine"])
        optimizer.compute_contacts()
        contacts = optimizer.contacts
        constraints = [MinContactDurationConstraint(min_duration=180.0), StationContactExclusionConstraint(), SatelliteContactExclusionConstraint()]

        res_limited = run_limited_provider_benchmark(scenario, contacts, constraints)
        res_baseline = run_optimization(scenario, contacts, MaxDataDownlinkObjective(), constraints)
        res_ocp = run_optimization(scenario, contacts, MaxDataWithOCPObjective(P_base=cfg["ocp_p_base"]), constraints)
        
        trial_results.append({
            "sat_count": sat_count,
            "limited": res_limited,
            "baseline": res_baseline,
            "ocp": res_ocp
        })
    return trial_results

def execute_budget_constrained_analysis(trial_seed: str):
    """Executes the Budget-Constrained analysis for a single trial."""
    cfg = CONFIG["budget_constrained"]
    logger.info(f"Executing Budget-Constrained analysis with {cfg['satellite_count']} satellite(s)...")
    opt_window = OptimizationWindow(datetime.datetime(2023, 1, 1), datetime.datetime(2024, 1, 1), datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 8))
    scengen = ScenarioGenerator(opt_window, seed=trial_seed)
    scengen.add_random_satellites(cfg["satellite_count"])
    scengen.add_all_providers()
    scenario = scengen.sample_scenario()

    # Use a temporary optimizer ONLY for contact computation
    temp_opt = MilpOptimizer(opt_window=scenario.opt_window)
    for sat in scenario.satellites: temp_opt.add_satellite(sat)
    for prov in scenario.providers: temp_opt.add_provider(prov)
    temp_opt.compute_contacts()
    contacts = temp_opt.contacts

    # Create two separate, clean constraint lists
    constraints_baseline = get_fresh_constraints()
    constraints_baseline.append(MaxOperationalCostConstraint(value=cfg["budget_usd_per_month"]))
    
    constraints_ocp = get_fresh_constraints()
    constraints_ocp.append(MaxOperationalCostConstraint(value=cfg["budget_usd_per_month"]))

    res_baseline = run_optimization(scenario, contacts, MaxDataDownlinkObjective(), constraints_baseline)
    res_ocp = run_optimization(scenario, contacts, MaxDataWithOCPObjective(P_base=cfg["ocp_p_base"]), constraints_ocp)
    
    return {"baseline": res_baseline, "ocp": res_ocp}


def run_single_trial(trial_number: int, config: dict):
    """
    Executes all active pillars for a single trial number and returns the results.
    """
    trial_seed = f'{config["base_seed"]}-{trial_number}'
    logger.info(f"--- Starting Trial {trial_number + 1}/{config['num_trials']} (Seed: {trial_seed}) ---")
    
    trial_results = {}
    try:
        if config["pareto"].get("active", False):
            trial_results["pareto"] = execute_pareto_analysis(trial_seed)
        
        if config["budget_constrained"].get("active", False):
            trial_results["budget"] = execute_budget_constrained_analysis(trial_seed)

        logger.info(f"--- Finished Trial {trial_number + 1}/{config['num_trials']} ---")
        return trial_results

    except Exception as e:
        logger.error(f"TRIAL {trial_number + 1} FAILED with an unhandled error: {e}", exc_info=True)
        return None # Return None on failure

def process_and_display_results(all_raw_results: dict):
    """Processes raw data from all trials and prints summary tables for all executed pillars."""
    
    # PILLAR 1: PARETO
    if all_raw_results.get("pareto") and any(all_raw_results["pareto"]):
        console.print("\n[bold]Aggregated Results: Pillar 1 - Pareto Frontier[/bold]")
        pareto_data = [item for trial in all_raw_results["pareto"] for item in trial if item is not None]
        if pareto_data:
            df_pareto = pd.DataFrame(pareto_data)
            df_pareto_succ = df_pareto[df_pareto['status'] == 'optimal']
            if not df_pareto_succ.empty:
                pareto_summary = df_pareto_succ.groupby('p_base').mean(numeric_only=True)
                pareto_total_counts = df_pareto.groupby('p_base').size()
                pareto_success_counts = df_pareto_succ.groupby('p_base').size()
                pareto_success_rate = (pareto_success_counts / pareto_total_counts * 100).fillna(0)
                
                table = Table(title=f"Aggregated Pareto Frontier (N={len(all_raw_results['pareto'])} trials)")
                table.add_column("P_base ($)", style="cyan")
                table.add_column("Success Rate", justify="right")
                table.add_column("Avg. Stations", justify="right")
                table.add_column("Avg. Data (GB)", justify="right", style="bold yellow")
                for p_base, row in pareto_summary.iterrows():
                    table.add_row(f"{p_base}", f"{pareto_success_rate.get(p_base, 0):.0f}%", f"{row['stations']:.1f}", f"{row['data_gb']:,.2f}")
                console.print(table)
            else:
                console.print("[red]No successful Pareto runs to analyze.[/red]")
        else:
            console.print("[yellow]No Pareto data to process.[/yellow]")

    # PILLAR 2: SCALABILITY
    if all_raw_results.get("scalability") and any(all_raw_results["scalability"]):
        console.print("\n[bold]Aggregated Results: Pillar 2 - Scalability Analysis[/bold]")
        scalability_data = [item for trial in all_raw_results["scalability"] for item in trial if item is not None]
        if scalability_data:
            df_scalability_raw = pd.DataFrame(scalability_data)
            df_scalability = pd.DataFrame()
            for model in ['limited', 'baseline', 'ocp']:
                if model in df_scalability_raw.columns:
                    unpacked = df_scalability_raw[model].apply(pd.Series).add_prefix(f'{model}_')
                    df_scalability = pd.concat([df_scalability, unpacked], axis=1)
            df_scalability['sat_count'] = df_scalability_raw['sat_count']

            table = Table(title=f"Aggregated Scalability Analysis (N={len(all_raw_results['scalability'])} trials)")
            table.add_column("Satellites", style="cyan")
            table.add_column("Model", style="white")
            table.add_column("Success Rate", justify="right")
            table.add_column("Avg. Stations (σ)", justify="right")
            table.add_column("Avg. Data (GB) (σ)", justify="right", style="bold yellow")

            for count in sorted(df_scalability['sat_count'].unique()):
                df_group = df_scalability[df_scalability['sat_count'] == count]
                for model in ['limited', 'baseline', 'ocp']:
                    if f'{model}_status' in df_group.columns:
                        df_model_succ = df_group[df_group[f'{model}_status'] == 'optimal']
                        success_rate = (len(df_model_succ) / len(df_group)) * 100 if len(df_group) > 0 else 0
                        mean = df_model_succ.mean(numeric_only=True)
                        std = df_model_succ.std(numeric_only=True)
                        table.add_row(f"{count}", model.replace('_', '-').title(), f"{success_rate:.0f}%", 
                                      f"{mean.get(f'{model}_stations', 0):.1f} (σ={std.get(f'{model}_stations', 0):.1f})",
                                      f"{mean.get(f'{model}_data_gb', 0):,.2f} (σ={std.get(f'{model}_data_gb', 0):.2f})",
                                      end_section=(model=='ocp'))
            console.print(table)
        else:
            console.print("[yellow]No Scalability data to process.[/yellow]")
    
    # PILLAR 3: BUDGET
    if all_raw_results.get("budget") and any(all_raw_results["budget"]):
        console.print("\n[bold]Aggregated Results: Pillar 3 - Budget-Constrained Analysis[/bold]")
        budget_data = [res for res in all_raw_results["budget"] if res is not None]
        if budget_data:
            df_budget_raw = pd.DataFrame(budget_data)
            df_budget = pd.concat([df_budget_raw['baseline'].apply(pd.Series).add_prefix('baseline_'), df_budget_raw['ocp'].apply(pd.Series).add_prefix('ocp_')], axis=1)
            
            table = Table(title=f"Aggregated Budget-Constrained Analysis (N={len(all_raw_results['budget'])} trials)")
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
        else:
            console.print("[yellow]No Budget data to process.[/yellow]")

if __name__ == "__main__":
    # --- Load Configuration and Set Up Logging ---
    try:
        with open('config.json', 'r') as f:
            CONFIG = json.load(f)
    except FileNotFoundError:
        console.print("[bold red]Error: config.json not found. Please create it.[/bold red]")
        exit()

    # Set up logging to both console and a file
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(CONFIG['log_filename'])
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    # Console handler for rich output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    
    # --- Load existing results to resume from checkpoint ---
    if os.path.exists(CONFIG["output_filename"]):
        try:
            with open(CONFIG["output_filename"], 'r') as f:
                saved_data = json.load(f)
                all_results = saved_data.get("raw_trial_data", {"pareto": [], "budget": []})
            logger.info(f"Found existing results file. Resuming session...")
        except (json.JSONDecodeError, KeyError):
            all_results = {"pareto": [], "budget": []}
    else:
        all_results = {"pareto": [], "budget": []}

    completed_trials = len(all_results.get("pareto", []))
    logger.info(f"Session Status: {completed_trials} of {CONFIG['num_trials']} trials already complete.")
    
    # --- Serial Execution Loop ---
    for i in range(completed_trials, CONFIG['num_trials']):
        result = run_single_trial(i, CONFIG)
        
        # Process and save result for the completed trial
        if result:
            if "pareto" in result: all_results["pareto"].append(result["pareto"])
            if "budget" in result: all_results["budget"].append(result["budget"])
        else:
            # Append placeholders for failed trials to keep counts consistent for resume logic
            if CONFIG["pareto"]["active"]: all_results["pareto"].append(None)
            if CONFIG["budget_constrained"]["active"]: all_results["budget"].append(None)

        try:
            # Clean up None values before saving to keep file clean
            # This makes the final dataset contain only successful trial data
            clean_results = {
                "pareto": [r for r in all_results["pareto"] if r is not None],
                "budget": [r for r in all_results["budget"] if r is not None]
            }
            with open(CONFIG["output_filename"], 'w') as f:
                json.dump({"config": CONFIG, "raw_trial_data": clean_results}, f, indent=4)
            logger.info(f"Successfully saved progress to {CONFIG['output_filename']}")
        except Exception as e:
            logger.error(f"Error saving progress after trial {i+1}: {e}")

    # --- Final Aggregation and Display ---
    console.print("\n[bold yellow]All trials complete. Aggregating and displaying final results...[/bold yellow]")
    process_and_display_results(all_results)
