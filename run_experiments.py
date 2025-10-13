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
import os
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.logging import RichHandler
from rich.table import Table


# Import all necessary components from the gsopt library
from gsopt.milp_objectives import MaxDataDownlinkObjective, MaxDataWithOCPObjective
from gsopt.milp_constraints import *
from gsopt.milp_optimizer import MilpOptimizer, get_optimizer
from gsopt.models import OptimizationWindow
from gsopt.scenarios import Scenario, ScenarioGenerator
from gsopt.utils import filter_warnings


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
    """Returns a new list of the foundational constraint objects."""
    return [
        MinContactDurationConstraint(min_duration=180.0),
        StationContactExclusionConstraint(),
        SatelliteContactExclusionConstraint(),
        MinSatelliteDataDownlinkConstraint(value=1e11),
    ]

def run_optimization(scenario: Scenario, contacts: dict, objective_class, constraints: list, config: dict) -> dict:
    """Helper function to run a single optimization instance and return key results."""
    optimizer_enum = get_optimizer(config["optimizer_engine"])
    optimizer = MilpOptimizer(opt_window=scenario.opt_window, optimizer=optimizer_enum)
    
    for sat in scenario.satellites: optimizer.add_satellite(sat)
    for prov in scenario.providers: optimizer.add_provider(prov)

    optimizer.contacts = contacts
    optimizer.set_objective(objective_class)
    optimizer.add_constraints(constraints)
    optimizer.solve()
    
    status = optimizer.solver_status.lower()
    if status != 'optimal':
        logger.warning(f"Solver returned non-optimal status: {status}")
        return { "status": status, "data_gb": 0, "cost_monthly": 0, "stations": 0, "providers": 0 }

    # --- THE FIX ---
    # 1. Get the TRUE objective value (in bits) directly from the solver.
    #    This value is already correctly scaled ONCE by the objective function.
    objective_value_bits = optimizer.obj_block.obj()
    
    # 2. Convert it to Gigabytes (GB).
    #    DataUnits.GB.value is 8e9.
    correct_data_gb = objective_value_bits / (8 * 1e9)

    # 3. We still call get_solution() to conveniently get cost, station, and provider counts.
    solution = optimizer.get_solution()
    stats = solution['statistics']

    # 4. Return the results, but OVERWRITE the flawed data_gb from get_solution() with our correct one.
    return {
        "status": status,
        "data_gb": correct_data_gb, # <-- Using the correct, directly-sourced value
        "cost_monthly": stats['costs']['monthly_operational'],
        "stations": len(solution['selected_stations']),
        "providers": len(solution['selected_providers']),
    }

# def run_optimization(scenario: Scenario, contacts: dict, objective_class, constraints: list, config: dict) -> dict:
#     """Helper function to run a single optimization instance and return key results."""
#     optimizer_enum = get_optimizer(config["optimizer_engine"])
#     optimizer = MilpOptimizer(opt_window=scenario.opt_window, optimizer=optimizer_enum)
    
#     for sat in scenario.satellites: optimizer.add_satellite(sat)
#     for prov in scenario.providers: optimizer.add_provider(prov)

#     optimizer.contacts = contacts
#     optimizer.set_objective(objective_class)
#     optimizer.add_constraints(constraints)
#     optimizer.solve()
    
#     status = optimizer.solver_status.lower()
#     if status != 'optimal':
#         logger.warning(f"Solver returned non-optimal status: {status}")
#         return { "status": status, "data_gb": 0, "cost_monthly": 0, "stations": 0, "providers": 0 }

#     solution = optimizer.get_solution()
#     stats = solution['statistics']
#     return {
#         "status": status, "data_gb": stats['data_downlinked']['total_GB'],
#         "cost_monthly": stats['costs']['monthly_operational'],
#         "stations": len(solution['selected_stations']),
#         "providers": len(solution['selected_providers']),
#     }


def run_limited_provider_benchmark(full_scenario: Scenario, all_contacts: dict, config: dict) -> dict:
    """Finds the best-performing solution when restricted to only one or two providers."""
    best_result = {"data_gb": 0, "status": "untested"}
    provider_ids = [p.id for p in full_scenario.providers]
    
    # Single-Provider Runs
    for p_id in provider_ids:
        scenario_copy = copy.deepcopy(full_scenario)
        scenario_copy.providers = [p for p in scenario_copy.providers if p.id == p_id]
        active_provider_ids = {p.id for p in scenario_copy.providers}
        filtered_contacts = {k: v for k, v in all_contacts.items() if v.provider_id in active_provider_ids}
        
        # --- THE FIX ---: Create fresh constraints inside the loop.
        constraints = get_fresh_constraints()
        constraints.append(MaxOperationalCostConstraint(value=1e9))
        result = run_optimization(scenario_copy, filtered_contacts, MaxDataDownlinkObjective(), constraints, config)
        
        if result["data_gb"] > best_result["data_gb"]:
            best_result = result

    # Dual-Provider Runs
    for p1_id, p2_id in itertools.combinations(provider_ids, 2):
        scenario_copy = copy.deepcopy(full_scenario)
        scenario_copy.providers = [p for p in scenario_copy.providers if p.id in [p1_id, p2_id]]
        active_provider_ids = {p.id for p in scenario_copy.providers}
        filtered_contacts = {k: v for k, v in all_contacts.items() if v.provider_id in active_provider_ids}

        # --- THE FIX ---: Create fresh constraints inside the loop again.
        constraints = get_fresh_constraints()
        constraints.append(MaxOperationalCostConstraint(value=1e9))
        result = run_optimization(scenario_copy, filtered_contacts, MaxDataDownlinkObjective(), constraints, config)

        if result["data_gb"] > best_result["data_gb"]:
            best_result = result
            
    return best_result

def execute_pareto_analysis(trial_seed: str, config: dict):
    """Executes the Pareto analysis for a single trial."""
    cfg = config["pareto"]
    logger.info(f"Executing Pareto analysis with {cfg['satellite_count']} satellite(s)...")
    opt_window = OptimizationWindow(datetime.datetime(2023, 1, 1), datetime.datetime(2024, 1, 1), datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 8))
    scengen = ScenarioGenerator(opt_window, seed=trial_seed)
    scengen.add_random_satellites(cfg["satellite_count"])
    scengen.add_all_providers()
    scenario = scengen.sample_scenario()
    
    temp_opt = MilpOptimizer(opt_window=scenario.opt_window)
    for sat in scenario.satellites: temp_opt.add_satellite(sat)
    for prov in scenario.providers: temp_opt.add_provider(prov)
    temp_opt.compute_contacts()
    contacts = temp_opt.contacts
    
    trial_results = []
    for p_base in cfg["p_base_sweep"]:
        logger.info(f"Running Pareto point with P_base = ${p_base}")
        objective = MaxDataWithOCPObjective(P_base=p_base)
        # --- THE FIX ---: Always get a fresh constraint list for each run in the loop.
        constraints = get_fresh_constraints()
        constraints.append(MaxOperationalCostConstraint(value=1e9))
        result = run_optimization(scenario, contacts, objective, constraints, config)
        trial_results.append({"p_base": p_base, **result})
    return trial_results

def execute_scalability_analysis(trial_seed: str, config: dict):
    """Executes the Scalability analysis for a single trial."""
    cfg = config["scalability"]
    logger.info(f"Executing Scalability analysis...")
    trial_results = []
    opt_window = OptimizationWindow(datetime.datetime(2023, 1, 1), datetime.datetime(2024, 1, 1), datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 8))
    
    scengen_base = ScenarioGenerator(opt_window, seed=trial_seed)
    scengen_base.add_all_providers()
    base_scenario_no_sats = scengen_base.sample_scenario()

    for sat_count in cfg["satellite_counts"]:
        logger.info(f"Running Scalability point for {sat_count} satellite(s)...")
        
        scenario = copy.deepcopy(base_scenario_no_sats)
        
        scengen_sats = ScenarioGenerator(opt_window, seed=f'{trial_seed}-{sat_count}')
        scengen_sats.add_random_satellites(sat_count)
        
        sat_only_scenario = scengen_sats.sample_scenario()
        scenario.satellites = sat_only_scenario.satellites
        
        temp_opt = MilpOptimizer(opt_window=scenario.opt_window)
        for sat in scenario.satellites: temp_opt.add_satellite(sat)
        for prov in scenario.providers: temp_opt.add_provider(prov)
        temp_opt.compute_contacts()
        contacts = temp_opt.contacts
        
        # --- THE FIX ---: Create separate, fresh constraint lists for each model run.
        constraints_baseline = get_fresh_constraints()
        constraints_baseline.append(MaxOperationalCostConstraint(value=1e9))

        constraints_ocp = get_fresh_constraints()
        constraints_ocp.append(MaxOperationalCostConstraint(value=1e9))

        #res_limited = run_limited_provider_benchmark(scenario, contacts, config)
        res_baseline = run_optimization(scenario, contacts, MaxDataDownlinkObjective(), constraints_baseline, config)
        res_ocp = run_optimization(scenario, contacts, MaxDataWithOCPObjective(P_base=cfg["ocp_p_base"]), constraints_ocp, config)
        
        trial_results.append({
            "sat_count": sat_count,
            #"limited": res_limited,
            "baseline": res_baseline,
            "ocp": res_ocp
        })
    return trial_results

def execute_budget_constrained_analysis(trial_seed: str, config: dict):
    """Executes the Budget-Constrained analysis for a single trial."""
    cfg = config["budget_constrained"]
    logger.info(f"Executing Budget-Constrained analysis with {cfg['satellite_count']} satellite(s)...")
    opt_window = OptimizationWindow(datetime.datetime(2023, 1, 1), datetime.datetime(2024, 1, 1), datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 8))
    scengen = ScenarioGenerator(opt_window, seed=trial_seed)
    scengen.add_random_satellites(cfg["satellite_count"])
    scengen.add_all_providers()
    scenario = scengen.sample_scenario()

    temp_opt = MilpOptimizer(opt_window=scenario.opt_window)
    for sat in scenario.satellites: temp_opt.add_satellite(sat)
    for prov in scenario.providers: temp_opt.add_provider(prov)
    temp_opt.compute_contacts()
    contacts = temp_opt.contacts

    # --- THE FIX ---: Create separate, fresh constraint lists for each call.
    constraints_baseline = get_fresh_constraints()
    constraints_baseline.append(MaxOperationalCostConstraint(value=cfg["budget_usd_per_month"]))
    
    constraints_ocp = get_fresh_constraints()
    constraints_ocp.append(MaxOperationalCostConstraint(value=cfg["budget_usd_per_month"]))

    res_baseline = run_optimization(scenario, contacts, MaxDataDownlinkObjective(), constraints_baseline, config)
    res_ocp = run_optimization(scenario, contacts, MaxDataWithOCPObjective(P_base=cfg["ocp_p_base"]), constraints_ocp, config)
    
    return {"baseline": res_baseline, "ocp": res_ocp}


def run_single_trial(trial_number: int, config: dict, progress):
    """
    Executes all active pillars for a single trial number and returns the results,
    updating the progress bar as it goes.
    """
    trial_seed = f'{config["base_seed"]}-{trial_number}'
    
    num_pillars_active = sum(1 for pillar in ["pareto", "scalability", "budget_constrained"] if config.get(pillar, {}).get("active", False))
    trial_task = progress.add_task(f"[green]Trial {trial_number + 1}", total=num_pillars_active)
    
    trial_results = {}
    try:
        if config.get("pareto", {}).get("active", False):
            progress.update(trial_task, description=f"[green]Trial {trial_number + 1}: Pareto")
            trial_results["pareto"] = execute_pareto_analysis(trial_seed, config)
            progress.update(trial_task, advance=1)
        
        if config.get("scalability", {}).get("active", False):
            progress.update(trial_task, description=f"[green]Trial {trial_number + 1}: Scalability")
            trial_results["scalability"] = execute_scalability_analysis(trial_seed, config)
            progress.update(trial_task, advance=1)

        if config.get("budget_constrained", {}).get("active", False):
            progress.update(trial_task, description=f"[green]Trial {trial_number + 1}: Budget")
            trial_results["budget"] = execute_budget_constrained_analysis(trial_seed, config)
            progress.update(trial_task, advance=1)

        progress.remove_task(trial_task)
        return trial_results

    except Exception as e:
        logger.error(f"TRIAL {trial_number + 1} FAILED with an unhandled error: {e}", exc_info=True)
        progress.remove_task(trial_task)
        return None

def process_and_display_results(all_raw_results: dict):
    """Processes raw data from all trials and prints summary tables for all executed pillars."""
    
    # PILLAR 1: PARETO
    if all_raw_results.get("pareto") and any(all_raw_results["pareto"]):
        console.print("\n[bold]Aggregated Results: Pillar 1 - Pareto Frontier[/bold]")
        pareto_data = [item for trial in all_raw_results["pareto"] if trial is not None for item in trial if item is not None]

        if pareto_data:
            df_pareto = pd.DataFrame(pareto_data)
            df_pareto_succ = df_pareto[df_pareto['status'] == 'optimal']
            if not df_pareto_succ.empty:
                pareto_summary = df_pareto_succ.groupby('p_base').mean(numeric_only=True)
                pareto_total_counts = df_pareto.groupby('p_base').size()
                pareto_success_counts = df_pareto_succ.groupby('p_base').size()
                pareto_success_rate = (pareto_success_counts / pareto_total_counts * 100).fillna(0)
                
                table = Table(title=f"Aggregated Pareto Frontier (N={CONFIG['num_trials']} trials)")
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
            console.print("[yellow]No Pareto data to process (all trials may have failed).[/yellow]")

    # PILLAR 2: SCALABILITY
    if all_raw_results.get("scalability") and any(all_raw_results["scalability"]):
        console.print("\n[bold]Aggregated Results: Pillar 2 - Scalability Analysis[/bold]")
        scalability_data = [item for trial in all_raw_results["scalability"] if trial is not None for item in trial]
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
            console.print("[yellow]No Scalability data to process (all trials may have failed).[/yellow]")
    
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
            console.print("[yellow]No Budget data to process (all trials may have failed).[/yellow]")

# --- Main Execution Block ---
if __name__ == "__main__":
    # Load Configuration
    try:
        with open('config.json', 'r') as f:
            CONFIG = json.load(f)
    except FileNotFoundError:
        print("[bold red]Error: config.json not found. Please create it.[/bold red]")
        exit()

    # Setup Logging
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    logger = logging.getLogger(__name__)
    filter_warnings()
    
    if os.path.exists(CONFIG["output_filename"]):
        try:
            with open(CONFIG["output_filename"], 'r') as f:
                saved_data = json.load(f)
                all_results = saved_data.get("raw_trial_data", {"pareto": [], "scalability": [], "budget": []})
            logger.info(f"Found existing results file. Resuming session...")
        except (json.JSONDecodeError, KeyError):
            all_results = {"pareto": [], "scalability": [], "budget": []}
    else:
        all_results = {"pareto": [], "scalability": [], "budget": []}

    # Use a robust key for checking completion
    active_pillars = [k for k, v_list in all_results.items() if v_list and any(v is not None for v in v_list)]
    completed_trials = len(all_results[active_pillars[0]]) if active_pillars else 0
    logger.info(f"Session Status: {completed_trials} of {CONFIG['num_trials']} trials already complete.")
    
    # --- Setup and run with Progress Bar ---
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
    )

    with progress:
        overall_task = progress.add_task("[red]Overall Progress", total=CONFIG['num_trials'])
        progress.update(overall_task, advance=completed_trials)
        
        # --- Serial Execution Loop ---
        for i in range(completed_trials, CONFIG['num_trials']):
            result = run_single_trial(i, CONFIG, progress)
            
            if result:
                if "pareto" in result: all_results["pareto"].append(result["pareto"])
                if "scalability" in result: all_results["scalability"].append(result["scalability"])
                if "budget" in result: all_results["budget"].append(result["budget"])
            else:
                # Append placeholders for failed trials
                if CONFIG.get("pareto", {}).get("active", False): all_results["pareto"].append(None)
                if CONFIG.get("scalability", {}).get("active", False): all_results["scalability"].append(None)
                if CONFIG.get("budget_constrained", {}).get("active", False): all_results["budget"].append(None)

            progress.update(overall_task, advance=1)

            try:
                with open(CONFIG["output_filename"], 'w') as f:
                    json.dump({"config": CONFIG, "raw_trial_data": all_results}, f, indent=4)
                logger.info(f"Successfully saved progress to {CONFIG['output_filename']}")
            except Exception as e:
                logger.error(f"Error saving progress after trial {i+1}: {e}")

    # --- Final Aggregation and Display ---
    console.print("\n[bold yellow]All trials complete. Aggregating and displaying final results...[/bold yellow]")
    process_and_display_results(all_results)
