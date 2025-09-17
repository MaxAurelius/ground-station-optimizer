#!/usr/bin/env python
"""
Ground station optimization for maximizing data downlink over the optimization window.

This script has been modified to serve as a comparative testbed. It runs two
optimization tests on the exact same scenario:
1. Baseline Model: Maximizes raw data throughput (`MaxDataDownlinkObjective`).
2. OCP-Enhanced Model: Maximizes utility using the Operational Complexity Penalty
   (`MaxDataWithOCPObjective`).

The output is a side-by-side comparison of the results.
"""

import logging
import datetime
from rich.console import Console
from rich.table import Table

# Make sure your new objective class is imported
from gsopt.milp_objectives import MaxDataDownlinkObjective, MaxDataWithOCPObjective
from gsopt.milp_constraints import *
from gsopt.milp_optimizer import MilpOptimizer, get_optimizer
from gsopt.models import OptimizationWindow, DataUnits
from gsopt.scenarios import ScenarioGenerator
from gsopt.utils import filter_warnings


filter_warnings()

# Set up logging
logging.basicConfig(
    datefmt='%Y-%m-%dT%H:%M:%S',
    format='%(asctime)s.%(msecs)03dZ %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Create Rich console for pretty printing
console = Console()

# ##############################################################################
# S C E N A R I O   S E T U P   (RUNS ONCE)
# ##############################################################################
console.print("\n[bold cyan]PHASE 1: GENERATING SCENARIO AND CONTACTS...[/bold cyan]")

# OPTIMIZER SELECTION
optimizer_engine = get_optimizer('cbc')  # 'scip', 'cbc', 'glpk', or 'gurobi'

# Define the optimization window
opt_start = datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc)
opt_end = opt_start + datetime.timedelta(days=365)
sim_start = opt_start
sim_end = sim_start + datetime.timedelta(days=7)

opt_window = OptimizationWindow(opt_start, opt_end, sim_start, sim_end)

# Create a scenario generator with a fixed seed for reproducibility
scengen = ScenarioGenerator(opt_window, seed="marcin-wins-2025")
scengen.add_constellation('UMBRA')
scengen.add_all_providers()
scen = scengen.sample_scenario()

console.print(scen)

# Pre-compute contacts once to ensure both tests use the same data
temp_optimizer = MilpOptimizer.from_scenario(scen, optimizer=optimizer_engine)
temp_optimizer.compute_contacts()
# Store the computed contacts to pass to our real test optimizers
computed_contacts = temp_optimizer.contacts
del temp_optimizer # Clean up

# ##############################################################################
# T E S T   1:   B A S E L I N E   M O D E L
# ##############################################################################
console.print("\n[bold magenta]PHASE 2: RUNNING TEST 1 - BASELINE MODEL (MAX DATA)...[/bold magenta]")

optimizer_baseline = MilpOptimizer.from_scenario(scen, optimizer=optimizer_engine)
# Manually set the pre-computed contacts
optimizer_baseline.contacts = computed_contacts

# Setup the optimization problem with the standard objective
optimizer_baseline.set_objective(
    MaxDataDownlinkObjective()
)

# Add Constraints
optimizer_baseline.add_constraints([
    # MaxProvidersConstraint(num_providers=3), # We remove this to see what the models do unconstrained
    MinContactDurationConstraint(min_duration=180.0),
    MaxOperationalCostConstraint(value=1e9), # Effectively unconstrained budget
    StationContactExclusionConstraint(),
    SatelliteContactExclusionConstraint(),
])

# Solve the problem
optimizer_baseline.solve()
console.print(optimizer_baseline)
solution_baseline = optimizer_baseline.get_solution()

# ##############################################################################
# T E S T   2:   O C P - E N H A N C E D   M O D E L
# ##############################################################################
console.print("\n[bold magenta]PHASE 3: RUNNING TEST 2 - OCP-ENHANCED MODEL (PRACTITIONER'S)...[/bold magenta]")

optimizer_ocp = MilpOptimizer.from_scenario(scen, optimizer=optimizer_engine)
# Manually set the same pre-computed contacts
optimizer_ocp.contacts = computed_contacts

# Setup the optimization problem with YOUR new objective
# NOTE: Ensure MaxDataWithOCPObjective is defined in gsopt/milp_objectives.py
optimizer_ocp.set_objective(
    MaxDataWithOCPObjective(P_base=500.0) # Using our P_base strategic lever
)

# Add the EXACT SAME constraints for a fair comparison
optimizer_ocp.add_constraints([
    # MaxProvidersConstraint(num_providers=3),
    MinContactDurationConstraint(min_duration=180.0),
    MaxOperationalCostConstraint(value=1e9),
    StationContactExclusionConstraint(),
    SatelliteContactExclusionConstraint(),
])

# Solve the problem
optimizer_ocp.solve()
console.print(optimizer_ocp)
solution_ocp = optimizer_ocp.get_solution()

# ##############################################################################
# V E R D I C T:   C O M P A R A T I V E   A N A L Y S I S
# ##############################################################################
console.print("\n[bold green]PHASE 4: COMPARATIVE ANALYSIS - THE VERDICT[/bold green]")

stats_baseline = solution_baseline['statistics']
stats_ocp = solution_ocp['statistics']

# Create the summary table
summary_table = Table(title="[bold]Comparative Results: Baseline vs. OCP-Enhanced[/bold]", show_header=True, header_style="bold magenta")
summary_table.add_column("Metric", style="cyan")
summary_table.add_column("Baseline Model (Max Data)", justify="right", style="white")
summary_table.add_column("OCP-Enhanced Model", justify="right", style="bold yellow")
summary_table.add_column("Delta", justify="right", style="bold")

# --- Populate Data ---
data_baseline_gb = stats_baseline['data_downlinked']['total_GB']
data_ocp_gb = stats_ocp['data_downlinked']['total_GB']
data_delta_percent = ((data_ocp_gb - data_baseline_gb) / data_baseline_gb) * 100 if data_baseline_gb else 0

cost_baseline = stats_baseline['costs']['monthly_operational']
cost_ocp = stats_ocp['costs']['monthly_operational']
cost_delta_percent = ((cost_ocp - cost_baseline) / cost_baseline) * 100 if cost_baseline else 0

stations_baseline = len(solution_baseline['selected_stations'])
stations_ocp = len(solution_ocp['selected_stations'])
stations_delta_percent = ((stations_ocp - stations_baseline) / stations_baseline) * 100 if stations_baseline else 0

providers_baseline = len(solution_baseline['selected_providers'])
providers_ocp = len(solution_ocp['selected_providers'])
providers_delta_percent = ((providers_ocp - providers_baseline) / providers_baseline) * 100 if providers_baseline else 0

# --- Add Rows to Table ---
summary_table.add_row(
    "Total Data Downlinked (GB)",
    f"{data_baseline_gb:,.2f}",
    f"{data_ocp_gb:,.2f}",
    f"{data_delta_percent:+.2f}%"
)
summary_table.add_row(
    "Monthly OpEx ($)",
    f"{cost_baseline:,.2f}",
    f"{cost_ocp:,.2f}",
    f"{cost_delta_percent:+.2f}%"
)
summary_table.add_row(
    "[bold red]Stations Activated[/bold red]",
    f"{stations_baseline}",
    f"{stations_ocp}",
    f"{stations_delta_percent:+.2f}%"
)
summary_table.add_row(
    "Providers Contracted",
    f"{providers_baseline}",
    f"{providers_ocp}",
    f"{providers_delta_percent:+.2f}%"
)
summary_table.add_row(
    "Solve Time (s)",
    f"{solution_baseline['runtime']['solve_time']:.2f}",
    f"{solution_ocp['runtime']['solve_time']:.2f}",
    "-"
)

# --- Render the final report ---
console.print(summary_table)