import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys
from pathlib import Path

def format_data_size(gb_value: float) -> str:
    """Converts a value in Gigabytes to a readable string in TB or PB."""
    if gb_value >= 1_000_000:
        return f"{gb_value / 1_000_000:.2f} PB"
    if gb_value >= 1_000:
        return f"{gb_value / 1_000:,.1f} TB"
    return f"{gb_value:,.0f} GB"

def process_single_file(file_path: Path):
    """
    Loads a single simulation file and returns the mean optimal results
    for both stations and data throughput.
    """
    try:
        with open(file_path, 'r') as f:
            results_data = json.load(f)

        raw_data_container = results_data.get('raw_trial_data')
        if not raw_data_container:
            print(f"WARNING: 'raw_trial_data' key missing in {file_path}. Skipping.")
            return None

        budget_raw_data = raw_data_container.get('budget')
        if not budget_raw_data:
            print(f"WARNING: No 'budget' data found in {file_path}. Skipping.")
            return None

        flat_data = []
        for i, trial_result in enumerate(budget_raw_data):
            # Praetorian Guard: Reject corrupted 'null' entries.
            if trial_result is None:
                continue

            if trial_result.get('baseline') and trial_result.get('ocp'):
                flat_data.append({'model': 'Baseline', **trial_result['baseline']})
                flat_data.append({'model': 'OCP-Enhanced', **trial_result['ocp']})

        if not flat_data:
            return None

        df = pd.DataFrame(flat_data)
        df_optimal = df[df['status'] == 'optimal']

        if df_optimal.empty:
            return None
            
        # --- MODIFIED: Extract both stations and data_gb ---
        summary = df_optimal.groupby('model')[['stations', 'data_gb']].mean()
        return summary.to_dict()

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR: Could not process file {file_path}. Reason: {e}. Skipping.")
        return None


def generate_multi_budget_plot(simulation_files: dict):
    """
    Generates a publication-quality bar chart comparing model performance
    across multiple budget constraints.
    """
    plot_data = []
    for budget_label, file_path in simulation_files.items():
        results = process_single_file(Path(file_path))
        if results:
            # --- MODIFIED: Append both metrics for each model ---
            plot_data.append({
                'budget': budget_label,
                'Baseline_stations': results.get('stations', {}).get('Baseline', 0),
                'OCP-Enhanced_stations': results.get('stations', {}).get('OCP-Enhanced', 0),
                'Baseline_data_gb': results.get('data_gb', {}).get('Baseline', 0),
                'OCP-Enhanced_data_gb': results.get('data_gb', {}).get('OCP-Enhanced', 0),
            })

    if not plot_data:
        print("FATAL: No valid data could be processed from any of the provided files. Mission aborted.")
        sys.exit(1)

    df = pd.DataFrame(plot_data).set_index('budget')
    
    # Separate dataframes for plotting and annotation for clarity
    df_stations = df[['Baseline_stations', 'OCP-Enhanced_stations']]
    df_stations.columns = ['Baseline', 'OCP-Enhanced'] # Clean column names for legend
    df_data = df[['Baseline_data_gb', 'OCP-Enhanced_data_gb']]
    
    # --- VISUAL STRATEGY: GRAVITAS AND AUTHORITY ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'axes.titlesize': 16,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.dpi': 300
    })

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot the data with monochrome colors and defined hatches for clarity
    df_stations.plot(
        kind='bar',
        ax=ax,
        color=['#444444', '#999999'], # Dark grey for Baseline, lighter for OCP
        width=0.8,
        edgecolor='black'
    )

    # Apply distinct patterns (hatching) for black-and-white print compatibility
    num_bars_per_group = len(df_stations.columns)
    num_groups = len(df_stations.index)
    for i, bar in enumerate(ax.patches):
        if (i // num_groups) % 2 == 1: # The second set of bars (OCP-Enhanced)
             bar.set_hatch('//')

    # --- TITLES AND LABELS: COMMANDING AND PRECISE ---
    ax.set_title('Average Model Performance Across Budgetary Constraints', weight='bold', pad=20)
    ax.set_ylabel('Average Stations Activated')
    ax.set_xlabel('Monthly Budget (USD)', labelpad=15)
    ax.tick_params(axis='x', rotation=0)

    # --- AESTHETICS: MINIMALIST AND FOCUSED ---
    ax.grid(axis='y', linestyle=':', color='gray', alpha=0.7)
    ax.grid(axis='x', linestyle='') # No vertical grid lines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # --- ANNOTATIONS: DUAL-INFORMATION WARFARE ---
    for i, bar in enumerate(ax.patches):
        # Determine the model and budget for this bar
        model_index = i // num_groups
        budget_index = i % num_groups
        model_name = df_stations.columns[model_index]
        budget_name = df_stations.index[budget_index]
        
        # Retrieve the corresponding data throughput
        data_gb_val = df_data.iloc[budget_index, model_index]
        formatted_data = format_data_size(data_gb_val)

        # Annotation 1: Stations (Primary Metric) - Above the bar
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1, # Small offset above the bar
            f'{bar.get_height():.1f}',
            ha='center',
            va='bottom',
            fontsize=9
        )
        
        # Annotation 2: Throughput (Secondary Metric) - Inside the bar
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2, # Centered vertically inside
            formatted_data,
            ha='center',
            va='center',
            fontsize=9,
            color='white',
            weight='bold'
        )
    
    # Improve y-axis limit for spacing
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    # --- LEGEND: CLEAR AND UNOBTRUSIVE ---
    ax.legend(title='Model', frameon=False, loc='upper left')

    output_filename = 'multi_budget_verdict.png'
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Victory: The final verdict has been rendered to '{output_filename}'")


if __name__ == '__main__':
    # --- MISSION DIRECTIVE ---
    simulation_files = {
        '$150k': 'budget_sim_n100_s5_150k_v2.json',
        '$225k': 'budget_sim_n100_s5_225k_v2.json',
        '$300k': 'budget_sim_n100_s5_300k_v2.json',
    }

    # Create dummy files for demonstration if they don't exist
    for path in simulation_files.values():
        if not Path(path).exists():
            print(f"NOTICE: Dummy file '{path}' not found. Creating for demonstration.")
            dummy_budget = float(Path(path).stem.split('_')[-1].replace('k', '')) * 1000
            base_stations = (dummy_budget / 150000) * 8
            ocp_stations = base_stations * 1.25
            base_data_gb = base_stations * 300000
            ocp_data_gb = ocp_stations * 310000 # OCP is also slightly more efficient per station

            dummy_data = {
                "config": {"budget_constrained": {"budget_usd_per_month": dummy_budget}},
                "raw_trial_data": {
                    "budget": [
                        {
                            "baseline": {"status": "optimal", "stations": base_stations + (i % 3 - 1), "data_gb": base_data_gb + (i % 5 - 2)*5000},
                            "ocp": {"status": "optimal", "stations": ocp_stations + (i % 3 - 1), "data_gb": ocp_data_gb + (i % 5 - 2)*5000}
                        } for i in range(100)
                    ]
                }
            }
            with open(path, 'w') as f:
                json.dump(dummy_data, f)

    generate_multi_budget_plot(simulation_files)

