import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def generate_pareto_plot(file_path='final_resuls_pareto_n8_sat1.json'):
    """
    Loads simulation results and generates the definitive, publication-quality
    Pareto Frontier plot. This version utilizes manual label placement for
    absolute clarity and visual impact, suitable for the IEEE Aerospace paper.
    """
    try:
        # --- 1. Load and Process Data ---
        with open(file_path, 'r') as f:
            results_data = json.load(f)

        pareto_raw_data = [
            item for trial in results_data['raw_trial_data']['pareto']
            if trial is not None
            for item in trial
        ]
        df = pd.DataFrame(pareto_raw_data)
        df_optimal = df[df['status'] == 'optimal'].copy()

        if df_optimal.empty:
            print("Error: No optimal solutions found. Cannot generate plot.")
            return

        pareto_summary = df_optimal.groupby('p_base').mean(numeric_only=True).reset_index()

        # --- 2. Create the Plot ---
        plt.style.use('seaborn-v0_8-ticks')
        fig, ax = plt.subplots(figsize=(10, 6.5))

        primary_color = '#002D62'
        marker_color = '#C8102E' # A slightly brighter, more assertive red.

        ax.plot(pareto_summary['stations'], pareto_summary['data_gb'],
                linestyle='-', color=primary_color, linewidth=2.5,
                label='Efficient Frontier', zorder=1)

        ax.scatter(pareto_summary['stations'], pareto_summary['data_gb'],
                   s=80, c=marker_color, edgecolor='black',
                   linewidth=0.75, zorder=5, label='OCP Setpoints')

        # --- 3. Manual Annotation with Strategic Placement ---
        # This dictionary defines the precise position and alignment for each label,
        # ensuring perfect clarity and zero overlap.
        # (xytext) defines the offset in points from the data point.
        # (ha, va) defines the horizontal and vertical alignment.
        label_placements = {
            0:    {'xytext': (0, -25), 'ha': 'center', 'va': 'bottom'},
            100:  {'xytext': (0, -25), 'ha': 'center', 'va': 'bottom'},
            250:  {'xytext': (10, 10), 'ha': 'left', 'va': 'bottom'},
            500:  {'xytext': (10, 0), 'ha': 'left', 'va': 'center'},
            1000: {'xytext': (10, -10), 'ha': 'left', 'va': 'top'},
            2500: {'xytext': (-10, -10), 'ha': 'right', 'va': 'top'},
        }

        for _, row in pareto_summary.iterrows():

            
            p_base = int(row["p_base"])
            
            placement = label_placements.get(p_base, {'xytext': (0, 15), 'ha': 'center', 'va': 'bottom'}) # Default placement
            
            label = f'$P_{{base}} = \\${p_base:,}$'
            
            ax.annotate(label,
                        xy=(row['stations'], row['data_gb']),
                        xytext=placement['xytext'],
                        textcoords='offset points',
                        fontsize=11,
                        fontweight='bold',
                        ha=placement['ha'],
                        va=placement['va'],
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.75))


        # --- 4. Professional Formatting and Styling ---
        plt.rcParams['font.family'] = 'serif'

        ax.set_title('OCP-Enhanced Model: The Efficient Frontier', fontsize=16, pad=20, weight='bold')
        ax.set_xlabel('Network Complexity (Number of Stations Activated)', fontsize=12)
        ax.set_ylabel('Average Data Throughput (GB)', fontsize=12)

        ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
        ax.tick_params(axis='both', which='major', labelsize=11)

        ax.invert_xaxis()
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        ax.set_facecolor('#F8F8F8')

        # --- 5. Save the Output ---
        output_filename =  file_path.replace(".json", "") + '_pareto_frontier.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Definitive plot successfully generated and saved to '{output_filename}'")

    except FileNotFoundError:
        print(f"Error: The results file '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    generate_pareto_plot(file_path='pareto_sim_n100_s5_v8.json')
