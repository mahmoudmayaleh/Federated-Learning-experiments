import json
import os
import matplotlib.pyplot as plt
from prettytable import PrettyTable

ALPHA_DIRICHLET = 1


class ResultsVisualizer:
    def __init__(self) -> None:
        self.results = {}

    def load_simulation_results(self, file_name: str) -> None:
        with open(file_name, 'r') as file:
            raw_results = json.load(file)
            self.results = {}

            for metric, data in raw_results.items():
                if isinstance(data, dict) and all(isinstance(v, list) and v and isinstance(v[0], list) for v in data.values()):
                    for sub_metric, values in data.items():
                        self.results[f"{metric}_{sub_metric}"] = {str(r): v for r, v in values}

                elif isinstance(data, dict) and all(isinstance(v, list) and len(v) == 2 for v in data.values()):
                    self.results[metric] = {str(k): v[1] for k, v in data.items()}
                else:
                    self.results[metric] = data

    def plot_results(self, fig_directory: str) -> None:
        if not os.path.exists(fig_directory):
            os.makedirs(fig_directory)


        metrics_to_plot = ["FL_loss", "FL_accuracy"]

        for metric_name in metrics_to_plot:
            if metric_name in self.results:
                rounds = [int(k) for k in self.results[metric_name].keys() if str(k).isdigit()]
                values = [self.results[metric_name][str(r)] for r in rounds]
                if not rounds:
                    continue
                plt.figure()
                plt.plot(rounds, values, marker='o')
                plt.title(f"{metric_name} over Rounds")
                plt.xlabel("Round")
                plt.ylabel(metric_name)
                plt.grid(True)
                plot_path = os.path.join(fig_directory, f"{metric_name}_{ALPHA_DIRICHLET}.png")
                plt.savefig(plot_path)
                plt.close()

    from typing import Optional

    def print_results_table(self, save_path: Optional[str] = None) -> None:
        table = PrettyTable()

        all_rounds = sorted({
            int(round_num)
            for metric in self.results.values()
            for round_num in metric.keys()
            if str(round_num).isdigit()
        })
        metrics = ["FL_loss", "FL_accuracy"]  

        table.field_names = ["Round"] + metrics

        for round_num in all_rounds:
            row = [str(round_num)]
            for metric in metrics:
                value = self.results.get(metric, {}).get(str(round_num), "N/A")
                formatted_value = str(round(value, 3)) if isinstance(value, (int, float)) else str(value)
                row.append(formatted_value)
            table.add_row(row)

        print(table)

        if save_path:
            import pandas as pd
            import dataframe_image as dfi
            df = pd.DataFrame([row for row in table._rows], columns=table.field_names)
            dfi.export(df, save_path)


# visualizer = ResultsVisualizer()
# visualizer.load_simulation_results(f"server_history_{NUM_ROUNDS}_{EPOCHS}.json")
# visualizer.plot_results("figures")
# visualizer.print_results_table(f"figures/FL_results_table_{NUM_ROUNDS}_{EPOCHS}.png")
