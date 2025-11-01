import os
import matplotlib.pyplot as plt
from results_visualizer import ResultsVisualizer
from prettytable import PrettyTable

NUM_ROUNDS = 15
EPOCHS = 1
NUM_CLIENTS = 10
LEARNING_RATES = [0.001, 0.01, 0.05, 0.1]

def main():
    all_accuracies = {}
    all_losses = {}
    all_rounds = None

    for lr in LEARNING_RATES:
        # Format learning rate for filename (e.g., 0.01 -> 0.01)
        lr_str = f"{lr:.3f}".rstrip('0').rstrip('.') if '.' in f"{lr:.3f}" else str(lr)
        filename = f"server_history_{NUM_ROUNDS}_{EPOCHS}_{NUM_CLIENTS}_{lr_str}.json"
        if not os.path.exists(filename):
            print(f"File {filename} not found, skipping.")
            continue

        visualizer = ResultsVisualizer()
        visualizer.load_simulation_results(filename)

        acc = visualizer.results.get("FL_accuracy", {})
        loss = visualizer.results.get("FL_loss", {})
        rounds = sorted([int(k) for k in acc.keys() if str(k).isdigit()])
        if all_rounds is None:
            all_rounds = rounds
        all_accuracies[lr] = [acc.get(str(r), None) for r in rounds]
        all_losses[lr] = [loss.get(str(r), None) for r in rounds]

    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    for lr, accs in all_accuracies.items():
        if all_rounds and len(all_rounds) == len(accs):
            plt.plot(all_rounds, accs, marker='o', label=f"LR={lr}")
    plt.title("FL Accuracy vs Rounds for Different Learning Rates")
    plt.xlabel("Round")
    plt.ylabel("FL Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("baseline_figures/compare_FL_accuracy_learning_rates.png")
    plt.close()

    # Plot loss comparison
    plt.figure(figsize=(10, 6))
    for lr, losses in all_losses.items():
        if all_rounds and len(all_rounds) == len(losses):
            plt.plot(all_rounds, losses, marker='o', label=f"LR={lr}")
    plt.title("FL Loss vs Rounds for Different Learning Rates")
    plt.xlabel("Round")
    plt.ylabel("FL Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("baseline_figures/compare_FL_loss_learning_rates.png")
    plt.close()

    # Print summary table
    table = PrettyTable()
    table.field_names = ["Round"] + [f"Acc (LR={lr})" for lr in LEARNING_RATES]
    if all_rounds is not None:
        for i, r in enumerate(all_rounds):
            row = [str(r)]
            for lr in LEARNING_RATES:
                acc = all_accuracies.get(lr, [None]*len(all_rounds))[i]
                row.append(f"{acc:.3f}" if acc is not None else "N/A")
            table.add_row(row)
    print(table)

    # Optionally, save table as PNG
    try:
        import pandas as pd
        import dataframe_image as dfi
        df = pd.DataFrame([row for row in table._rows], columns=table.field_names)
        dfi.export(df, "baseline_figures/compare_FL_accuracy_learning_rates_table.png")
    except ImportError:
        print("Install pandas and dataframe_image to save the table as PNG.")

if __name__ == "__main__":
    main()
