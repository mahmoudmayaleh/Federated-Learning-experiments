import os
import matplotlib.pyplot as plt
from results_visualizer import ResultsVisualizer
from prettytable import PrettyTable

NUM_ROUNDS = 15 
CLIENTS_LIST = [2, 5, 10]
EPOCHS = 1
ALPHA_DIRICHLET = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.01

def main():
    all_accuracies = {}
    all_losses = {}
    all_rounds = None

    for num_clients in CLIENTS_LIST:
        filename = f"server_history_{NUM_ROUNDS}_{EPOCHS}_{num_clients}.json"
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
        all_accuracies[num_clients] = [acc.get(str(r), None) for r in rounds]
        all_losses[num_clients] = [loss.get(str(r), None) for r in rounds]

    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    for num_clients, accs in all_accuracies.items():
        if all_rounds and len(all_rounds) == len(accs):
            plt.plot(all_rounds, accs, marker='o', label=f"Clients={num_clients}")
    plt.title("FL Accuracy vs Rounds for Different Number of Clients")
    plt.xlabel("Round")
    plt.ylabel("FL Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("baseline_figures/compare_FL_accuracy_num_clients.png")
    plt.close()

    # Plot loss comparison
    plt.figure(figsize=(10, 6))
    for num_clients, losses in all_losses.items():
        # Filter out None values for plotting
        if all_rounds is not None:
            filtered_rounds = [r for r, l in zip(all_rounds, losses) if l is not None]
            filtered_losses = [l for l in losses if l is not None]
            plt.plot(filtered_rounds, filtered_losses, marker='o', label=f"Clients={num_clients}")
    plt.title("FL Loss vs Rounds for Different Number of Clients")
    plt.xlabel("Round")
    plt.ylabel("FL Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("baseline_figures/compare_FL_loss_num_clients.png")
    plt.close()

    # Print summary table
    table = PrettyTable()
    table.field_names = ["Round"] + [f"Acc (Clients={c})" for c in CLIENTS_LIST]
    if all_rounds is not None:
        for i, r in enumerate(all_rounds):
            row = [str(r)]
            for num_clients in CLIENTS_LIST:
                acc = all_accuracies.get(num_clients, [None]*len(all_rounds))[i]
                row.append(f"{acc:.3f}" if acc is not None else "N/A")
            table.add_row(row)
    print(table)

    try:
        import pandas as pd
        import dataframe_image as dfi
        df = pd.DataFrame([row for row in table._rows], columns=table.field_names)
        dfi.export(df, "baseline_figures/compare_FL_accuracy_num_clients_table.png")
    except ImportError:
        print("Install pandas and dataframe_image to save the table as PNG.")

if __name__ == "__main__":
    main()
