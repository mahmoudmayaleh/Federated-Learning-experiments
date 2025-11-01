from results_visualizer import ResultsVisualizer


ALPHA_DIRICHLET = 1
def main():
    visualizer = ResultsVisualizer()
    visualizer.load_simulation_results(f"server_history_{ALPHA_DIRICHLET}.json")
    visualizer.print_results_table()
    visualizer.plot_results("baseline_figures")


if __name__ == "__main__":
    main() 
