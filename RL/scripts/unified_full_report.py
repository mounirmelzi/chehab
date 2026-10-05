import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def generate_report(file_a, label):
    try:
        df_a = pd.read_excel(file_a)
       # df_b = pd.read_excel(file_b)

        # Full metric spectrum: Mokrane (Noise/Safety) + Melzi (Speed/Keys)
        all_metrics = [
            "Initial Exec", "Agent Final Exec", "Safe Final Exec", "Safe Cost Reduction (%)",
            "Initial Keys", "Agent Final Keys", "Safe Final Keys",
            "Initial Noise", "Agent Final Noise", "Safe Final Noise", "Noise Margin",
            "Agent Violated", "Safe Violated", "Safety Activated", "RL Time (ms)"
        ]

        # Filter only metrics present in the files
        valid_metrics = [m for m in all_metrics if m in df_a.columns]

        summary_a = df_a.groupby("Budget")[valid_metrics].mean().add_prefix("A_")
       # summary_b = df_b.groupby("Budget")[valid_metrics].mean().add_prefix("B_")

        # combined = pd.concat([summary_a, summary_b], axis=1)
        combined = pd.concat([summary_a], axis=1)
        print(f"\n==================== FULL UNIFIED METRICS REPORT: {label} ====================")
        print(combined.T)

    except FileNotFoundError as e:
        print(f"Skipping {label}: {e}")

# Run for all test suites
generate_report("./resultsABench.xlsx", "SCALAR BENCHMARKS")
generate_report("./resultsABenchDNN.xlsx", "DNN VECTOR BENCHMARKS")
generate_report("./resultsABenchDNNScaled.xlsx", "Scaled DNN VECTOR BENCHMARKS")