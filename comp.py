import pandas as pd
a = pd.read_csv("results_RL_budget_dp_test.csv")   # before
b = pd.read_csv("results_RL_budget_dp_v2.csv")     # after
cols = ["benchmark","noise_budget","final_ops_cost","final_keys_cost","rotation_keys_count"]
print(a[cols].merge(b[cols], on=["benchmark","noise_budget"], suffixes=("_old","_new")).to_string())