import os
import datetime
import pandas as pd

TRAINING_LOG_FILE = "job_logs.xlsx"
TEST_LOG_FILE = "test_results.xlsx"

def _training_log_path(job_id: str) -> str:
    job_str = str(job_id) if job_id else "nojob"
    return f"job_logs_{job_str}.xlsx"


def log_training_details(model_params, job_id, num_data, num_actions, total_timesteps, output_model_name, notes=""):
    log_entry = {
        "Job ID": job_id,
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Learning Rate": model_params.get("learning_rate", ""),
        "n_steps": model_params.get("n_steps", ""),
        "batch_size": model_params.get("batch_size", ""),
        "n_epochs": model_params.get("n_epochs", ""),
        "gamma": model_params.get("gamma", ""),
        "gae_lambda": model_params.get("gae_lambda", ""),
        "ent_coef": model_params.get("policy_kwargs", {}).get("ent_coef", ""),
        "num_data": num_data,
        "num_actions": num_actions,
        "total_timesteps": total_timesteps,
        "Output Model Name": output_model_name,
        "TensorBoard Log Dir": model_params.get("tensorboard_log", ""),
        "Notes": notes
    }
    
    log_path = _training_log_path(job_id)
    if not os.path.exists(log_path):
        df = pd.DataFrame(columns=log_entry.keys())
    else:
        try:
            df = pd.read_excel(log_path)
        except Exception:
            # fallback if file is corrupted/non-excel
            df = pd.DataFrame(columns=log_entry.keys())
    # Avoid concat with empty/all-NA frames to prevent FutureWarning
    if df.empty:
        df = pd.DataFrame([log_entry])
    else:
        df.loc[len(df)] = log_entry
    df.to_excel(log_path, index=False)
    print(f"Logged training details to {log_path}")

def log_test_results(results, sheet_name="TestResults"):
    """
    Logs testing metrics (e.g. initial/final expressions, costs, and steps) into an Excel file.
    Each run can be logged in a separate sheet.
    """
    if os.path.exists(TEST_LOG_FILE):
        with pd.ExcelWriter(TEST_LOG_FILE, mode="a", engine="openpyxl", if_sheet_exists="new") as writer:
            pd.DataFrame(results).to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(TEST_LOG_FILE, mode="w", engine="openpyxl") as writer:
            pd.DataFrame(results).to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Test results written to {TEST_LOG_FILE} under sheet '{sheet_name}'")
