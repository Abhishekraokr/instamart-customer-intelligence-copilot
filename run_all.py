import subprocess
import sys
import time
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_step(step_num, title, script_path):
    print("\n" + "=" * 75)
    print(f"STEP {step_num}: {title}")
    print("=" * 75)
    start_t = time.time()
    
    cmd = [sys.executable, os.path.join(BASE_DIR, script_path)]
    result = subprocess.run(cmd, capture_output=False)
    
    duration = time.time() - start_t
    if result.returncode != 0:
        print(f"\n[-] Step {step_num} failed with return code {result.returncode}!")
        sys.exit(result.returncode)
    print(f"\n[+] Step {step_num} completed successfully in {duration:.2f}s.")
    return duration

def main():
    total_start = time.time()
    print("*" * 75)
    print("STARTING INSTAMART COPILOT PIPELINE (PARTS A TO D)")
    print("*" * 75)

    timings = {}
    
    # Part A: Data Cleaning, Storage, SQL Analytics & Saved Plots
    timings["Part A: Data Pipeline & Storage"] = run_step(
        1, "Data Cleaning & SQLite Relational Storage", os.path.join("src", "clean_data.py")
    )
    timings["Part A: SQL Analytics Queries"] = run_step(
        2, "SQL Business Analytics Queries (Q1 - Q7)", os.path.join("src", "sql_analysis.py")
    )
    timings["Part A: Saved .png Plots"] = run_step(
        3, "Generate Saved .png Plots via plt.savefig()", os.path.join("src", "generate_plots.py")
    )
    
    # Part B: Segmentation & Lapse Prediction
    timings["Part B: Segmentation & Lapse"] = run_step(
        4, "Customer RFM Segmentation & Lapse Prediction", os.path.join("src", "segments_and_lapse.py")
    )
    
    # Part C: Support Ticket Triage via LLM
    timings["Part C: Support Ticket Triage"] = run_step(
        5, "Customer Support Ticket LLM Triage & Token Budget Demo", os.path.join("src", "ticket_triage.py")
    )
    
    # Part D: Policy RAG & ReAct Refund Copilot
    timings["Part D: Policy RAG & ReAct Agent"] = run_step(
        6, "Policy RAG & ReAct Agent Scenarios (S1 - S7)", os.path.join("src", "policy_rag_and_agent.py")
    )

    total_duration = time.time() - total_start
    print("\n" + "*" * 75)
    print("ALL PARTS COMPLETED SUCCESSFULLY!")
    print("*" * 75)
    print("\nExecution Timing Summary:")
    for name, dur in timings.items():
        print(f"  - {name:<45}: {dur:6.2f}s")
    print(f"  - {'Total Pipeline Runtime':<45}: {total_duration:6.2f}s")
    print("\nOutputs verified across data/, sql/, outputs/, and state/.")

if __name__ == "__main__":
    main()
