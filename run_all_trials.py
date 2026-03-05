#!/usr/bin/env python3
"""
THESIS SIMULATION — STEP 2: RUN ALL 50 TRIALS
Peters PG/2415890

USAGE:
  python3 run_all_trials.py                        # all 50 trials
  python3 run_all_trials.py --trials 1 --workers 1 # single test trial
  python3 run_all_trials.py --debug                # show full NS-3 output
  python3 run_all_trials.py --fault power          # one fault type only
"""

import subprocess, os, sys, time, argparse
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

NS3_DIR    = os.path.expanduser("~/ns-3.38")
SIM_SCRIPT = "thesis-fault-sim"
OUTPUT_DIR = os.path.expanduser("~/thesis-sim/output/raw")
MERGED_CSV = os.path.expanduser("~/thesis-sim/output/kpi_master_dataset.csv")
N_TRIALS   = 50
FAULT_TYPES = ["none", "power", "congestion", "hardware"]

# Global flag — set from args
DEBUG = False

def run_trial(args):
    trial, fault, output_dir = args
    cmd = [f"{NS3_DIR}/ns3", "run",
           f"{SIM_SCRIPT} --trial={trial} --fault={fault} --outputDir={output_dir}"]
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, cwd=NS3_DIR,
            capture_output=True, text=True, timeout=600)
        elapsed = time.time() - t0

        if result.returncode != 0:
            # NS-3 writes errors to stdout AND stderr — combine both
            combined = (result.stdout + "\n" + result.stderr).strip()
            lines = [l for l in combined.split('\n') if l.strip()]
            # Show last 5 meaningful lines
            snippet = ' | '.join(lines[-5:]) if lines else '(no output)'
            print(f"  [FAIL] trial={trial} fault={fault}\n         {snippet}")
            return (trial, fault, False, elapsed)

        return (trial, fault, True, elapsed)

    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] trial={trial} fault={fault}")
        return (trial, fault, False, 600.0)
    except Exception as e:
        print(f"  [ERROR] trial={trial} fault={fault}: {e}")
        return (trial, fault, False, 0.0)


def debug_single_trial():
    """Run one trial in foreground showing full NS-3 output. Use to diagnose failures."""
    print("\n" + "="*60)
    print("  DEBUG MODE — running trial=0 fault=none")
    print("  Showing full NS-3 output...")
    print("="*60 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cmd = [f"{NS3_DIR}/ns3", "run",
           f"{SIM_SCRIPT} --trial=0 --fault=none --outputDir={OUTPUT_DIR}"]

    result = subprocess.run(cmd, cwd=NS3_DIR, text=True)  # no capture — prints live

    print("\n" + "="*60)
    print(f"  Return code: {result.returncode}")
    csv = os.path.join(OUTPUT_DIR, "kpi_trial0_none.csv")
    if os.path.exists(csv):
        size = os.path.getsize(csv)
        print(f"  CSV created: {csv} ({size} bytes)")
        if size > 0:
            with open(csv) as f:
                lines = f.readlines()
            print(f"  CSV rows: {len(lines)}")
            print(f"  First row: {lines[0].strip()}")
            if len(lines) > 1:
                print(f"  Second row: {lines[1].strip()}")
        else:
            print("  CSV is EMPTY — NS-3 crashed before writing any data")
    else:
        print("  CSV NOT CREATED — NS-3 crashed before opening output file")
    print("="*60 + "\n")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",  type=int, default=N_TRIALS)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fault",   type=str, default=None)
    parser.add_argument("--debug",   action='store_true',
                        help="Run one trial in foreground to see full error output")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Debug mode: just run one trial and show everything ─────────────────
    if args.debug:
        ok = debug_single_trial()
        if not ok:
            print("Debug trial FAILED. Check the output above for the error.")
            print("\nCommon causes:")
            print("  1. thesis-fault-sim.cc has a C++ compile error")
            print("     Fix: check ~/ns-3.38/scratch/thesis-fault-sim.cc")
            print("  2. NS-3 module missing (e.g. LTE not built)")
            print("     Fix: cd ~/ns-3.38 && ./ns3 configure --enable-modules=lte,... && ./ns3 build")
            print("  3. Wrong NS-3 version")
            print("     Fix: ls ~/ns-3.38/src/ | grep lte")
        sys.exit(0 if ok else 1)

    fault_list = [args.fault] if args.fault else FAULT_TYPES
    total_runs = args.trials * len(fault_list)

    print(f"\n{'='*60}")
    print(f"  THESIS NS-3 SIMULATION RUNNER")
    print(f"  Trials: {args.trials}  |  Fault types: {fault_list}")
    print(f"  Total runs: {total_runs}  |  Workers: {args.workers}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    # ── Build ───────────────────────────────────────────────────────────────
    print("[1] Building NS-3 simulation script...")
    build = subprocess.run(
        [f"{NS3_DIR}/ns3", "build", SIM_SCRIPT],
        cwd=NS3_DIR, capture_output=True, text=True)
    if build.returncode != 0:
        print(f"BUILD FAILED:\n{build.stdout}\n{build.stderr}")
        print("\nMake sure thesis-fault-sim.cc is in ~/ns-3.38/scratch/")
        sys.exit(1)
    print("  Build successful.\n")

    # ── Quick sanity check before launching all workers ────────────────────
    print("[1b] Quick sanity check (1 trial before launching all workers)...")
    test_cmd = [f"{NS3_DIR}/ns3", "run",
                f"{SIM_SCRIPT} --trial=0 --fault=none --outputDir={OUTPUT_DIR}"]
    test = subprocess.run(test_cmd, cwd=NS3_DIR, capture_output=True, text=True, timeout=120)
    if test.returncode != 0:
        combined = (test.stdout + "\n" + test.stderr).strip()
        print(f"  SANITY CHECK FAILED. Full output:")
        print(combined[-2000:])
        print("\n  Run with --debug for full live output:")
        print("  python3 run_all_trials.py --debug")
        sys.exit(1)

    csv_check = os.path.join(OUTPUT_DIR, "kpi_trial0_none.csv")
    if not os.path.exists(csv_check) or os.path.getsize(csv_check) == 0:
        print(f"  SANITY CHECK FAILED: CSV empty or missing: {csv_check}")
        print("  Run: python3 run_all_trials.py --debug")
        sys.exit(1)

    rows = sum(1 for _ in open(csv_check)) - 1
    print(f"  Sanity check PASSED — CSV has {rows} data rows.\n")

    # ── Run all trials ──────────────────────────────────────────────────────
    jobs = [(t, f, OUTPUT_DIR) for f in fault_list for t in range(args.trials)]
    print(f"[2] Running {total_runs} simulation trials...")
    completed = failed = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_trial, job): job for job in jobs}
        for future in as_completed(futures):
            trial, fault, success, elapsed = future.result()
            completed += 1
            if not success:
                failed += 1
            eta = (time.time() - t_start) / completed * (total_runs - completed)
            print(f"  [{completed:3d}/{total_runs}] trial={trial:2d} fault={fault:12s} "
                  f"{'OK' if success else 'FAIL'} {elapsed:5.1f}s | ETA {eta/60:.1f} min")

    print(f"\n  Completed: {completed - failed}/{total_runs}  |  Failed: {failed}")

    if failed == total_runs:
        print("\n  ALL trials failed. Run: python3 run_all_trials.py --debug")
        sys.exit(1)

    # ── Merge CSVs ──────────────────────────────────────────────────────────
    print(f"\n[3] Merging CSV files...")
    all_dfs = []
    for fault in fault_list:
        for trial in range(args.trials):
            csv_path = os.path.join(OUTPUT_DIR, f"kpi_trial{trial}_{fault}.csv")
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 100:
                try:
                    df = pd.read_csv(csv_path)
                    if len(df) > 0:
                        all_dfs.append(df)
                except Exception as e:
                    print(f"  Warning: {csv_path}: {e}")

    if not all_dfs:
        print("  No valid CSV files found.")
        sys.exit(1)

    master = pd.concat(all_dfs, ignore_index=True)
    master.to_csv(MERGED_CSV, index=False)
    label_map = {0:"Normal", 1:"Power Fault", 2:"Congestion", 3:"gNB HW Failure"}
    print(f"  Saved: {MERGED_CSV}  ({len(master):,} rows)")
    for k, v in master['fault_label'].value_counts().sort_index().items():
        print(f"    {label_map.get(k,k)}: {v:,} ({100*v/len(master):.1f}%)")

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  DONE — {total_time/60:.1f} minutes")
    print(f"  Next: python3 preprocess_and_train.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
