#!/usr/bin/env python3
"""
THESIS SIMULATION — STEP 2: RUN ALL 50 TRIALS
Peters

USAGE:
  python3 run_all_trials.py                        # all 50 trials
  python3 run_all_trials.py --trials 1 --workers 1 # single test trial
  python3 run_all_trials.py --debug                # show full NS-3 output
  python3 run_all_trials.py --fault power          # one fault type only
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

NS3_DIR = os.path.expanduser("~/ns-3.38")
NS3_WRAPPER = os.path.expanduser("~/thesis-sim/bin/ns3")
VENV_PY = os.path.expanduser("~/thesis-sim/venv/bin/python")
SIM_SCRIPT = "thesis-fault-sim"
SIM_SCRIPT_LTE = "thesis-fault-sim-lte"
OUTPUT_DIR = os.path.expanduser("~/thesis-sim/output/raw")
MERGED_CSV = os.path.expanduser("~/thesis-sim/output/kpi_master_dataset.csv")
N_TRIALS = 50
FAULT_TYPES = ["none", "power", "congestion", "hardware"]
LTE_SIM_TIME = 120  # seconds; full 300s is ~2h/trial on typical hardware
LTE_NUM_UES = 280  # NS-3 HARQ stable limit (use --num-ues 500 for thesis target if simTime>=120)

# Global flag — set from args
DEBUG = False
ACTIVE_SIM = SIM_SCRIPT
LTE_SIM_TIME_ACTIVE = LTE_SIM_TIME
LTE_NUM_UES_ACTIVE = LTE_NUM_UES


def ns3_cmd(*args):
    """Run NS-3 via venv Python 3.11 (system python3 may be 3.14 and break ./ns3)."""
    if os.path.isfile(NS3_WRAPPER):
        return [NS3_WRAPPER, *args]
    return [VENV_PY, f"{NS3_DIR}/ns3", *args]


def run_trial(args):
    trial, fault, output_dir, sim, sim_time, num_ues = args
    sim_args = f"--trial={trial} --fault={fault} --outputDir={output_dir}"
    if sim == SIM_SCRIPT_LTE:
        sim_args += f" --simTime={sim_time} --numUes={num_ues}"
    cmd = ns3_cmd("run", f"{sim} {sim_args}")
    if sim == SIM_SCRIPT_LTE:
        timeout = max(3600, int(sim_time * 50))
    else:
        timeout = 600
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, cwd=NS3_DIR, capture_output=True, text=True, timeout=timeout
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            # NS-3 writes errors to stdout AND stderr — combine both
            combined = (result.stdout + "\n" + result.stderr).strip()
            lines = [l for l in combined.split("\n") if l.strip()]
            # Show last 5 meaningful lines
            snippet = " | ".join(lines[-5:]) if lines else "(no output)"
            print(f"  [FAIL] trial={trial} fault={fault}\n         {snippet}")
            return (trial, fault, False, elapsed)

        return (trial, fault, True, elapsed)

    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] trial={trial} fault={fault}")
        return (trial, fault, False, 600.0)
    except Exception as e:
        print(f"  [ERROR] trial={trial} fault={fault}: {e}")
        return (trial, fault, False, 0.0)


def debug_single_trial(sim=SIM_SCRIPT, sim_time=LTE_SIM_TIME, num_ues=LTE_NUM_UES):
    """Run one trial in foreground showing full NS-3 output. Use to diagnose failures."""
    print("\n" + "=" * 60)
    print(f"  DEBUG MODE — {sim} trial=0 fault=none")
    if sim == SIM_SCRIPT_LTE:
        print(f"  simTime={sim_time}s  numUes={num_ues}")
    print("  Showing full NS-3 output...")
    print("=" * 60 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sim_args = f"--trial=0 --fault=none --outputDir={OUTPUT_DIR}"
    if sim == SIM_SCRIPT_LTE:
        sim_args += f" --simTime={sim_time} --numUes={num_ues}"
    cmd = ns3_cmd("run", f"{sim} {sim_args}")

    result = subprocess.run(cmd, cwd=NS3_DIR, text=True)  # no capture — prints live

    print("\n" + "=" * 60)
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
    print("=" * 60 + "\n")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fault", type=str, default=None)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run one trial in foreground to see full error output",
    )
    parser.add_argument(
        "--lte",
        action="store_true",
        help="Use real LTE/EPC simulation (thesis-fault-sim-lte, slower, Ch.3 aligned)",
    )
    parser.add_argument(
        "--sim-time",
        type=int,
        default=None,
        help=f"LTE sim duration in seconds (default {LTE_SIM_TIME} with --lte, 300 in C++ if omitted)",
    )
    parser.add_argument(
        "--num-ues",
        type=int,
        default=None,
        help=f"LTE UE count (default {LTE_NUM_UES} with --lte; thesis Table 3.1 target is 500)",
    )
    args = parser.parse_args()

    global ACTIVE_SIM, LTE_SIM_TIME_ACTIVE, LTE_NUM_UES_ACTIVE
    ACTIVE_SIM = SIM_SCRIPT_LTE if args.lte else SIM_SCRIPT
    if args.sim_time is not None:
        LTE_SIM_TIME_ACTIVE = args.sim_time
    elif ACTIVE_SIM == SIM_SCRIPT_LTE:
        LTE_SIM_TIME_ACTIVE = LTE_SIM_TIME
    if args.num_ues is not None:
        LTE_NUM_UES_ACTIVE = args.num_ues
    elif ACTIVE_SIM == SIM_SCRIPT_LTE:
        LTE_NUM_UES_ACTIVE = LTE_NUM_UES

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Debug mode: just run one trial and show everything ─────────────────
    if args.debug:
        ok = debug_single_trial(ACTIVE_SIM, LTE_SIM_TIME_ACTIVE, LTE_NUM_UES_ACTIVE)
        if not ok:
            print("Debug trial FAILED. Check the output above for the error.")
            print("\nCommon causes:")
            print("  1. thesis-fault-sim.cc has a C++ compile error")
            print("     Fix: check ~/ns-3.38/scratch/thesis-fault-sim.cc")
            print("  2. NS-3 module missing (e.g. LTE not built)")
            print(
                "     Fix: cd ~/ns-3.38 && ~/thesis-sim/bin/ns3 configure --enable-modules=lte,... && ~/thesis-sim/bin/ns3 build"
            )
            print("  3. Wrong NS-3 version")
            print("     Fix: ls ~/ns-3.38/src/ | grep lte")
        sys.exit(0 if ok else 1)

    fault_list = [args.fault] if args.fault else FAULT_TYPES
    total_runs = args.trials * len(fault_list)

    print(f"\n{'=' * 60}")
    print("  THESIS NS-3 SIMULATION RUNNER")
    print(f"  Simulator: {ACTIVE_SIM}")
    if ACTIVE_SIM == SIM_SCRIPT_LTE:
        print(f"  LTE simTime: {LTE_SIM_TIME_ACTIVE}s  |  numUes: {LTE_NUM_UES_ACTIVE}")
    print(f"  Trials: {args.trials}  |  Fault types: {fault_list}")
    print(f"  Total runs: {total_runs}  |  Workers: {args.workers}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'=' * 60}\n")

    # ── Build ───────────────────────────────────────────────────────────────
    print("[1] Building NS-3 simulation script...")
    build = subprocess.run(
        ns3_cmd("build", ACTIVE_SIM),
        cwd=NS3_DIR,
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        print(f"BUILD FAILED:\n{build.stdout}\n{build.stderr}")
        print(f"\nMake sure {ACTIVE_SIM}.cc is in ~/ns-3.38/scratch/")
        sys.exit(1)
    print("  Build successful.\n")

    # ── Quick sanity check before launching all workers ────────────────────
    print("[1b] Quick sanity check (1 trial before launching all workers)...")
    test_timeout = max(3600, LTE_SIM_TIME_ACTIVE * 50) if ACTIVE_SIM == SIM_SCRIPT_LTE else 120
    test_args = f"{ACTIVE_SIM} --trial=0 --fault=none --outputDir={OUTPUT_DIR}"
    if ACTIVE_SIM == SIM_SCRIPT_LTE:
        test_args += f" --simTime={LTE_SIM_TIME_ACTIVE} --numUes={LTE_NUM_UES_ACTIVE}"
    test_cmd = ns3_cmd("run", test_args)
    test = subprocess.run(
        test_cmd, cwd=NS3_DIR, capture_output=True, text=True, timeout=test_timeout
    )
    if test.returncode != 0:
        combined = (test.stdout + "\n" + test.stderr).strip()
        print("  SANITY CHECK FAILED. Full output:")
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
    jobs = [
        (t, f, OUTPUT_DIR, ACTIVE_SIM, LTE_SIM_TIME_ACTIVE, LTE_NUM_UES_ACTIVE)
        for f in fault_list
        for t in range(args.trials)
    ]
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
            print(
                f"  [{completed:3d}/{total_runs}] trial={trial:2d} fault={fault:12s} "
                f"{'OK' if success else 'FAIL'} {elapsed:5.1f}s | ETA {eta / 60:.1f} min"
            )

    print(f"\n  Completed: {completed - failed}/{total_runs}  |  Failed: {failed}")

    if failed == total_runs:
        print("\n  ALL trials failed. Run: python3 run_all_trials.py --debug")
        sys.exit(1)

    # ── Merge CSVs ──────────────────────────────────────────────────────────
    print("\n[3] Merging CSV files...")
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
    label_map = {0: "Normal", 1: "Power Fault", 2: "Congestion", 3: "gNB HW Failure"}
    print(f"  Saved: {MERGED_CSV}  ({len(master):,} rows)")
    for k, v in master["fault_label"].value_counts().sort_index().items():
        print(f"    {label_map.get(k, k)}: {v:,} ({100 * v / len(master):.1f}%)")

    total_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  DONE — {total_time / 60:.1f} minutes")
    print("  Next: python3 preprocess_and_train.py")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
