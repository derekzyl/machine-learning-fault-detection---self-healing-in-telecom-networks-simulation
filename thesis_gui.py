#!/usr/bin/env python3
"""
Thesis Pipeline — graphical launcher (Tkinter).

No terminal knowledge required. Double-click or run:
  python3 thesis_gui.py
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

# ── Paths ─────────────────────────────────────────────────────────────────────
HOME = Path.home()
THESIS_DIR = Path(os.environ.get("THESIS_DIR", HOME / "thesis-sim"))
REPO_DIR = Path(__file__).resolve().parent
VENV_PY = THESIS_DIR / "venv" / "bin" / "python"
PYTHON = str(VENV_PY) if VENV_PY.is_file() else sys.executable


def thesis_script(name: str) -> Path:
    p = THESIS_DIR / name
    if p.is_file():
        return p
    p = REPO_DIR / name
    return p


class ThesisGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Thesis Pipeline — ML Fault Detection & Self-Healing")
        self.geometry("960x680")
        self.minsize(820, 560)

        self._proc: subprocess.Popen | None = None
        self._log_queue: queue.Queue[str] = queue.Queue()
        self._busy = False

        self._build_ui()
        self._poll_log()
        self._refresh_status()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")

        header = ttk.Frame(self, padding=(12, 10))
        header.pack(fill=tk.X)
        ttk.Label(
            header,
            text="Telecom Network Fault Detection Pipeline",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor=tk.W)
        ttk.Label(
            header,
            text="Run simulations, train models, and generate reports — click buttons below.",
            font=("Segoe UI", 10),
        ).pack(anchor=tk.W)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(header, textvariable=self.status_var, font=("Segoe UI", 9)).pack(
            anchor=tk.W, pady=(6, 0)
        )

        body = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        left = ttk.Frame(body, width=380)
        body.add(left, weight=1)

        right = ttk.Frame(body)
        body.add(right, weight=2)

        self._build_setup_panel(left)
        self._build_sim_panel(left)
        self._build_ml_panel(left)
        self._build_reports_panel(left)
        self._build_log_panel(right)

        foot = ttk.Frame(self, padding=(10, 6))
        foot.pack(fill=tk.X)
        self.progress = ttk.Progressbar(foot, mode="indeterminate")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Button(foot, text="Stop", command=self._stop_task).pack(side=tk.RIGHT)
        ttk.Button(foot, text="Quit", command=self.destroy).pack(side=tk.RIGHT, padx=(0, 6))

    def _section(self, parent: ttk.Frame, title: str) -> ttk.LabelFrame:
        lf = ttk.LabelFrame(parent, text=title, padding=8)
        lf.pack(fill=tk.X, pady=4)
        return lf

    def _build_setup_panel(self, parent: ttk.Frame) -> None:
        f = self._section(parent, "1 · Setup & Check")
        ttk.Button(f, text="Run full setup (first time only)", command=self._run_setup).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(f, text="Check environment", command=self._check_env).pack(fill=tk.X, pady=2)
        ttk.Button(f, text="Open thesis folder", command=self._open_thesis_dir).pack(
            fill=tk.X, pady=2
        )

    def _build_sim_panel(self, parent: ttk.Frame) -> None:
        f = self._section(parent, "2 · NS-3 Simulation")
        self.sim_mode = tk.StringVar(value="kpi")
        for val, label in [
            ("kpi", "Fast KPI generator (recommended first)"),
            ("lte", "Real LTE HetNet (slow)"),
            ("nr", "5G NR HetNet (very slow)"),
        ]:
            ttk.Radiobutton(f, text=label, variable=self.sim_mode, value=val).pack(
                anchor=tk.W
            )

        row = ttk.Frame(f)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text="Trials:").pack(side=tk.LEFT)
        self.trials_var = tk.StringVar(value="50")
        ttk.Entry(row, textvariable=self.trials_var, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(row, text="Workers:").pack(side=tk.LEFT, padx=(8, 0))
        self.workers_var = tk.StringVar(value="2")
        ttk.Entry(row, textvariable=self.workers_var, width=4).pack(side=tk.LEFT, padx=4)

        row2 = ttk.Frame(f)
        row2.pack(fill=tk.X)
        ttk.Label(row2, text="Sim time (s):").pack(side=tk.LEFT)
        self.simtime_var = tk.StringVar(value="120")
        ttk.Entry(row2, textvariable=self.simtime_var, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(row2, text="UEs:").pack(side=tk.LEFT, padx=(8, 0))
        self.ues_var = tk.StringVar(value="280")
        ttk.Entry(row2, textvariable=self.ues_var, width=6).pack(side=tk.LEFT, padx=4)

        ttk.Button(f, text="Run simulation trials", command=self._run_simulation).pack(
            fill=tk.X, pady=(6, 2)
        )
        ttk.Button(f, text="Test one trial (debug)", command=self._run_debug_trial).pack(
            fill=tk.X, pady=2
        )

    def _build_ml_panel(self, parent: ttk.Frame) -> None:
        f = self._section(parent, "3 · Machine Learning")
        self.skip_svm = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Skip SVM (faster training)", variable=self.skip_svm).pack(
            anchor=tk.W
        )
        ttk.Button(f, text="Train models (RF + LSTM + SVM)", command=self._run_training).pack(
            fill=tk.X, pady=4
        )
        ttk.Button(f, text="Run MAPE-K evaluation", command=self._run_mapek).pack(
            fill=tk.X, pady=2
        )

    def _build_reports_panel(self, parent: ttk.Frame) -> None:
        f = self._section(parent, "4 · Figures & Documents")
        ttk.Button(f, text="Full Chapter 4 pipeline", command=self._run_ch4).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(f, text="Chapter 3 figures", command=self._run_fig3).pack(fill=tk.X, pady=2)
        ttk.Button(f, text="Chapter 4 figures", command=self._run_fig4).pack(fill=tk.X, pady=2)
        ttk.Button(f, text="Generate PDF + PowerPoint overview", command=self._run_overview).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(f, text="Open reports folder", command=self._open_reports).pack(
            fill=tk.X, pady=2
        )

    def _build_log_panel(self, parent: ttk.Frame) -> None:
        f = ttk.LabelFrame(parent, text="Activity log", padding=6)
        f.pack(fill=tk.BOTH, expand=True)
        self.log = scrolledtext.ScrolledText(
            f, wrap=tk.WORD, font=("Consolas", 9), state=tk.DISABLED
        )
        self.log.pack(fill=tk.BOTH, expand=True)
        ttk.Button(f, text="Clear log", command=self._clear_log).pack(anchor=tk.E, pady=(4, 0))

    # ── Logging & subprocess ──────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        self._log_queue.put(msg)

    def _poll_log(self) -> None:
        try:
            while True:
                line = self._log_queue.get_nowait()
                self.log.configure(state=tk.NORMAL)
                self.log.insert(tk.END, line + "\n")
                self.log.see(tk.END)
                self.log.configure(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.after(120, self._poll_log)

    def _clear_log(self) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)

    def _set_busy(self, busy: bool, status: str = "") -> None:
        self._busy = busy
        if status:
            self.status_var.set(status)
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()
            self._refresh_status()

    def _refresh_status(self) -> None:
        if self._busy:
            return
        parts = []
        csv = THESIS_DIR / "output" / "kpi_master_dataset.csv"
        if csv.is_file():
            try:
                n = sum(1 for _ in open(csv)) - 1
                parts.append(f"Dataset: {n:,} rows")
            except OSError:
                parts.append("Dataset: present")
        else:
            parts.append("Dataset: not yet created")
        if (THESIS_DIR / "models" / "lstm_model.h5").is_file():
            parts.append("Models: trained")
        if (THESIS_DIR / "reports" / "mapek_summary.json").is_file():
            parts.append("MAPE-K: done")
        self.status_var.set(" · ".join(parts) if parts else "Ready — run Setup first if this is a new machine")

    def _run_shell(self, cmd: list[str], cwd: Path | None = None, title: str = "Task") -> None:
        if self._busy:
            messagebox.showwarning("Busy", "Please wait for the current task to finish or click Stop.")
            return

        cwd = cwd or THESIS_DIR
        self._set_busy(True, f"Running: {title}…")
        self._log(f"\n{'='*60}\n▶ {title}\n  $ {' '.join(cmd)}\n  cwd: {cwd}\n{'='*60}")

        def worker() -> None:
            env = os.environ.copy()
            env["PYTHONNOUSERSITE"] = "1"
            env["THESIS_DIR"] = str(THESIS_DIR)
            bin_path = str(THESIS_DIR / "bin")
            env["PATH"] = f"{bin_path}:{env.get('PATH', '')}"
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    cwd=str(cwd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                )
                assert self._proc.stdout is not None
                for line in self._proc.stdout:
                    self._log(line.rstrip())
                code = self._proc.wait()
                self._log(f"\n✓ Finished (exit {code})" if code == 0 else f"\n✗ Failed (exit {code})")
                if code != 0:
                    self.after(0, lambda: messagebox.showerror(title, f"Task failed (exit code {code}). See log."))
                else:
                    self.after(0, lambda: messagebox.showinfo(title, "Completed successfully."))
            except Exception as e:
                self._log(f"ERROR: {e}")
                self.after(0, lambda: messagebox.showerror(title, str(e)))
            finally:
                self._proc = None
                self.after(0, lambda: self._set_busy(False))

        threading.Thread(target=worker, daemon=True).start()

    def _stop_task(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._log("⚠ Task stopped by user.")
            self._set_busy(False)
        else:
            messagebox.showinfo("Stop", "No task is currently running.")

    # ── Actions ───────────────────────────────────────────────────────────────

    def _run_setup(self) -> None:
        setup = REPO_DIR / "setup.sh"
        if not setup.is_file():
            messagebox.showerror("Setup", f"setup.sh not found in\n{REPO_DIR}")
            return
        if not messagebox.askyesno(
            "Setup",
            "This installs NS-3, Python packages, and builds simulators.\n"
            "It can take 30–60 minutes.\n\nContinue?",
        ):
            return
        self._run_shell(["bash", str(setup)], cwd=REPO_DIR, title="Full setup")

    def _check_env(self) -> None:
        script = thesis_script("check_environment.py")
        if not script.is_file():
            messagebox.showerror("Check", "check_environment.py not found. Run Setup first.")
            return
        self._run_shell([PYTHON, str(script)], title="Environment check")

    def _sim_args(self, debug: bool = False) -> list[str]:
        script = thesis_script("run_all_trials.py")
        args = [PYTHON, str(script)]
        mode = self.sim_mode.get()
        if mode == "lte":
            args.append("--lte")
        elif mode == "nr":
            args.append("--nr")
        if debug:
            args.append("--debug")
        else:
            try:
                args += ["--trials", str(int(self.trials_var.get()))]
                args += ["--workers", str(int(self.workers_var.get()))]
            except ValueError:
                raise ValueError("Trials and workers must be whole numbers.")
            if mode in ("lte", "nr"):
                args += ["--sim-time", str(int(self.simtime_var.get()))]
                args += ["--num-ues", str(int(self.ues_var.get()))]
        return args

    def _run_simulation(self) -> None:
        try:
            args = self._sim_args(debug=False)
        except ValueError as e:
            messagebox.showerror("Simulation", str(e))
            return
        mode = self.sim_mode.get()
        warn = ""
        if mode == "lte":
            warn = "LTE simulation can take many hours.\n"
        elif mode == "nr":
            warn = "NR simulation is very slow (days possible).\nUse Workers = 1.\n"
        if not messagebox.askyesno("Simulation", f"{warn}Start simulation now?"):
            return
        self._run_shell(args, title="NS-3 simulation")

    def _run_debug_trial(self) -> None:
        try:
            args = self._sim_args(debug=True)
        except ValueError as e:
            messagebox.showerror("Debug trial", str(e))
            return
        self._run_shell(args, title="Debug trial")

    def _run_training(self) -> None:
        script = thesis_script("preprocess_and_train.py")
        if not script.is_file():
            messagebox.showerror("Training", "preprocess_and_train.py not found.")
            return
        csv = THESIS_DIR / "output" / "kpi_master_dataset.csv"
        if not csv.is_file():
            if not messagebox.askyesno(
                "Training",
                "No kpi_master_dataset.csv found.\nRun simulation first.\n\nTry anyway?",
            ):
                return
        args = [PYTHON, str(script)]
        if self.skip_svm.get():
            args.append("--skip_svm")
        self._run_shell(args, title="ML training")

    def _run_mapek(self) -> None:
        script = thesis_script("mapek_loop.py")
        if not script.is_file():
            messagebox.showerror("MAPE-K", "mapek_loop.py not found.")
            return
        self._run_shell([PYTHON, str(script), "--model", "all"], title="MAPE-K evaluation")

    def _run_ch4(self) -> None:
        script = thesis_script("run_chapter4_pipeline.sh")
        if not script.is_file():
            script = REPO_DIR / "run_chapter4_pipeline.sh"
        if not script.is_file():
            messagebox.showerror("Chapter 4", "run_chapter4_pipeline.sh not found.")
            return
        self._run_shell(["bash", str(script)], title="Chapter 4 pipeline")

    def _run_fig3(self) -> None:
        script = thesis_script("scripts/generate_figures.py")
        if not script.is_file():
            script = REPO_DIR / "scripts" / "generate_figures.py"
        self._run_shell([PYTHON, str(script)], title="Chapter 3 figures")

    def _run_fig4(self) -> None:
        script = thesis_script("scripts/generate_chapter4_figures.py")
        if not script.is_file():
            script = REPO_DIR / "scripts" / "generate_chapter4_figures.py"
        self._run_shell([PYTHON, str(script)], title="Chapter 4 figures")

    def _run_overview(self) -> None:
        script = REPO_DIR / "scripts" / "generate_overview_docs.py"
        if not script.is_file():
            messagebox.showerror("Overview", "generate_overview_docs.py not found.")
            return
        py = PYTHON
        # overview generator may need python-pptx
        try:
            import pptx  # noqa: F401
        except ImportError:
            if messagebox.askyesno(
                "Overview",
                "python-pptx is required for PowerPoint.\nInstall it now? (one-time, ~1 min)",
            ):
                self._run_shell(
                    [PYTHON, "-m", "pip", "install", "python-pptx", "-q"],
                    cwd=REPO_DIR,
                    title="Install python-pptx",
                )
                return
        self._run_shell([py, str(script)], cwd=REPO_DIR, title="PDF + PowerPoint")
        out = REPO_DIR / "docs"
        self._log(f"\nFiles: {out / 'Project_Overview.pdf'}")
        self._log(f"       {out / 'Project_Overview.pptx'}")

    def _open_thesis_dir(self) -> None:
        path = THESIS_DIR
        path.mkdir(parents=True, exist_ok=True)
        self._open_path(path)

    def _open_reports(self) -> None:
        path = THESIS_DIR / "reports"
        path.mkdir(parents=True, exist_ok=True)
        self._open_path(path)

    def _open_path(self, path: Path) -> None:
        try:
            if sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", str(path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                os.startfile(str(path))  # type: ignore[attr-defined]
        except Exception as e:
            messagebox.showinfo("Open folder", f"{path}\n\n({e})")


def main() -> None:
    if not THESIS_DIR.is_dir():
        THESIS_DIR.mkdir(parents=True, exist_ok=True)
    app = ThesisGui()
    app.mainloop()


if __name__ == "__main__":
    main()
