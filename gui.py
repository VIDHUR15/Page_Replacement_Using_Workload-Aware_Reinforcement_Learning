"""
Desktop GUI for the workload-aware adaptive page replacement project.

Lets you pick a workload (or run all of them), set simulation parameters,
train both RL agents (windowed and per-eviction), and compare against
FIFO / LRU / LFU / MRU / OPT as a results table + hit-rate bar chart, with
a "gap to OPT closed" figure (the same relative metric used in published
caching-RL papers).

Run:
    python3 gui.py

Requires: tkinter (ships with most Python installs; on some Linux distros
install it separately with `sudo apt-get install python3-tk`) and
matplotlib (`pip install matplotlib`).
"""

import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from page_replacement.algorithms import run_fixed_policy, run_opt
from page_replacement.lecar import run_lecar
from page_replacement.adaptive_simulator import (
    run_adaptive_windowed, train_agent_windowed,
    run_adaptive_per_eviction, train_agent_per_eviction,
)
from page_replacement.rl_agent import QLearningPolicySelector
from page_replacement.workloads import WORKLOADS

BASELINE_POLICIES = ["FIFO", "LRU", "LFU", "MRU"]
ALL_POLICIES = BASELINE_POLICIES + ["OPT", "LeCaR", "RL-Adaptive (windowed)", "RL-Adaptive (per-eviction)"]
POLICY_COLORS = {
    "FIFO": "#6b7280", "LRU": "#3b82f6", "LFU": "#10b981",
    "MRU": "#f59e0b", "OPT": "#111827", "LeCaR": "#14b8a6",
    "RL-Adaptive (windowed)": "#a855f7", "RL-Adaptive (per-eviction)": "#ef4444",
}


def gap_to_opt_closed(fault_rate_lru, fault_rate_opt, fault_rate_policy):
    denom = fault_rate_lru - fault_rate_opt
    if abs(denom) < 1e-12:
        return None
    return (fault_rate_lru - fault_rate_policy) / denom * 100.0


class PageReplacementGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Workload-Aware Adaptive Page Replacement (RL)")
        self.geometry("1220x760")
        self.minsize(1040, 640)

        self._build_controls()
        self._build_results_area()
        self._worker_thread = None

    # ---------------------------------------------------------------- UI --

    def _build_controls(self):
        frame = ttk.LabelFrame(self, text="Simulation parameters")
        frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.workload_var = tk.StringVar(value="All")
        self.capacity_var = tk.IntVar(value=20)
        self.trace_len_var = tk.IntVar(value=3000)
        self.page_range_var = tk.IntVar(value=100)
        self.window_var = tk.IntVar(value=50)
        self.lookback_var = tk.IntVar(value=20)
        self.min_commit_var = tk.IntVar(value=20)
        self.episodes_var = tk.IntVar(value=25)

        def labeled(row, col, label, widget):
            ttk.Label(frame, text=label).grid(row=row, column=col * 2, sticky="w", padx=(10, 4), pady=6)
            widget.grid(row=row, column=col * 2 + 1, sticky="w", pady=6)

        workload_choices = ["All"] + list(WORKLOADS.keys())
        workload_menu = ttk.Combobox(frame, textvariable=self.workload_var,
                                      values=workload_choices, width=10, state="readonly")
        labeled(0, 0, "Workload:", workload_menu)
        labeled(0, 1, "Frame capacity:", ttk.Spinbox(frame, from_=2, to=200, textvariable=self.capacity_var, width=8))
        labeled(0, 2, "Trace length:", ttk.Spinbox(frame, from_=100, to=50000, increment=100, textvariable=self.trace_len_var, width=8))
        labeled(0, 3, "Page range:", ttk.Spinbox(frame, from_=10, to=5000, increment=10, textvariable=self.page_range_var, width=8))

        labeled(1, 0, "Window size (windowed RL):", ttk.Spinbox(frame, from_=5, to=1000, increment=5, textvariable=self.window_var, width=8))
        labeled(1, 1, "Lookback (per-eviction RL):", ttk.Spinbox(frame, from_=2, to=200, textvariable=self.lookback_var, width=8))
        labeled(1, 2, "Train episodes:", ttk.Spinbox(frame, from_=1, to=500, textvariable=self.episodes_var, width=8))
        labeled(1, 3, "Min commit (per-eviction RL):", ttk.Spinbox(frame, from_=1, to=200, textvariable=self.min_commit_var, width=8))

        self.run_button = ttk.Button(frame, text="Run comparison", command=self._on_run)
        self.run_button.grid(row=2, column=0, columnspan=2, padx=10, pady=(4, 10), sticky="w")

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(frame, textvariable=self.status_var, foreground="#555").grid(
            row=2, column=2, columnspan=6, sticky="w", pady=(4, 10))

    def _build_results_area(self):
        paned = ttk.PanedWindow(self, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # --- results table ---
        table_frame = ttk.Frame(paned)
        columns = ("workload", "policy", "hit_rate", "fault_rate", "faults",
                   "gap_to_opt", "exec_ms", "train_ms")
        headings = {"workload": "Workload", "policy": "Policy", "hit_rate": "Hit rate",
                    "fault_rate": "Fault rate", "faults": "Faults",
                    "gap_to_opt": "Gap to OPT closed", "exec_ms": "Exec (ms)",
                    "train_ms": "Train (ms)"}
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=12)
        for c in columns:
            self.tree.heading(c, text=headings[c])
            self.tree.column(c, width=110, anchor="center")
        self.tree.column("policy", width=190)
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        paned.add(table_frame, weight=1)

        # --- chart ---
        chart_frame = ttk.Frame(paned)
        self.figure = Figure(figsize=(6, 3.2), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        paned.add(chart_frame, weight=1)

        # --- RL policy usage summary ---
        self.usage_var = tk.StringVar(value="")
        usage_label = ttk.Label(self, textvariable=self.usage_var, foreground="#333",
                                 justify="left", wraplength=1040)
        usage_label.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=(0, 10))

    # ------------------------------------------------------------ actions --

    def _on_run(self):
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self.run_button.config(state=tk.DISABLED)
        self.status_var.set("Running...")
        self.usage_var.set("")
        for row in self.tree.get_children():
            self.tree.delete(row)

        params = dict(
            workload=self.workload_var.get(),
            capacity=self.capacity_var.get(),
            trace_len=self.trace_len_var.get(),
            page_range=self.page_range_var.get(),
            window=self.window_var.get(),
            lookback=self.lookback_var.get(),
            min_commit=self.min_commit_var.get(),
            episodes=self.episodes_var.get(),
        )
        self._worker_thread = threading.Thread(target=self._run_evaluation, args=(params,), daemon=True)
        self._worker_thread.start()

    def _run_evaluation(self, params):
        try:
            names = list(WORKLOADS.keys()) if params["workload"] == "All" else [params["workload"]]
            all_rows = []
            usage_lines = []
            for name in names:
                self._set_status(f"Evaluating '{name}'...")
                rows, usage_w, usage_pe = self._evaluate_workload(name, params)
                all_rows.extend(rows)
                usage_lines.append(f"{name} — windowed: {usage_w}  |  per-eviction: {usage_pe}")
            self.after(0, self._show_results, all_rows, usage_lines)
        except Exception as exc:  # surface errors in the GUI instead of a silent thread crash
            self.after(0, self._show_error, str(exc))

    def _evaluate_workload(self, name, params):
        gen = WORKLOADS[name]
        capacity, trace_len, page_range, window, lookback, episodes, min_commit = (
            params["capacity"], params["trace_len"], params["page_range"],
            params["window"], params["lookback"], params["episodes"], params["min_commit"])

        eval_ref = gen(trace_len, page_range, seed=999_999)
        rows = []

        for policy in BASELINE_POLICIES:
            t0 = time.perf_counter()
            r = run_fixed_policy(eval_ref, capacity, policy)
            r["exec_time_s"] = time.perf_counter() - t0
            r["train_time_s"] = 0.0
            r["workload"] = name
            rows.append(r)

        t0 = time.perf_counter()
        opt_r = run_opt(eval_ref, capacity)
        opt_r["exec_time_s"] = time.perf_counter() - t0
        opt_r["train_time_s"] = 0.0
        opt_r["workload"] = name
        rows.append(opt_r)

        t0 = time.perf_counter()
        lecar_r = run_lecar(eval_ref, capacity, seed=0)
        lecar_r["exec_time_s"] = time.perf_counter() - t0
        lecar_r["train_time_s"] = 0.0
        lecar_r["workload"] = name
        rows.append(lecar_r)

        def factory(seed=None):
            return gen(trace_len, page_range, seed=seed)

        agent_w = QLearningPolicySelector(seed=42)
        t_train0 = time.perf_counter()
        train_agent_windowed(factory, capacity, window, episodes, agent=agent_w)
        train_time_w = time.perf_counter() - t_train0
        t0 = time.perf_counter()
        r_w = run_adaptive_windowed(eval_ref, capacity, window, agent_w, train=False)
        r_w["exec_time_s"] = time.perf_counter() - t0
        r_w["train_time_s"] = train_time_w
        r_w["workload"] = name
        rows.append(r_w)

        agent_pe = QLearningPolicySelector(seed=42)
        t_train0 = time.perf_counter()
        train_agent_per_eviction(factory, capacity, lookback, episodes, agent=agent_pe,
                                  min_commit=min_commit)
        train_time_pe = time.perf_counter() - t_train0
        t0 = time.perf_counter()
        r_pe = run_adaptive_per_eviction(eval_ref, capacity, lookback, agent_pe, train=False,
                                          min_commit=min_commit)
        r_pe["exec_time_s"] = time.perf_counter() - t0
        r_pe["train_time_s"] = train_time_pe
        r_pe["workload"] = name
        rows.append(r_pe)

        lru_fr = next(r["fault_rate"] for r in rows if r["policy"] == "LRU")
        opt_fr = opt_r["fault_rate"]
        for r in rows:
            r["gap_to_opt_pct"] = gap_to_opt_closed(lru_fr, opt_fr, r["fault_rate"])

        return rows, r_w["policy_usage"], r_pe["policy_usage"]

    def _set_status(self, text):
        self.after(0, lambda: self.status_var.set(text))

    def _show_error(self, message):
        self.run_button.config(state=tk.NORMAL)
        self.status_var.set("Error.")
        messagebox.showerror("Simulation error", message)

    def _show_results(self, rows, usage_lines):
        for r in rows:
            gap = r.get("gap_to_opt_pct")
            gap_str = f"{gap:.1f}%" if gap is not None else "--"
            self.tree.insert("", tk.END, values=(
                r["workload"], r["policy"], f"{r['hit_rate']:.3f}", f"{r['fault_rate']:.3f}",
                r["faults"], gap_str, f"{r['exec_time_s']*1000:.2f}",
                f"{r.get('train_time_s', 0)*1000:.1f}",
            ))

        self._draw_chart(rows)
        self.usage_var.set("RL policy usage (per workload) — " + "   //   ".join(usage_lines))
        self.status_var.set(f"Done. {len(rows)} rows.")
        self.run_button.config(state=tk.NORMAL)

    def _draw_chart(self, rows):
        self.ax.clear()
        workloads = sorted(set(r["workload"] for r in rows))
        n_policies = len(ALL_POLICIES)
        width = 0.8 / n_policies

        for i, policy in enumerate(ALL_POLICIES):
            xs = []
            ys = []
            for wi, wl in enumerate(workloads):
                match = next((r for r in rows if r["workload"] == wl and r["policy"] == policy), None)
                if match:
                    xs.append(wi + i * width)
                    ys.append(match["hit_rate"])
            self.ax.bar(xs, ys, width=width, label=policy, color=POLICY_COLORS.get(policy))

        self.ax.set_xticks([wi + width * (n_policies - 1) / 2 for wi in range(len(workloads))])
        self.ax.set_xticklabels(workloads, rotation=0)
        self.ax.set_ylabel("Hit rate")
        self.ax.set_ylim(0, 1)
        self.ax.set_title("Hit rate by policy and workload (OPT = theoretical ceiling)")
        self.ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = PageReplacementGUI()
    app.mainloop()
