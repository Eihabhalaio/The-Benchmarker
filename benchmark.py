#!/usr/bin/env python3
"""
The-Benchmarker — measure LLM inference performance on your hardware.
https://github.com/Eihabhalaio/The-Benchmarker
"""

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from io import StringIO
from pathlib import Path

import psutil
import requests
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

console = Console()

OLLAMA_BASE = "http://localhost:11434"

# ------------------------------------------------------------------
# Benchmark prompts (same across all models for fair comparison)
# ------------------------------------------------------------------
BENCHMARKS = [
    {
        "id": "short",
        "label": "Short (haiku)",
        "prompt": "Write a haiku about artificial intelligence.",
    },
    {
        "id": "reasoning",
        "label": "Reasoning",
        "prompt": "Explain how a CPU works in exactly 3 sentences.",
    },
    {
        "id": "code",
        "label": "Code generation",
        "prompt": (
            "Write a Python function that computes the nth Fibonacci number "
            "using memoization. Include a docstring and example usage."
        ),
    },
    {
        "id": "long",
        "label": "Long generation",
        "prompt": (
            "Write a detailed story (at least 300 words) about a robot that "
            "learns to paint. Include vivid color descriptions, the robot's "
            "emotions, and its creative process."
        ),
    },
]

# Popular models with sizes for reference
DEFAULT_MODELS = [
    "llama3.2:3b",
    "llama3.1:8b",
]

SUGGESTED_MODELS = [
    ("llama3.2:3b",   "2.0 GB",  "Fast, good for quick tasks"),
    ("llama3.1:8b",   "4.9 GB",  "Balanced quality/speed"),
    ("mistral:7b",    "4.1 GB",  "Strong reasoning"),
    ("codellama:7b",  "3.8 GB",  "Code-focused"),
    ("phi3:mini",     "2.2 GB",  "Microsoft, very fast"),
    ("gemma2:9b",     "5.5 GB",  "Google, high quality"),
    ("llama3.1:70b",  "40 GB",   "Requires high VRAM / RAM"),
]


# ------------------------------------------------------------------
# System information
# ------------------------------------------------------------------
def get_system_info() -> dict:
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "python": platform.python_version(),
        "cpu": _get_cpu_name(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "gpu": None,
        "gpu_vram_gb": None,
        "cuda_version": None,
    }
    gpu = _get_gpu_info()
    info.update(gpu)
    return info


def _get_cpu_name() -> str:
    if platform.system() == "Windows":
        return platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown"


def _get_gpu_info() -> dict:
    result = {"gpu": None, "gpu_vram_gb": None, "cuda_version": None, "gpu_vendor": None}
    # NVIDIA via nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"],
                text=True, stderr=subprocess.DEVNULL
            ).strip().splitlines()
            if out:
                name, vram, _ = out[0].split(", ")
                result["gpu"] = name.strip()
                result["gpu_vram_gb"] = round(int(vram.replace(" MiB", "")) / 1024, 1)
                result["gpu_vendor"] = "nvidia"
                hdr = subprocess.check_output(
                    ["nvidia-smi"], text=True, stderr=subprocess.DEVNULL
                )
                for line in hdr.splitlines():
                    if "CUDA Version:" in line:
                        result["cuda_version"] = line.split("CUDA Version:")[1].strip().split()[0]
                        break
        except Exception:
            pass
    # AMD via rocm-smi
    elif shutil.which("rocm-smi"):
        try:
            out = subprocess.check_output(
                ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"],
                text=True, stderr=subprocess.DEVNULL
            )
            data = json.loads(out)
            for card_key, card_data in data.items():
                if not card_key.startswith("card"):
                    continue
                result["gpu"] = card_data.get("Card Series") or card_data.get("Card Model") or "AMD GPU"
                total_bytes = int(card_data.get("VRAM Total Memory (B)", 0))
                if total_bytes:
                    result["gpu_vram_gb"] = round(total_bytes / 1024**3, 1)
                result["gpu_vendor"] = "amd"
                break
        except Exception:
            # Fallback: just get name
            try:
                out = subprocess.check_output(
                    ["rocm-smi", "--showproductname"], text=True, stderr=subprocess.DEVNULL
                )
                for line in out.splitlines():
                    if "GPU" in line and ":" in line:
                        result["gpu"] = line.split(":")[1].strip()
                        result["gpu_vendor"] = "amd"
                        break
            except Exception:
                pass
    # Apple Silicon via system_profiler
    elif platform.system() == "Darwin":
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                text=True, stderr=subprocess.DEVNULL
            )
            data = json.loads(out)
            for display in data.get("SPDisplaysDataType", []):
                model = display.get("sppci_model", "")
                if "Apple" in model:
                    result["gpu"] = model
                    result["gpu_vendor"] = "apple"
                    # Unified memory — fetch from hardware overview
                    hw_out = subprocess.check_output(
                        ["system_profiler", "SPHardwareDataType", "-json"],
                        text=True, stderr=subprocess.DEVNULL
                    )
                    hw = json.loads(hw_out)
                    for item in hw.get("SPHardwareDataType", []):
                        mem = item.get("physical_memory", "")
                        if mem:
                            gb = mem.lower().replace("gb", "").strip()
                            try:
                                result["gpu_vram_gb"] = float(gb)  # unified memory
                            except ValueError:
                                pass
                    break
        except Exception:
            pass
    return result


def _get_gpu_live() -> dict:
    """Real-time GPU stats during inference."""
    stats = {"vram_used_mb": None, "gpu_util_pct": None, "gpu_temp_c": None}
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL
            ).strip()
            parts = [p.strip() for p in out.split(",")]
            if len(parts) == 3:
                stats["vram_used_mb"] = int(parts[0])
                stats["gpu_util_pct"] = int(parts[1])
                stats["gpu_temp_c"]   = int(parts[2])
        except Exception:
            pass
    elif shutil.which("rocm-smi"):
        try:
            out = subprocess.check_output(
                ["rocm-smi", "--showmeminfo", "vram", "--showuse", "--showtemp", "--json"],
                text=True, stderr=subprocess.DEVNULL
            )
            data = json.loads(out)
            for card_key, card_data in data.items():
                if not card_key.startswith("card"):
                    continue
                used_bytes = int(card_data.get("VRAM Total Used Memory (B)", 0))
                if used_bytes:
                    stats["vram_used_mb"] = used_bytes // 1024 // 1024
                util = card_data.get("GPU use (%)")
                if util is not None:
                    stats["gpu_util_pct"] = int(util)
                temp = card_data.get("Temperature (Sensor edge) (C)")
                if temp is not None:
                    stats["gpu_temp_c"] = int(float(temp))
                break
        except Exception:
            pass
    return stats


# ------------------------------------------------------------------
# Ollama helpers
# ------------------------------------------------------------------
def check_ollama() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def list_local_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def pull_model(model: str) -> bool:
    console.print(f"  [cyan]Pulling[/cyan] [bold]{model}[/bold] …")
    try:
        with requests.post(
            f"{OLLAMA_BASE}/api/pull",
            json={"name": model},
            stream=True,
            timeout=600,
        ) as resp:
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")
                    if "pulling" in status and "completed" in data and "total" in data:
                        pct = int(data["completed"] / data["total"] * 100)
                        console.print(f"    {status} {pct}%", end="\r")
                    elif status == "success":
                        console.print(f"  [green]✓[/green] {model} ready        ")
                        return True
        return True
    except Exception as e:
        console.print(f"  [red]✗[/red] Failed to pull {model}: {e}")
        return False


# ------------------------------------------------------------------
# Core benchmark runner
# ------------------------------------------------------------------
def run_single_benchmark(model: str, benchmark: dict) -> dict:
    payload = {
        "model": model,
        "prompt": benchmark["prompt"],
        "stream": False,
    }
    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json=payload,
            timeout=300,
        )
        elapsed = time.perf_counter() - start
        data = resp.json()
    except Exception as e:
        return {"error": str(e)}

    gpu = _get_gpu_live()

    prompt_tokens = data.get("prompt_eval_count", 0)
    gen_tokens    = data.get("eval_count", 0)
    prompt_ns     = data.get("prompt_eval_duration", 1)
    gen_ns        = data.get("eval_duration", 1)
    total_ns      = data.get("total_duration", 1)
    load_ns       = data.get("load_duration", 0)

    return {
        "benchmark_id":      benchmark["id"],
        "benchmark_label":   benchmark["label"],
        "prompt_tokens":     prompt_tokens,
        "gen_tokens":        gen_tokens,
        "prompt_tok_per_s":  round(prompt_tokens / (prompt_ns / 1e9), 1) if prompt_ns else 0,
        "gen_tok_per_s":     round(gen_tokens    / (gen_ns    / 1e9), 1) if gen_ns    else 0,
        "total_s":           round(total_ns / 1e9, 2),
        "load_s":            round(load_ns  / 1e9, 3),
        "wall_s":            round(elapsed, 2),
        "response_preview":  data.get("response", "")[:200],
        **gpu,
    }


def benchmark_model(model: str, runs: list[dict], progress, task) -> dict:
    results = []
    for bench in runs:
        progress.update(task, description=f"[cyan]{model}[/cyan] → {bench['label']}")
        r = run_single_benchmark(model, bench)
        results.append(r)
        progress.advance(task)
    return {"model": model, "benchmarks": results}


# ------------------------------------------------------------------
# Display helpers
# ------------------------------------------------------------------
def print_system_panel(info: dict):
    lines = [
        f"[bold]OS[/bold]          {info['os']} ({info['arch']})",
        f"[bold]CPU[/bold]         {info['cpu']}",
        f"[bold]CPU Cores[/bold]   {info['cpu_cores']} cores / {info['cpu_threads']} threads",
        f"[bold]RAM[/bold]         {info['ram_gb']} GB",
    ]
    if info["gpu"]:
        lines.append(f"[bold]GPU[/bold]         {info['gpu']}")
        lines.append(f"[bold]VRAM[/bold]        {info['gpu_vram_gb']} GB")
    if info["cuda_version"]:
        lines.append(f"[bold]CUDA[/bold]        {info['cuda_version']}")
    console.print(Panel("\n".join(lines), title="[bold yellow]🖥  System Info[/bold yellow]", expand=False))


def build_results_table(all_results: list[dict]) -> Table:
    table = Table(
        title="[bold green]📊 Benchmark Results[/bold green]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("Model",        style="cyan",  no_wrap=True)
    table.add_column("Benchmark",    style="white")
    table.add_column("Gen tok/s",    style="bold green",  justify="right")
    table.add_column("Prompt tok/s", style="green",       justify="right")
    table.add_column("Gen tokens",   style="yellow",      justify="right")
    table.add_column("Total (s)",    style="yellow",      justify="right")
    table.add_column("VRAM used",    style="blue",        justify="right")
    table.add_column("GPU util %",   style="blue",        justify="right")
    table.add_column("GPU °C",       style="blue",        justify="right")

    for entry in all_results:
        model = entry["model"]
        for i, b in enumerate(entry["benchmarks"]):
            if "error" in b:
                table.add_row(model if i == 0 else "", b["benchmark_label"],
                              "[red]ERROR[/red]", "", "", "", "", "", "")
                continue
            vram = f"{b['vram_used_mb']} MB" if b.get("vram_used_mb") else "—"
            util = f"{b['gpu_util_pct']}%"    if b.get("gpu_util_pct") is not None else "—"
            temp = f"{b['gpu_temp_c']}°C"     if b.get("gpu_temp_c")  is not None else "—"
            table.add_row(
                model if i == 0 else "",
                b["benchmark_label"],
                str(b["gen_tok_per_s"]),
                str(b["prompt_tok_per_s"]),
                str(b["gen_tokens"]),
                str(b["total_s"]),
                vram, util, temp,
            )
        table.add_section()
    return table


def build_summary_table(all_results: list[dict]) -> Table:
    table = Table(
        title="[bold green]⚡ Summary — Average Generation Speed[/bold green]",
        box=box.SIMPLE_HEAVY,
        header_style="bold magenta",
        expand=False,
    )
    table.add_column("Model",         style="cyan", no_wrap=True)
    table.add_column("Avg tok/s",     style="bold green", justify="right")
    table.add_column("Best tok/s",    style="green",      justify="right")
    table.add_column("Avg total (s)", style="yellow",     justify="right")
    table.add_column("Avg VRAM (MB)", style="blue",       justify="right")

    for entry in all_results:
        benches = [b for b in entry["benchmarks"] if "error" not in b]
        if not benches:
            continue
        avg_tok  = round(sum(b["gen_tok_per_s"] for b in benches) / len(benches), 1)
        best_tok = max(b["gen_tok_per_s"] for b in benches)
        avg_tot  = round(sum(b["total_s"] for b in benches) / len(benches), 2)
        vrams    = [b["vram_used_mb"] for b in benches if b.get("vram_used_mb")]
        avg_vram = round(sum(vrams) / len(vrams)) if vrams else None
        table.add_row(
            entry["model"],
            str(avg_tok),
            str(best_tok),
            str(avg_tot),
            str(avg_vram) if avg_vram else "—",
        )
    return table


# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------
def save_results(system_info: dict, all_results: list[dict], output_dir: str = "results"):
    Path(output_dir).mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"bench_{ts}.json"
    payload = {
        "timestamp":   datetime.now().isoformat(),
        "system":      system_info,
        "results":     all_results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return str(path)


def generate_markdown_report(system_info: dict, all_results: list[dict]) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# The-Benchmarker Report",
        f"*Generated: {ts}*\n",
        "## System",
        "| Component | Spec |",
        "|-----------|------|",
        f"| OS | {system_info['os']} ({system_info['arch']}) |",
        f"| CPU | {system_info['cpu']} |",
        f"| Cores | {system_info['cpu_cores']} cores / {system_info['cpu_threads']} threads |",
        f"| RAM | {system_info['ram_gb']} GB |",
    ]
    if system_info.get("gpu"):
        lines.append(f"| GPU | {system_info['gpu']} |")
        lines.append(f"| VRAM | {system_info['gpu_vram_gb']} GB |")
    if system_info.get("cuda_version"):
        lines.append(f"| CUDA | {system_info['cuda_version']} |")
    lines += [
        "",
        "## Results",
        "| Model | Benchmark | Gen tok/s | Prompt tok/s | Total (s) | VRAM used |",
        "|-------|-----------|----------:|-------------:|----------:|-----------|",
    ]
    for entry in all_results:
        for b in entry["benchmarks"]:
            if "error" in b:
                continue
            vram = f"{b['vram_used_mb']} MB" if b.get("vram_used_mb") else "—"
            lines.append(
                f"| {entry['model']} | {b['benchmark_label']} | "
                f"{b['gen_tok_per_s']} | {b['prompt_tok_per_s']} | "
                f"{b['total_s']} | {vram} |"
            )
    return "\n".join(lines)


def generate_csv_report(system_info: dict, all_results: list[dict]) -> str:
    """Return CSV string with one row per benchmark result."""
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "timestamp", "cpu", "gpu", "gpu_vram_gb",
        "model", "benchmark", "gen_tok_per_s", "prompt_tok_per_s",
        "gen_tokens", "total_s", "vram_used_mb", "gpu_util_pct", "gpu_temp_c",
    ])
    ts = datetime.now().isoformat()
    for entry in all_results:
        for b in entry["benchmarks"]:
            if "error" in b:
                continue
            writer.writerow([
                ts,
                system_info.get("cpu", ""),
                system_info.get("gpu", ""),
                system_info.get("gpu_vram_gb", ""),
                entry["model"],
                b["benchmark_label"],
                b.get("gen_tok_per_s", ""),
                b.get("prompt_tok_per_s", ""),
                b.get("gen_tokens", ""),
                b.get("total_s", ""),
                b.get("vram_used_mb", ""),
                b.get("gpu_util_pct", ""),
                b.get("gpu_temp_c", ""),
            ])
    return buf.getvalue()


def generate_html_report(system_info: dict, all_results: list[dict]) -> str:
    """Generate a self-contained HTML report with Chart.js bar charts."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build chart data
    models  = [r["model"] for r in all_results]
    bench_ids = [b["id"] for b in BENCHMARKS]
    bench_labels = [b["label"] for b in BENCHMARKS]
    colors = ["#4ade80", "#60a5fa", "#f59e0b", "#f87171", "#a78bfa", "#34d399"]

    # gen_tok_per_s[model][benchmark]
    datasets = []
    for bi, blabel in enumerate(bench_labels):
        data_points = []
        for entry in all_results:
            bench = next((b for b in entry["benchmarks"] if b.get("benchmark_label") == blabel), None)
            data_points.append(bench["gen_tok_per_s"] if bench and "error" not in bench else 0)
        datasets.append({
            "label": blabel,
            "data": data_points,
            "backgroundColor": colors[bi % len(colors)],
        })

    chart_data = json.dumps({"labels": models, "datasets": datasets})

    # Summary rows
    summary_rows = ""
    for entry in all_results:
        benches = [b for b in entry["benchmarks"] if "error" not in b]
        if not benches:
            continue
        avg_tok  = round(sum(b["gen_tok_per_s"] for b in benches) / len(benches), 1)
        best_tok = max(b["gen_tok_per_s"] for b in benches)
        avg_tot  = round(sum(b["total_s"] for b in benches) / len(benches), 2)
        vrams    = [b["vram_used_mb"] for b in benches if b.get("vram_used_mb")]
        avg_vram = f"{round(sum(vrams)/len(vrams))} MB" if vrams else "—"
        summary_rows += (
            f"<tr><td>{entry['model']}</td><td><b>{avg_tok}</b></td>"
            f"<td>{best_tok}</td><td>{avg_tot}</td><td>{avg_vram}</td></tr>\n"
        )

    # Detail rows
    detail_rows = ""
    for entry in all_results:
        for b in entry["benchmarks"]:
            if "error" in b:
                continue
            vram = f"{b['vram_used_mb']} MB" if b.get("vram_used_mb") else "—"
            util = f"{b['gpu_util_pct']}%" if b.get("gpu_util_pct") is not None else "—"
            temp = f"{b['gpu_temp_c']}°C"  if b.get("gpu_temp_c") is not None else "—"
            detail_rows += (
                f"<tr><td>{entry['model']}</td><td>{b['benchmark_label']}</td>"
                f"<td><b>{b['gen_tok_per_s']}</b></td><td>{b['prompt_tok_per_s']}</td>"
                f"<td>{b['gen_tokens']}</td><td>{b['total_s']}</td>"
                f"<td>{vram}</td><td>{util}</td><td>{temp}</td></tr>\n"
            )

    sys_rows = ""
    for k, label in [
        ("os", "OS"), ("cpu", "CPU"),
        ("cpu_cores", "Cores"), ("ram_gb", "RAM (GB)"),
        ("gpu", "GPU"), ("gpu_vram_gb", "VRAM (GB)"), ("cuda_version", "CUDA"),
    ]:
        v = system_info.get(k)
        if v:
            sys_rows += f"<tr><td>{label}</td><td>{v}</td></tr>\n"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>The-Benchmarker Report — {ts}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {{ --bg:#0f172a; --card:#1e293b; --border:#334155; --green:#4ade80;
           --text:#e2e8f0; --muted:#94a3b8; --accent:#60a5fa; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'Segoe UI',system-ui,sans-serif; background:var(--bg);
          color:var(--text); padding:2rem; }}
  h1 {{ font-size:1.8rem; color:var(--green); margin-bottom:.25rem; }}
  .subtitle {{ color:var(--muted); margin-bottom:2rem; }}
  h2 {{ font-size:1.1rem; color:var(--accent); margin:2rem 0 .75rem; text-transform:uppercase;
        letter-spacing:.05em; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:1rem;
           margin-bottom:2rem; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:.75rem;
           padding:1.25rem; }}
  .card .label {{ font-size:.75rem; color:var(--muted); text-transform:uppercase; margin-bottom:.25rem; }}
  .card .value {{ font-size:1.05rem; font-weight:600; }}
  .chart-wrap {{ background:var(--card); border:1px solid var(--border); border-radius:.75rem;
                 padding:1.5rem; margin-bottom:2rem; max-height:420px; }}
  table {{ width:100%; border-collapse:collapse; background:var(--card);
           border:1px solid var(--border); border-radius:.75rem; overflow:hidden; }}
  th {{ background:#0f172a; color:var(--accent); text-align:left; padding:.65rem 1rem;
        font-size:.8rem; text-transform:uppercase; letter-spacing:.04em; }}
  td {{ padding:.6rem 1rem; border-top:1px solid var(--border); font-size:.9rem; }}
  tr:hover td {{ background:#263048; }}
  b {{ color:var(--green); }}
  .tag {{ display:inline-block; background:#1e3a5f; color:var(--accent); border-radius:99px;
          padding:.15rem .6rem; font-size:.75rem; }}
  footer {{ margin-top:3rem; color:var(--muted); font-size:.8rem; text-align:center; }}
</style>
</head>
<body>
<h1>🚀 The-Benchmarker</h1>
<p class="subtitle">Generated: {ts} &nbsp;|&nbsp;
  <a href="https://github.com/Eihabhalaio/The-Benchmarker" style="color:var(--accent)">
    github.com/Eihabhalaio/The-Benchmarker</a></p>

<h2>System</h2>
<table style="margin-bottom:2rem">
  <tr><th>Component</th><th>Spec</th></tr>
  {sys_rows}
</table>

<h2>Generation Speed (tok/s)</h2>
<div class="chart-wrap">
  <canvas id="speedChart"></canvas>
</div>

<h2>Summary</h2>
<table style="margin-bottom:2rem">
  <tr><th>Model</th><th>Avg tok/s</th><th>Best tok/s</th><th>Avg total (s)</th><th>Avg VRAM</th></tr>
  {summary_rows}
</table>

<h2>Full Results</h2>
<table>
  <tr><th>Model</th><th>Benchmark</th><th>Gen tok/s</th><th>Prompt tok/s</th>
      <th>Gen tokens</th><th>Total (s)</th><th>VRAM used</th><th>GPU util</th><th>GPU °C</th></tr>
  {detail_rows}
</table>

<footer>The-Benchmarker &nbsp;·&nbsp; MIT License</footer>

<script>
const ctx = document.getElementById('speedChart');
new Chart(ctx, {{
  type: 'bar',
  data: {chart_data},
  options: {{
    responsive: true,
    maintainAspectRatio: true,
    plugins: {{
      legend: {{ labels: {{ color: '#e2e8f0' }} }},
      tooltip: {{ callbacks: {{ label: (c) => ` ${{c.dataset.label}}: ${{c.raw}} tok/s` }} }}
    }},
    scales: {{
      x: {{ ticks: {{ color:'#94a3b8' }}, grid: {{ color:'#334155' }} }},
      y: {{ ticks: {{ color:'#94a3b8' }}, grid: {{ color:'#334155' }},
             title: {{ display:true, text:'tokens / second', color:'#94a3b8' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""


def upload_to_gist(system_info: dict, all_results: list[dict], json_path: str) -> str | None:
    """Upload results to a public GitHub Gist. Returns the gist URL or None."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        console.print(
            "\n[yellow]ℹ  To upload to the leaderboard, set[/yellow] "
            "[bold]GITHUB_TOKEN[/bold] [yellow]in your environment.[/yellow]\n"
            "  export GITHUB_TOKEN=ghp_yourtoken\n"
            "  Get one at: https://github.com/settings/tokens (no scopes needed for public gists)"
        )
        return None

    cpu  = system_info.get("cpu", "Unknown CPU")
    gpu  = system_info.get("gpu", "No GPU")
    desc = f"The-Benchmarker | {cpu} | {gpu}"
    models = ", ".join(r["model"] for r in all_results)

    with open(json_path) as f:
        json_content = f.read()

    md_content = generate_markdown_report(system_info, all_results)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    payload = {
        "description": desc,
        "public": True,
        "files": {
            f"ollama-bench-{ts}.json": {"content": json_content},
            f"ollama-bench-{ts}.md":   {"content": md_content},
        },
    }
    try:
        resp = requests.post(
            "https://api.github.com/gists",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("html_url")
    except Exception as e:
        console.print(f"  [red]✗[/red] Upload failed: {e}")
        return None


def show_leaderboard(output_dir: str):
    """Load all local result files and display a ranked leaderboard."""
    result_files = sorted(Path(output_dir).glob("bench_*.json"))
    if not result_files:
        console.print(f"[yellow]No result files found in {output_dir}/[/yellow]")
        return

    rows = []
    for fp in result_files:
        try:
            with open(fp) as f:
                data = json.load(f)
            sys_info = data.get("system", {})
            ts = data.get("timestamp", "")[:16].replace("T", " ")
            for entry in data.get("results", []):
                benches = [b for b in entry.get("benchmarks", []) if "error" not in b]
                if not benches:
                    continue
                avg_tok  = round(sum(b["gen_tok_per_s"] for b in benches) / len(benches), 1)
                best_tok = max(b["gen_tok_per_s"] for b in benches)
                rows.append({
                    "ts":       ts,
                    "model":    entry["model"],
                    "cpu":      sys_info.get("cpu", "?")[:40],
                    "gpu":      sys_info.get("gpu", "CPU only")[:35],
                    "avg_tok":  avg_tok,
                    "best_tok": best_tok,
                    "file":     fp.name,
                })
        except Exception:
            continue

    rows.sort(key=lambda r: r["avg_tok"], reverse=True)

    table = Table(
        title="[bold green]🏆 Local Leaderboard[/bold green]",
        box=box.ROUNDED,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("#",         style="dim",         justify="right", width=3)
    table.add_column("Model",     style="cyan",        no_wrap=True)
    table.add_column("Avg tok/s", style="bold green",  justify="right")
    table.add_column("Best tok/s",style="green",       justify="right")
    table.add_column("GPU",       style="blue")
    table.add_column("CPU",       style="white")
    table.add_column("Date",      style="dim")

    for i, r in enumerate(rows, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, str(i))
        table.add_row(medal, r["model"], str(r["avg_tok"]), str(r["best_tok"]),
                      r["gpu"], r["cpu"], r["ts"])

    console.print()
    console.print(table)
    console.print(f"\n[dim]{len(result_files)} result file(s) in {output_dir}/[/dim]")
    console.print("[dim]Share your results by uploading with --upload[/dim]")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Ollama LLM inference performance on your hardware.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py                          # interactive mode
  python benchmark.py -m llama3.2:3b           # specific model
  python benchmark.py -m llama3.2:3b mistral:7b  # compare models
  python benchmark.py --quick                  # short benchmark only
  python benchmark.py --export-html            # save HTML report with charts
  python benchmark.py --export-csv             # save CSV for spreadsheet analysis
  python benchmark.py --export-md              # save Markdown report
  python benchmark.py --upload                 # upload to GitHub Gist leaderboard
  python benchmark.py --leaderboard            # view local results leaderboard
        """,
    )
    parser.add_argument("-m", "--models",    nargs="+", help="Models to benchmark")
    parser.add_argument("--quick",           action="store_true", help="Run only the short benchmark")
    parser.add_argument("--export-md",       action="store_true", help="Export a Markdown report")
    parser.add_argument("--export-html",     action="store_true", help="Export a self-contained HTML report with charts")
    parser.add_argument("--export-csv",      action="store_true", help="Export results as CSV")
    parser.add_argument("--upload",          action="store_true", help="Upload results to GitHub Gist (requires GITHUB_TOKEN)")
    parser.add_argument("--leaderboard",     action="store_true", help="Show local results leaderboard and exit")
    parser.add_argument("--output",          default="results",   help="Output directory (default: results/)")
    parser.add_argument("--url",             default=OLLAMA_BASE, help=f"Ollama base URL (default: {OLLAMA_BASE})")
    return parser.parse_args()


def interactive_model_select(local_models: list[str]) -> list[str]:
    console.print("\n[bold yellow]Available models on this machine:[/bold yellow]")
    if local_models:
        for i, m in enumerate(local_models, 1):
            console.print(f"  [cyan]{i}.[/cyan] {m}")
    else:
        console.print("  [dim]None yet[/dim]")

    console.print("\n[bold yellow]Suggested models to pull:[/bold yellow]")
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("#")
    table.add_column("Model")
    table.add_column("Size")
    table.add_column("Notes")
    for i, (name, size, note) in enumerate(SUGGESTED_MODELS, len(local_models) + 1):
        already = " [green](local)[/green]" if name in local_models else ""
        table.add_row(str(i), name + already, size, note)
    console.print(table)

    all_options = local_models + [m[0] for m in SUGGESTED_MODELS if m[0] not in local_models]

    raw = Prompt.ask(
        "\nEnter model numbers or names (comma-separated, e.g. [cyan]1,2[/cyan] or [cyan]llama3.2:3b[/cyan])",
        default="1,2" if len(local_models) >= 2 else "1",
    )
    selected = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(all_options):
                selected.append(all_options[idx])
        elif part:
            selected.append(part)
    return list(dict.fromkeys(selected))  # deduplicate, preserve order


def main():
    global OLLAMA_BASE

    args = parse_args()
    OLLAMA_BASE = args.url

    console.print(Panel.fit(
        "[bold green]🚀 The-Benchmarker[/bold green]\n"
        "[dim]Measure LLM inference speed on your hardware[/dim]",
        border_style="green",
    ))

    # --leaderboard: show local history and exit
    if args.leaderboard:
        show_leaderboard(args.output)
        return

    # System info
    console.print("\n[bold]Detecting system…[/bold]")
    sys_info = get_system_info()
    print_system_panel(sys_info)

    # Check Ollama
    console.print("\n[bold]Checking Ollama…[/bold]")
    if not check_ollama():
        console.print("[red]✗ Cannot reach Ollama at[/red] " + OLLAMA_BASE)
        console.print("  Start it with: [bold]ollama serve[/bold]")
        sys.exit(1)
    console.print("[green]✓[/green] Ollama is running")

    # Model selection
    local_models = list_local_models()
    if args.models:
        models_to_run = args.models
    else:
        models_to_run = interactive_model_select(local_models)

    if not models_to_run:
        console.print("[red]No models selected. Exiting.[/red]")
        sys.exit(1)

    # Pull missing models
    for model in models_to_run:
        if model not in local_models:
            if not Confirm.ask(f"\n[yellow]{model}[/yellow] is not local. Pull it now?", default=True):
                models_to_run.remove(model)
                continue
            pull_model(model)

    # Benchmarks to run
    benchmarks = [BENCHMARKS[0]] if args.quick else BENCHMARKS
    total_tasks = len(models_to_run) * len(benchmarks)

    console.print(f"\n[bold]Running [cyan]{len(benchmarks)}[/cyan] benchmark(s) "
                  f"on [cyan]{len(models_to_run)}[/cyan] model(s)…[/bold]\n")

    all_results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Starting…", total=total_tasks)
        for model in models_to_run:
            result = benchmark_model(model, benchmarks, progress, task)
            all_results.append(result)

    # Display results
    console.print()
    console.print(build_results_table(all_results))
    console.print()
    console.print(build_summary_table(all_results))

    # Always save JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = save_results(sys_info, all_results, args.output)
    console.print(f"\n[green]✓[/green] JSON saved   → [bold]{json_path}[/bold]")

    # Markdown
    if args.export_md:
        md_path = Path(args.output) / f"bench_{ts}.md"
        md_path.write_text(generate_markdown_report(sys_info, all_results))
        console.print(f"[green]✓[/green] Markdown     → [bold]{md_path}[/bold]")

    # CSV
    if args.export_csv:
        csv_path = Path(args.output) / f"bench_{ts}.csv"
        csv_path.write_text(generate_csv_report(sys_info, all_results))
        console.print(f"[green]✓[/green] CSV          → [bold]{csv_path}[/bold]")

    # HTML
    if args.export_html:
        html_path = Path(args.output) / f"bench_{ts}.html"
        html_path.write_text(generate_html_report(sys_info, all_results))
        console.print(f"[green]✓[/green] HTML report  → [bold]{html_path}[/bold]")

    # Upload to Gist leaderboard
    if args.upload:
        console.print("\n[bold]Uploading to GitHub Gist leaderboard…[/bold]")
        gist_url = upload_to_gist(sys_info, all_results, json_path)
        if gist_url:
            console.print(f"[green]✓[/green] Published   → [bold cyan]{gist_url}[/bold cyan]")

    console.print("\n[bold green]Done! 🎉[/bold green]")


if __name__ == "__main__":
    main()
