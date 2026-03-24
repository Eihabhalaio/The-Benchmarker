#!/usr/bin/env python3
"""
Ollama Benchmark — measure LLM inference performance on your hardware.
https://github.com/Eihabhalaio/The-Benchmarker
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
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
    result = {"gpu": None, "gpu_vram_gb": None, "cuda_version": None}
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
                # CUDA version from nvidia-smi header
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
                ["rocm-smi", "--showproductname"], text=True, stderr=subprocess.DEVNULL
            )
            for line in out.splitlines():
                if "GPU" in line and ":" in line:
                    result["gpu"] = line.split(":")[1].strip()
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
                stats["gpu_temp_c"] = int(parts[2])
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
        "# Ollama Benchmark Report",
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


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Ollama LLM inference performance on your hardware.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py                        # interactive mode
  python benchmark.py -m llama3.2:3b         # specific model
  python benchmark.py -m llama3.2:3b mistral:7b  # compare models
  python benchmark.py --quick                # short benchmark only
  python benchmark.py --export-md            # also save markdown report
        """,
    )
    parser.add_argument("-m", "--models", nargs="+", help="Models to benchmark")
    parser.add_argument("--quick",     action="store_true", help="Run only the short benchmark")
    parser.add_argument("--export-md", action="store_true", help="Export a Markdown report")
    parser.add_argument("--output",    default="results",   help="Output directory (default: results/)")
    parser.add_argument("--url",       default=OLLAMA_BASE, help=f"Ollama base URL (default: {OLLAMA_BASE})")
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
        "[bold green]🚀 Ollama Benchmark[/bold green]\n"
        "[dim]Measure LLM inference speed on your hardware[/dim]",
        border_style="green",
    ))

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

    # Save JSON
    json_path = save_results(sys_info, all_results, args.output)
    console.print(f"\n[green]✓[/green] Results saved → [bold]{json_path}[/bold]")

    # Save Markdown
    if args.export_md:
        md = generate_markdown_report(sys_info, all_results)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_path = Path(args.output) / f"bench_{ts}.md"
        md_path.write_text(md)
        console.print(f"[green]✓[/green] Markdown report → [bold]{md_path}[/bold]")

    console.print("\n[bold green]Done! 🎉[/bold green]")


if __name__ == "__main__":
    main()
