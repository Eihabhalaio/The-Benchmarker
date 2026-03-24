# The-Benchmarker

> Measure and compare LLM inference performance on **your** hardware — CPU speed, GPU tokens/sec, VRAM usage, and more.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Ollama](https://img.shields.io/badge/requires-Ollama-black)

---

## Features

- 🖥️ **Auto-detects** CPU, RAM, GPU (NVIDIA/AMD), VRAM, CUDA version
- 📊 **4 standardized benchmarks** — short, reasoning, code, long generation
- ⚡ **Key metrics**: generation tok/s, prompt eval tok/s, total time, VRAM used, GPU utilization, temperature
- 🔄 **Compare multiple models** side by side
- 💾 **JSON export** with full results + system info (timestamped)
- 📝 **Markdown report** generation for sharing
- 🎨 Beautiful **rich** terminal UI with live progress bars

---

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) installed and running

---

## Installation

```bash
git clone https://github.com/Eihabhalaio/The-Benchmarker.git
cd The-Benchmarker
pip install -r requirements.txt
```

---

## Usage

### Interactive mode (recommended for first run)

```bash
python benchmark.py
```

You'll be shown locally available models and a list of suggested ones to pull, then prompted to select.

### Benchmark specific models

```bash
python benchmark.py -m llama3.2:3b llama3.1:8b
```

### Quick benchmark (short prompt only)

```bash
python benchmark.py --quick
```

### Export a Markdown report

```bash
python benchmark.py --export-md
```

### All options

```
usage: benchmark.py [-h] [-m MODELS [MODELS ...]] [--quick] [--export-md] [--output OUTPUT] [--url URL]

options:
  -m, --models    Models to benchmark (e.g. llama3.2:3b mistral:7b)
  --quick         Run only the short benchmark
  --export-md     Also save a Markdown report to results/
  --output        Output directory for results (default: results/)
  --url           Ollama base URL (default: http://localhost:11434)
```

---

## Example Output

```
╭─────────────────────────────╮
│ 🚀 The-Benchmarker          │
│ Measure LLM inference speed │
╰─────────────────────────────╯

🖥  System Info
  OS          Linux 5.15 (x86_64)
  CPU         12th Gen Intel Core i9-12950HX
  CPU Cores   12 cores / 24 threads
  RAM         32.0 GB
  GPU         NVIDIA RTX A2000 8GB Laptop GPU
  VRAM        8.0 GB
  CUDA        12.8

📊 Benchmark Results
╭──────────────┬──────────────────┬───────────┬──────────────┬───────────┬───────────┬──────────╮
│ Model        │ Benchmark        │ Gen tok/s │ Prompt tok/s │ Total (s) │ VRAM used │ GPU util │
├──────────────┼──────────────────┼───────────┼──────────────┼───────────┼───────────┼──────────┤
│ llama3.2:3b  │ Short (haiku)    │      82.3 │        261.4 │      1.42 │  2780 MB  │      45% │
│              │ Reasoning        │      75.1 │        258.9 │      2.10 │  2780 MB  │      50% │
│              │ Code generation  │      73.8 │        255.2 │      4.31 │  2780 MB  │      52% │
│              │ Long generation  │      71.2 │        257.6 │      9.40 │  2780 MB  │      55% │
├──────────────┼──────────────────┼───────────┼──────────────┼───────────┼───────────┼──────────┤
│ llama3.1:8b  │ Short (haiku)    │      31.4 │        139.2 │      2.10 │  5348 MB  │      88% │
│              │ Reasoning        │      29.8 │        137.4 │      4.20 │  5348 MB  │      91% │
│              │ Code generation  │      28.9 │        136.8 │      9.74 │  5348 MB  │      92% │
│              │ Long generation  │      27.3 │        138.1 │     22.14 │  5348 MB  │      93% │
╰──────────────┴──────────────────┴───────────┴──────────────┴───────────┴───────────┴──────────╯

⚡ Summary — Average Generation Speed
 Model         Avg tok/s  Best tok/s  Avg total (s)  Avg VRAM (MB)
 llama3.2:3b       75.6        82.3           4.31           2780
 llama3.1:8b       29.4        31.4           9.55           5348
```

---

## Benchmark Prompts

All models are tested with the same 4 prompts for fair comparison:

| ID         | Label            | Prompt summary                                  |
|------------|------------------|-------------------------------------------------|
| `short`    | Short (haiku)    | Write a haiku about artificial intelligence     |
| `reasoning`| Reasoning        | Explain how a CPU works in 3 sentences          |
| `code`     | Code generation  | Python Fibonacci with memoization + docstring   |
| `long`     | Long generation  | 300-word story about a robot learning to paint  |

---

## Results JSON format

Each run saves a timestamped JSON to `results/`:

```json
{
  "timestamp": "2025-01-01T12:00:00",
  "system": {
    "os": "Linux 5.15 (x86_64)",
    "cpu": "Intel Core i9-12950HX",
    "cpu_cores": 12,
    "cpu_threads": 24,
    "ram_gb": 32.0,
    "gpu": "NVIDIA RTX A2000 8GB Laptop GPU",
    "gpu_vram_gb": 8.0,
    "cuda_version": "12.8"
  },
  "results": [
    {
      "model": "llama3.2:3b",
      "benchmarks": [
        {
          "benchmark_id": "short",
          "gen_tok_per_s": 82.3,
          "total_s": 1.42,
          ...
        }
      ]
    }
  ]
}
```

---

## Suggested Models to Try

| Model           | Size   | Notes                          |
|-----------------|--------|--------------------------------|
| `llama3.2:3b`   | 2.0 GB | Fast, good for quick tasks     |
| `llama3.1:8b`   | 4.9 GB | Balanced quality/speed         |
| `mistral:7b`    | 4.1 GB | Strong reasoning               |
| `codellama:7b`  | 3.8 GB | Code-focused                   |
| `phi3:mini`     | 2.2 GB | Microsoft, very fast           |
| `gemma2:9b`     | 5.5 GB | Google, high quality           |
| `llama3.1:70b`  | 40 GB  | Requires high VRAM / RAM       |

---

## Contributing

Pull requests welcome! Ideas for improvement:
- [ ] Web UI / HTML report
- [ ] Upload results to a public leaderboard
- [ ] AMD ROCm VRAM stats
- [ ] Apple Silicon (Metal) GPU detection
- [ ] CSV export

---

## License

MIT
