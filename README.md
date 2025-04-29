# LLM Benchmarking Framework

A comprehensive framework for benchmarking large language models (LLMs) locally using Ollama. This project allows you to evaluate and compare the performance of different LLMs across various tasks including question answering, coding, reasoning, and agent-based scenarios.

## Features

- Benchmark multiple LLMs against a variety of task types
- Generate detailed visualizations comparing model performance
- Create comprehensive reports analyzing strengths and weaknesses
- Evaluate models on specific categories (e.g., coding, reasoning)
- Run entirely locally using Ollama

## Setup

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and configured
- Required Python packages:
  - matplotlib
  - numpy
  - pandas
  - ollama (Python client)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/TBD/llm-benchmark.git
   cd llm-benchmark
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install matplotlib numpy pandas ollama
   ```

3. Make sure Ollama is installed and running:
   ```bash
   # Check Ollama installation
   ollama --version
   
   # Start Ollama service
   ollama serve
   ```

## Usage

### Basic Usage

To run a benchmark with the default configuration:

```bash
python run_benchmark.py
```

This will:
1. Load benchmark tasks from the `data/` directory
2. Initialize models specified in `config.json`
3. Run all benchmarks on all models
4. Generate visualizations and reports in the `results/` directory

### Advanced Usage

You can customize the benchmarking process with command-line arguments:

```bash
# Run specific models
python run_benchmark.py --models llama2 mistral

# Run specific benchmarks
python run_benchmark.py --benchmarks qa_benchmark reasoning_benchmark

# Generate only specific reports
python run_benchmark.py --report-type comparison

# Skip visualizations
python run_benchmark.py --skip-visualizations

# Use a different configuration file
python run_benchmark.py --config custom_config.json
```

### Configuration

The `config.json` file allows you to customize:
- Which models to benchmark
- Model parameters (max tokens, temperature, etc.)
- Output directory
- Whether to generate visualizations and reports

Example configuration:

```json
{
  "models": [
    {
      "id": "llama2",
      "model_name": "llama2",
      "type": "ollama",
      "parameters": {
        "max_tokens": 2048,
        "temperature": 0.7
      }
    },
    {
      "id": "mistral",
      "model_name": "mistral",
      "type": "ollama",
      "parameters": {
        "max_tokens": 2048,
        "temperature": 0.7
      }
    }
  ],
  "output_dir": "results",
  "generate_visualizations": true,
  "generate_reports": true
}
```

## Benchmark Types

The framework includes four main categories of benchmarks:

1. **Question Answering** (`qa_benchmark.json`):
   - Factual knowledge
   - Common sense
   - Contextual understanding
   - Multiple choice

2. **Coding** (`code_benchmark.json`):
   - Code generation
   - Debugging
   - Code explanation

3. **Reasoning** (`reasoning_benchmark.json`):
   - Logic puzzles
   - Mathematical reasoning
   - Analytical reasoning
   - Syllogistic reasoning
   - Counterfactual reasoning

4. **Summarization and Agent Tasks** (`summarization_benchmark.json`):
   - Text summarization
   - Agent decision-making
   - Multi-step tasks
   - Information extraction
   - Ethical reasoning

## Reports and Visualizations

The framework generates several types of reports and visualizations:

### Reports
- **Model Reports**: Detailed analysis of a specific model's performance
- **Benchmark Reports**: Comparison of all models on a specific benchmark
- **Comparison Reports**: Direct comparison between multiple models
- **Comprehensive Report**: Overall analysis of all models and benchmarks

### Visualizations
- Bar charts comparing model performance on each benchmark
- Task performance charts for each model
- Radar charts comparing models across benchmarks
- Heatmaps of model performance
- Task type performance charts

## Project Structure

```
llm-benchmark/
├── benchmark_framework/
│   ├── __init__.py
│   ├── benchmark.py      # Core benchmarking engine
│   ├── tasks.py          # Task loaders and definitions
│   ├── visualization.py  # Generate plots and visuals
│   └── report.py         # Auto-generate benchmark reports
├── data/
│   ├── qa_benchmark.json
│   ├── code_benchmark.json
│   ├── reasoning_benchmark.json
│   └── summarization_benchmark.json
├── results/              # Output directory (created by the code)
│   ├── plots/            # Visualization images
│   └── reports/          # Generated reports
├── config.json           # Configuration file
└── run_benchmark.py      # Main execution script
```

## Extending the Framework

### Adding New Benchmarks

Create a new JSON file in the `data/` directory with the following structure:

```json
{
  "benchmark_info": {
    "name": "Your Benchmark Name",
    "description": "Description of the benchmark",
    "version": "1.0",
    "category": "Category Name"
  },
  "tasks": [
    {
      "id": "unique_task_id",
      "type": "task_type",
      "difficulty": "easy|medium|hard",
      "prompt": "Task prompt or question",
      "reference_answer": "Reference answer",
      "evaluation_criteria": {
        "criterion1": true,
        "criterion2": false
      }
    }
  ]
}
```

### Supporting New Model Types

To add support for other types of models, extend the `ModelInterface` class in `benchmark.py` and implement the `generate()` method.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by the DeepSeek R1 paper
- Uses [Ollama](https://ollama.ai/) for local LLM inferencing