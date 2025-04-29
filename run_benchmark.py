#!/usr/bin/env python3
"""
Main script for running the LLM benchmarking framework.
"""
import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import framework modules
from benchmark_framework.benchmark import LLMBenchmark, ModelInterface
from benchmark_framework.tasks import TaskLoader
from benchmark_framework.visualization import BenchmarkVisualizer
from benchmark_framework.report import BenchmarkReporter


class OllamaModel(ModelInterface):
    """
    Implementation for Ollama models (local inference).
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the Ollama model.
        
        Args:
            model_name: Name of the Ollama model (e.g., "llama2", "mistral")
            **kwargs: Additional parameters for the Ollama API
        """
        super().__init__(model_name, **kwargs)
        self.max_tokens = kwargs.get("max_tokens", 2048)
        self.temperature = kwargs.get("temperature", 0.7)
        
        try:
            import ollama
            self.ollama = ollama
            
            # Check if model exists locally
            try:
                models = self.ollama.list()
                model_exists = any(model["name"] == model_name for model in models.get("models", []))
                if not model_exists:
                    print(f"Warning: Model '{model_name}' not found locally. Will attempt to pull it.")
                    self.ollama.pull(model_name)
            except Exception as e:
                print(f"Warning: Could not check if model exists: {e}")
        except ImportError:
            print("Error: ollama package not installed. Install with: pip install ollama")
            sys.exit(1)
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response using the Ollama model.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated response
        """
        try:
            response = self.ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "num_predict": self.max_tokens,
                    "temperature": self.temperature
                }
            )
            return response.get("response", "")
        except Exception as e:
            print(f"Error generating response with Ollama: {e}")
            return f"Error: {str(e)}"


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run LLM benchmarks on local models with Ollama")
    
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to configuration file (default: config.json)")
    
    parser.add_argument("--models", type=str, nargs="+",
                        help="Specific models to benchmark (defaults to all in config)")
    
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        help="Specific benchmarks to run (defaults to all in data directory)")
    
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to store results (default: results)")
    
    parser.add_argument("--skip-visualizations", action="store_true",
                        help="Skip generating visualizations")
    
    parser.add_argument("--skip-reports", action="store_true",
                        help="Skip generating reports")
    
    parser.add_argument("--report-type", type=str, choices=["model", "benchmark", "comparison", "comprehensive"], 
                        default="comprehensive",
                        help="Type of report to generate (default: comprehensive)")
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        print("Creating default configuration...")
        
        default_config = {
            "models": [
                {
                    "id": "llama3-8b",
                    "model_name": "llama3:8b",
                    "type": "ollama",
                    "parameters": {
                        "max_tokens": 2048,
                        "temperature": 0.7
                    }
                },
                {
                    "id": "mistral-7b",
                    "model_name": "mistral:7b",
                    "type": "ollama",
                    "parameters": {
                        "max_tokens": 2048,
                        "temperature": 0.7
                    }
                },
                {
                    "id": "phi3-3.8b",
                    "model_name": "phi3:3.8b",
                    "type": "ollama",
                    "parameters": {
                        "max_tokens": 2048,
                        "temperature": 0.7
                    }
                },
                {
                    "id": "deepseek-r1-8b",
                    "model_name": "deepseek-r1:8b",
                    "type": "ollama",
                    "parameters": {
                        "max_tokens": 2048,
                        "temperature": 0.7
                    }
                }
            ],
            "output_dir": "results",
            "generate_visualizations": True,
            "generate_reports": True
        }
        
        os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"Default configuration saved to {config_path}")
        return default_config


def setup_models(config: Dict[str, Any], model_filter: Optional[List[str]] = None) -> Dict[str, ModelInterface]:
    """
    Set up models based on configuration.
    
    Args:
        config: Configuration dictionary
        model_filter: Optional list of model IDs to include
        
    Returns:
        Dictionary mapping model IDs to model instances
    """
    models = {}
    
    for model_config in config.get("models", []):
        model_id = model_config.get("id")
        model_name = model_config.get("model_name", model_id)
        model_type = model_config.get("type", "ollama")
        parameters = model_config.get("parameters", {})
        
        # Skip if not in filter (if filter is provided)
        if model_filter and model_id not in model_filter:
            continue
        
        # Currently, we only support Ollama models
        if model_type.lower() == "ollama":
            try:
                model = OllamaModel(model_name, **parameters)
                models[model_id] = model
                print(f"Initialized Ollama model: {model_id} ({model_name})")
            except Exception as e:
                print(f"Error initializing Ollama model {model_id}: {e}")
        else:
            print(f"Unsupported model type: {model_type}")
    
    return models


def main():
    """
    Main function to run the benchmarking process.
    """
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine output directory
    output_dir = args.output_dir or config.get("output_dir", "results")
    
    # Initialize benchmark framework
    benchmark = LLMBenchmark(data_dir="data", results_dir=output_dir)
    
    # Load benchmarks
    benchmark.load_benchmarks()
    
    # Filter benchmarks if specified
    if args.benchmarks:
        all_benchmarks = set(benchmark.tasks.keys())
        requested_benchmarks = set(args.benchmarks)
        valid_benchmarks = all_benchmarks.intersection(requested_benchmarks)
        invalid_benchmarks = requested_benchmarks - all_benchmarks
        
        if invalid_benchmarks:
            print(f"Warning: The following requested benchmarks were not found: {', '.join(invalid_benchmarks)}")
        
        if not valid_benchmarks:
            print("Error: No valid benchmarks found. Available benchmarks:")
            for b in sorted(all_benchmarks):
                print(f"  - {b}")
            sys.exit(1)
        
        # Filter tasks to only include requested benchmarks
        benchmark.tasks = {k: v for k, v in benchmark.tasks.items() if k in valid_benchmarks}
    
    # Set up models
    models = setup_models(config, args.models)
    
    if not models:
        print("Error: No models available. Please check your configuration.")
        sys.exit(1)
    
    # Register models with the benchmark
    for model_id, model_instance in models.items():
        benchmark.register_model(model_id, model_instance)
    
    # Run all benchmarks
    print(f"Running {len(benchmark.tasks)} benchmarks on {len(models)} models...")
    start_time = time.time()
    
    results = benchmark.run_all_benchmarks()
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Completed all benchmarks in {total_time:.2f} seconds")
    
    # Generate visualizations if not skipped
    if not args.skip_visualizations and config.get("generate_visualizations", True):
        print("Generating visualizations...")
        visualizer = BenchmarkVisualizer(results_dir=output_dir)
        visualizer.load_results()
        
        # Create comparison charts for each benchmark
        for benchmark_name in benchmark.tasks.keys():
            visualizer.create_comparison_chart(benchmark_name)
            
            # Create task performance charts for each model on this benchmark
            for model_id in models.keys():
                visualizer.create_task_performance_chart(benchmark_name, model_id)
        
        # Create task type performance charts for each model
        for model_id in models.keys():
            visualizer.create_task_type_performance_chart(model_id)
        
        # Create model comparison visualizations
        visualizer.create_model_comparison_radar(list(models.keys()))
        visualizer.create_benchmark_heatmap()
        
        print("Visualizations generated successfully")
    
    # Generate reports if not skipped
    if not args.skip_reports and config.get("generate_reports", True):
        print("Generating reports...")
        reporter = BenchmarkReporter(results_dir=output_dir)
        reporter.load_results()
        
        if args.report_type == "model":
            # Generate model reports
            for model_id in models.keys():
                reporter.generate_model_report(model_id)
        
        elif args.report_type == "benchmark":
            # Generate benchmark reports
            for benchmark_name in benchmark.tasks.keys():
                reporter.generate_benchmark_report(benchmark_name)
        
        elif args.report_type == "comparison":
            # Generate comparison report
            reporter.generate_comparison_report(list(models.keys()))
        
        else:  # comprehensive
            # Generate comprehensive report
            reporter.generate_comprehensive_report()
        
        print("Reports generated successfully")
    
    print("Benchmarking complete! Results are available in the output directory.")


if __name__ == "__main__":
    main()