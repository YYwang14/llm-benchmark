"""
Visualization utilities for benchmark results.
"""
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


class BenchmarkVisualizer:
    """
    Generate visualizations of benchmark results.
    """
    
    def __init__(self, results_dir: str = "results", output_dir: str = "results/plots"):
        """
        Initialize the visualizer.
        
        Args:
            results_dir: Directory containing benchmark result JSON files
            output_dir: Directory to save visualizations
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.results_data = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_results(self) -> Dict[str, Any]:
        """
        Load all benchmark result files from the results directory.
        
        Returns:
            Dictionary containing loaded results
        """
        for json_file in self.results_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    result_data = json.load(f)
                
                # Use benchmark_name and model_id as key
                benchmark_name = result_data.get("benchmark_name", "unknown")
                model_id = result_data.get("model_id", "unknown")
                key = f"{benchmark_name}_{model_id}"
                
                self.results_data[key] = result_data
                print(f"Loaded result: {key}")
            except Exception as e:
                print(f"Error loading result file {json_file}: {e}")
        
        return self.results_data
    
    def create_comparison_chart(self, benchmark_name: str, save: bool = True) -> plt.Figure:
        """
        Create a bar chart comparing model performance on a specific benchmark.
        
        Args:
            benchmark_name: Name of the benchmark to visualize
            save: Whether to save the chart to a file
            
        Returns:
            Matplotlib figure
        """
        # Find results for this benchmark
        benchmark_results = {}
        for key, result in self.results_data.items():
            if result.get("benchmark_name") == benchmark_name:
                model_id = result.get("model_id", "unknown")
                benchmark_results[model_id] = result
        
        if not benchmark_results:
            print(f"No results found for benchmark '{benchmark_name}'")
            return None
        
        # Extract scores for each model
        models = []
        scores = []
        
        for model_id, result in benchmark_results.items():
            models.append(model_id)
            avg_score = result["summary"].get("average_score", 0.0)
            scores.append(avg_score)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(models, scores, color='skyblue')
        
        # Add labels and title
        ax.set_xlabel('Models')
        ax.set_ylabel('Average Score')
        ax.set_title(f'Model Performance on {benchmark_name}')
        ax.set_ylim(0, 1.0)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save figure if requested
        if save:
            output_path = self.output_dir / f"{benchmark_name}_comparison.png"
            plt.savefig(output_path)
            print(f"Saved comparison chart to {output_path}")
        
        return fig
    
    def create_task_performance_chart(self, benchmark_name: str, model_id: str, save: bool = True) -> plt.Figure:
        """
        Create a chart showing performance on individual tasks for a specific model and benchmark.
        
        Args:
            benchmark_name: Name of the benchmark
            model_id: ID of the model
            save: Whether to save the chart to a file
            
        Returns:
            Matplotlib figure
        """
        # Find the specific result
        key = f"{benchmark_name}_{model_id}"
        if key not in self.results_data:
            print(f"No results found for benchmark '{benchmark_name}' and model '{model_id}'")
            return None
        
        result = self.results_data[key]
        
        # Extract task scores
        task_ids = []
        task_scores = []
        task_types = {}  # Map task IDs to task types for grouping
        
        for task_result in result.get("tasks", []):
            task_id = task_result.get("task_id", "unknown")
            task_ids.append(task_id)
            task_scores.append(task_result.get("score", 0.0))
            task_types[task_id] = task_result.get("type", "unknown")
        
        if not task_ids:
            print(f"No task results found for benchmark '{benchmark_name}' and model '{model_id}'")
            return None
        
        # Group by task type
        unique_types = set(task_types.values())
        type_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
        type_color_map = dict(zip(unique_types, type_colors))
        
        # Create bar chart with color coding by task type
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(task_ids, task_scores)
        
        # Color bars by task type
        for i, task_id in enumerate(task_ids):
            bars[i].set_color(type_color_map[task_types[task_id]])
        
        # Add labels and title
        ax.set_xlabel('Tasks')
        ax.set_ylabel('Score')
        ax.set_title(f'Task Performance for {model_id} on {benchmark_name}')
        ax.set_ylim(0, 1.0)
        
        # Add legend for task types
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=type_color_map[t], label=t) for t in unique_types]
        ax.legend(handles=legend_elements, title="Task Types")
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save figure if requested
        if save:
            output_path = self.output_dir / f"{benchmark_name}_{model_id}_tasks.png"
            plt.savefig(output_path)
            print(f"Saved task performance chart to {output_path}")
        
        return fig
    
    def create_model_comparison_radar(self, model_ids: List[str], save: bool = True) -> plt.Figure:
        """
        Create a radar chart comparing multiple models across all benchmarks.
        
        Args:
            model_ids: List of model IDs to compare
            save: Whether to save the chart to a file
            
        Returns:
            Matplotlib figure
        """
        # Get unique benchmarks
        benchmarks = set()
        for key, result in self.results_data.items():
            benchmark_name = result.get("benchmark_name")
            if benchmark_name:
                benchmarks.add(benchmark_name)
        
        if not benchmarks:
            print("No benchmark results found")
            return None
        
        # Extract scores for each model on each benchmark
        benchmark_list = sorted(list(benchmarks))
        model_scores = {}
        
        for model_id in model_ids:
            scores = []
            for benchmark in benchmark_list:
                key = f"{benchmark}_{model_id}"
                if key in self.results_data:
                    avg_score = self.results_data[key]["summary"].get("average_score", 0.0)
                    scores.append(avg_score)
                else:
                    scores.append(0.0)
            
            model_scores[model_id] = scores
        
        # Create radar chart
        num_benchmarks = len(benchmark_list)
        angles = np.linspace(0, 2 * np.pi, num_benchmarks, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        for model_id, scores in model_scores.items():
            scores += scores[:1]  # Close the loop
            ax.plot(angles, scores, linewidth=2, label=model_id)
            ax.fill(angles, scores, alpha=0.1)
        
        # Set labels and title
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(benchmark_list)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"])
        ax.set_ylim(0, 1.0)
        
        plt.title("Model Performance Across Benchmarks", size=15, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Save figure if requested
        if save:
            output_path = self.output_dir / "model_comparison_radar.png"
            plt.savefig(output_path)
            print(f"Saved radar chart to {output_path}")
        
        return fig
    
    def create_benchmark_heatmap(self, save: bool = True) -> plt.Figure:
        """
        Create a heatmap of model performance across all benchmarks.
        
        Args:
            save: Whether to save the chart to a file
            
        Returns:
            Matplotlib figure
        """
        # Extract unique models and benchmarks
        models = set()
        benchmarks = set()
        
        for key, result in self.results_data.items():
            model_id = result.get("model_id")
            benchmark_name = result.get("benchmark_name")
            if model_id and benchmark_name:
                models.add(model_id)
                benchmarks.add(benchmark_name)
        
        if not models or not benchmarks:
            print("Not enough data for heatmap")
            return None
        
        # Create score matrix
        benchmark_list = sorted(list(benchmarks))
        model_list = sorted(list(models))
        score_matrix = np.zeros((len(model_list), len(benchmark_list)))
        
        for i, model_id in enumerate(model_list):
            for j, benchmark in enumerate(benchmark_list):
                key = f"{benchmark}_{model_id}"
                if key in self.results_data:
                    score_matrix[i, j] = self.results_data[key]["summary"].get("average_score", 0.0)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(score_matrix, cmap="YlGnBu", vmin=0, vmax=1)
        
        # Add labels and title
        ax.set_xticks(np.arange(len(benchmark_list)))
        ax.set_yticks(np.arange(len(model_list)))
        ax.set_xticklabels(benchmark_list)
        ax.set_yticklabels(model_list)
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Score", rotation=-90, va="bottom")
        
        # Add text annotations
        for i in range(len(model_list)):
            for j in range(len(benchmark_list)):
                text = ax.text(j, i, f"{score_matrix[i, j]:.2f}",
                               ha="center", va="center", color="black")
        
        ax.set_title("Model Performance Heatmap")
        fig.tight_layout()
        
        # Save figure if requested
        if save:
            output_path = self.output_dir / "benchmark_heatmap.png"
            plt.savefig(output_path)
            print(f"Saved heatmap to {output_path}")
        
        return fig
    
    def create_task_type_performance_chart(self, model_id: str, save: bool = True) -> plt.Figure:
        """
        Create a chart showing model performance across different task types.
        
        Args:
            model_id: ID of the model to analyze
            save: Whether to save the chart to a file
            
        Returns:
            Matplotlib figure
        """
        # Collect all task results for this model
        task_type_scores = {}
        
        for key, result in self.results_data.items():
            if result.get("model_id") != model_id:
                continue
            
            for task_result in result.get("tasks", []):
                task_type = task_result.get("type", "unknown")
                score = task_result.get("score", 0.0)
                
                if task_type not in task_type_scores:
                    task_type_scores[task_type] = []
                
                task_type_scores[task_type].append(score)
        
        if not task_type_scores:
            print(f"No task results found for model '{model_id}'")
            return None
        
        # Calculate average score for each task type
        task_types = []
        avg_scores = []
        
        for task_type, scores in task_type_scores.items():
            task_types.append(task_type)
            avg_scores.append(sum(scores) / len(scores))
        
        # Sort by task type name
        sorted_indices = np.argsort(task_types)
        task_types = [task_types[i] for i in sorted_indices]
        avg_scores = [avg_scores[i] for i in sorted_indices]
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(task_types, avg_scores, color='lightgreen')
        
        # Add labels and title
        ax.set_xlabel('Task Types')
        ax.set_ylabel('Average Score')
        ax.set_title(f'Performance by Task Type for {model_id}')
        ax.set_ylim(0, 1.0)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save figure if requested
        if save:
            output_path = self.output_dir / f"{model_id}_task_type_performance.png"
            plt.savefig(output_path)
            print(f"Saved task type performance chart to {output_path}")
        
        return fig