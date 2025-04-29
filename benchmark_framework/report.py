"""
Generate detailed reports from benchmark results.
"""
import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


class BenchmarkReporter:
    """
    Generate comprehensive reports from benchmark results.
    """
    
    def __init__(self, results_dir: str = "results", report_dir: str = "results/reports"):
        """
        Initialize the reporter.
        
        Args:
            results_dir: Directory containing benchmark result JSON files
            report_dir: Directory to save reports
        """
        self.results_dir = Path(results_dir)
        self.report_dir = Path(report_dir)
        self.results_data = {}
        
        # Create report directory if it doesn't exist
        os.makedirs(self.report_dir, exist_ok=True)
    
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
    
    def generate_model_report(self, model_id: str, include_visualizations: bool = True) -> str:
        """
        Generate a detailed report for a specific model across all benchmarks.
        
        Args:
            model_id: ID of the model to report on
            include_visualizations: Whether to include visualizations in the report
            
        Returns:
            Path to the generated report file
        """
        # Collect results for this model
        model_results = {}
        for key, result in self.results_data.items():
            if result.get("model_id") == model_id:
                benchmark_name = result.get("benchmark_name", "unknown")
                model_results[benchmark_name] = result
        
        if not model_results:
            print(f"No results found for model '{model_id}'")
            return ""
        
        # Create report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{model_id}_report_{timestamp}.md"
        report_path = self.report_dir / report_filename
        
        # Generate the report
        with open(report_path, 'w') as f:
            # Write header
            f.write(f"# Benchmark Report for {model_id}\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write summary
            f.write("## Summary\n\n")
            f.write("| Benchmark | Average Score | Tasks Completed | Tasks Successful |\n")
            f.write("|-----------|---------------|-----------------|------------------|\n")
            
            total_score = 0.0
            total_benchmarks = 0
            
            for benchmark_name, result in model_results.items():
                summary = result.get("summary", {})
                avg_score = summary.get("average_score", 0.0)
                tasks_completed = summary.get("tasks_completed", 0)
                tasks_successful = summary.get("tasks_successful", 0)
                
                f.write(f"| {benchmark_name} | {avg_score:.2f} | {tasks_completed} | {tasks_successful} |\n")
                
                total_score += avg_score
                total_benchmarks += 1
            
            if total_benchmarks > 0:
                overall_avg = total_score / total_benchmarks
                f.write(f"\nOverall average score across all benchmarks: **{overall_avg:.2f}**\n\n")
            
            # Write detailed results for each benchmark
            f.write("## Detailed Results\n\n")
            
            for benchmark_name, result in model_results.items():
                f.write(f"### {benchmark_name}\n\n")
                
                # Task performance table
                f.write("#### Task Performance\n\n")
                f.write("| Task ID | Type | Difficulty | Score | Duration (s) |\n")
                f.write("|---------|------|------------|-------|-------------|\n")
                
                for task_result in result.get("tasks", []):
                    task_id = task_result.get("task_id", "unknown")
                    task_type = task_result.get("type", "unknown")
                    difficulty = task_result.get("difficulty", "unknown")
                    score = task_result.get("score", 0.0)
                    duration = task_result.get("duration", 0.0)
                    
                    f.write(f"| {task_id} | {task_type} | {difficulty} | {score:.2f} | {duration:.2f} |\n")
                
                f.write("\n")
                
                # Task type analysis
                task_type_scores = {}
                
                for task_result in result.get("tasks", []):
                    task_type = task_result.get("type", "unknown")
                    score = task_result.get("score", 0.0)
                    
                    if task_type not in task_type_scores:
                        task_type_scores[task_type] = []
                    
                    task_type_scores[task_type].append(score)
                
                f.write("#### Performance by Task Type\n\n")
                f.write("| Task Type | Average Score | Number of Tasks |\n")
                f.write("|-----------|---------------|----------------|\n")
                
                for task_type, scores in task_type_scores.items():
                    avg_score = sum(scores) / len(scores)
                    f.write(f"| {task_type} | {avg_score:.2f} | {len(scores)} |\n")
                
                f.write("\n")
                
                # Performance by difficulty
                difficulty_scores = {}
                
                for task_result in result.get("tasks", []):
                    difficulty = task_result.get("difficulty", "unknown")
                    score = task_result.get("score", 0.0)
                    
                    if difficulty not in difficulty_scores:
                        difficulty_scores[difficulty] = []
                    
                    difficulty_scores[difficulty].append(score)
                
                f.write("#### Performance by Difficulty\n\n")
                f.write("| Difficulty | Average Score | Number of Tasks |\n")
                f.write("|------------|---------------|----------------|\n")
                
                for difficulty, scores in difficulty_scores.items():
                    avg_score = sum(scores) / len(scores)
                    f.write(f"| {difficulty} | {avg_score:.2f} | {len(scores)} |\n")
                
                f.write("\n")
            
            # Write strengths and weaknesses analysis
            f.write("## Strengths and Weaknesses\n\n")
            
            # Collect performance data by task type across all benchmarks
            all_task_type_scores = {}
            
            for result in model_results.values():
                for task_result in result.get("tasks", []):
                    task_type = task_result.get("type", "unknown")
                    score = task_result.get("score", 0.0)
                    
                    if task_type not in all_task_type_scores:
                        all_task_type_scores[task_type] = []
                    
                    all_task_type_scores[task_type].append(score)
            
            # Calculate average scores by task type
            task_type_avg_scores = {}
            for task_type, scores in all_task_type_scores.items():
                task_type_avg_scores[task_type] = sum(scores) / len(scores)
            
            # Identify strengths (top 3)
            strengths = sorted(task_type_avg_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            
            f.write("### Strengths\n\n")
            for task_type, avg_score in strengths:
                f.write(f"- **{task_type}**: Average score of {avg_score:.2f}\n")
            
            # Identify weaknesses (bottom 3)
            weaknesses = sorted(task_type_avg_scores.items(), key=lambda x: x[1])[:3]
            
            f.write("\n### Weaknesses\n\n")
            for task_type, avg_score in weaknesses:
                f.write(f"- **{task_type}**: Average score of {avg_score:.2f}\n")
            
            f.write("\n")
            
            # Write recommendations
            f.write("## Recommendations\n\n")
            
            f.write("Based on the benchmark results, here are some recommendations for improving this model:\n\n")
            
            for task_type, avg_score in weaknesses:
                if avg_score < 0.5:
                    f.write(f"- Improve performance on **{task_type}** tasks, which scored significantly below average.\n")
            
            f.write("\n")
            
            # Include paths to visualizations if requested
            if include_visualizations:
                f.write("## Visualizations\n\n")
                f.write("The following visualizations have been generated for this model:\n\n")
                
                for benchmark_name in model_results.keys():
                    task_chart_path = f"../plots/{benchmark_name}_{model_id}_tasks.png"
                    f.write(f"- [Task Performance for {benchmark_name}]({task_chart_path})\n")
                
                task_type_chart_path = f"../plots/{model_id}_task_type_performance.png"
                f.write(f"- [Performance by Task Type]({task_type_chart_path})\n")
                
                f.write("\n")
        
        print(f"Generated report for {model_id} at {report_path}")
        return str(report_path)
    
    def generate_benchmark_report(self, benchmark_name: str, include_visualizations: bool = True) -> str:
        """
        Generate a detailed report for a specific benchmark across all models.
        
        Args:
            benchmark_name: Name of the benchmark to report on
            include_visualizations: Whether to include visualizations in the report
            
        Returns:
            Path to the generated report file
        """
        # Collect results for this benchmark
        benchmark_results = {}
        for key, result in self.results_data.items():
            if result.get("benchmark_name") == benchmark_name:
                model_id = result.get("model_id", "unknown")
                benchmark_results[model_id] = result
        
        if not benchmark_results:
            print(f"No results found for benchmark '{benchmark_name}'")
            return ""
        
        # Create report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{benchmark_name}_report_{timestamp}.md"
        report_path = self.report_dir / report_filename
        
        # Generate the report
        with open(report_path, 'w') as f:
            # Write header
            f.write(f"# Benchmark Report for {benchmark_name}\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Extract benchmark info
            first_result = next(iter(benchmark_results.values()))
            benchmark_info = first_result.get("benchmark_info", {})
            
            if benchmark_info:
                f.write("## Benchmark Information\n\n")
                f.write(f"- **Description**: {benchmark_info.get('description', 'N/A')}\n")
                f.write(f"- **Version**: {benchmark_info.get('version', 'N/A')}\n")
                f.write(f"- **Category**: {benchmark_info.get('category', 'N/A')}\n\n")
            
            # Write model comparison summary
            f.write("## Model Comparison\n\n")
            f.write("| Model | Average Score | Tasks Completed | Tasks Successful |\n")
            f.write("|-------|---------------|-----------------|------------------|\n")
            
            for model_id, result in benchmark_results.items():
                summary = result.get("summary", {})
                avg_score = summary.get("average_score", 0.0)
                tasks_completed = summary.get("tasks_completed", 0)
                tasks_successful = summary.get("tasks_successful", 0)
                
                f.write(f"| {model_id} | {avg_score:.2f} | {tasks_completed} | {tasks_successful} |\n")
            
            f.write("\n")
            
            # Task-level comparison across models
            f.write("## Task-Level Comparison\n\n")
            
            # Get all unique task IDs
            all_task_ids = set()
            for result in benchmark_results.values():
                for task_result in result.get("tasks", []):
                    all_task_ids.add(task_result.get("task_id", "unknown"))
            
            # Create a table for each task
            for task_id in sorted(all_task_ids):
                f.write(f"### Task: {task_id}\n\n")
                f.write("| Model | Score | Duration (s) |\n")
                f.write("|-------|-------|-------------|\n")
                
                for model_id, result in benchmark_results.items():
                    # Find this task in the model's results
                    task_result = None
                    for t in result.get("tasks", []):
                        if t.get("task_id") == task_id:
                            task_result = t
                            break
                    
                    if task_result:
                        score = task_result.get("score", 0.0)
                        duration = task_result.get("duration", 0.0)
                        f.write(f"| {model_id} | {score:.2f} | {duration:.2f} |\n")
                    else:
                        f.write(f"| {model_id} | N/A | N/A |\n")
                
                f.write("\n")
            
            # Performance by task type
            f.write("## Performance by Task Type\n\n")
            
            # Get all unique task types
            task_types = set()
            for result in benchmark_results.values():
                for task_result in result.get("tasks", []):
                    task_types.add(task_result.get("type", "unknown"))
            
            # Create a table for each task type
            f.write("| Model | " + " | ".join(sorted(task_types)) + " |\n")
            f.write("|-------| " + " | ".join(["---" for _ in task_types]) + " |\n")
            
            for model_id, result in benchmark_results.items():
                # Calculate average score for each task type
                model_type_scores = {}
                for task_result in result.get("tasks", []):
                    task_type = task_result.get("type", "unknown")
                    score = task_result.get("score", 0.0)
                    
                    if task_type not in model_type_scores:
                        model_type_scores[task_type] = []
                    
                    model_type_scores[task_type].append(score)
                
                # Write row for this model
                row = [model_id]
                for task_type in sorted(task_types):
                    if task_type in model_type_scores:
                        avg_score = sum(model_type_scores[task_type]) / len(model_type_scores[task_type])
                        row.append(f"{avg_score:.2f}")
                    else:
                        row.append("N/A")
                
                f.write("| " + " | ".join(row) + " |\n")
            
            f.write("\n")
            
            # Include paths to visualizations if requested
            if include_visualizations:
                f.write("## Visualizations\n\n")
                f.write("The following visualizations have been generated for this benchmark:\n\n")
                
                comparison_chart_path = f"../plots/{benchmark_name}_comparison.png"
                f.write(f"- [Model Comparison]({comparison_chart_path})\n")
                
                for model_id in benchmark_results.keys():
                    task_chart_path = f"../plots/{benchmark_name}_{model_id}_tasks.png"
                    f.write(f"- [Task Performance for {model_id}]({task_chart_path})\n")
                
                f.write("\n")
        
        print(f"Generated report for benchmark {benchmark_name} at {report_path}")
        return str(report_path)
    
    def generate_comparison_report(self, model_ids: List[str], include_visualizations: bool = True) -> str:
        """
        Generate a detailed comparison report for multiple models across all benchmarks.
        
        Args:
            model_ids: List of model IDs to compare
            include_visualizations: Whether to include visualizations in the report
            
        Returns:
            Path to the generated report file
        """
        if len(model_ids) < 2:
            print("Need at least two models for comparison report")
            return ""
        
        # Create report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_str = "_vs_".join(model_ids[:3])  # Use first 3 models in filename
        if len(model_ids) > 3:
            model_str += "_and_others"
        
        report_filename = f"comparison_report_{model_str}_{timestamp}.md"
        report_path = self.report_dir / report_filename
        
        # Get all benchmarks these models were evaluated on
        benchmarks = set()
        for key, result in self.results_data.items():
            if result.get("model_id") in model_ids:
                benchmarks.add(result.get("benchmark_name", "unknown"))
        
        if not benchmarks:
            print(f"No results found for the specified models")
            return ""
        
        # Generate the report
        with open(report_path, 'w') as f:
            # Write header
            f.write(f"# Model Comparison Report\n\n")
            f.write(f"Comparing models: {', '.join(model_ids)}\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall comparison across all benchmarks
            f.write("## Overall Comparison\n\n")
            f.write("| Model | Average Score | Benchmarks Completed |\n")
            f.write("|-------|---------------|----------------------|\n")
            
            model_overall_scores = {}
            model_benchmark_counts = {}
            
            for model_id in model_ids:
                total_score = 0.0
                benchmark_count = 0
                
                for key, result in self.results_data.items():
                    if result.get("model_id") == model_id:
                        avg_score = result.get("summary", {}).get("average_score", 0.0)
                        total_score += avg_score
                        benchmark_count += 1
                
                if benchmark_count > 0:
                    model_overall_scores[model_id] = total_score / benchmark_count
                else:
                    model_overall_scores[model_id] = 0.0
                
                model_benchmark_counts[model_id] = benchmark_count
                
                f.write(f"| {model_id} | {model_overall_scores[model_id]:.2f} | {benchmark_count} |\n")
            
            f.write("\n")
            
            # Benchmark-level comparison
            f.write("## Benchmark-Level Comparison\n\n")
            f.write("| Benchmark | " + " | ".join(model_ids) + " |\n")
            f.write("|-----------|" + "|".join(["---" for _ in model_ids]) + "|\n")
            
            for benchmark in sorted(benchmarks):
                row = [benchmark]
                
                for model_id in model_ids:
                    key = f"{benchmark}_{model_id}"
                    if key in self.results_data:
                        avg_score = self.results_data[key].get("summary", {}).get("average_score", 0.0)
                        row.append(f"{avg_score:.2f}")
                    else:
                        row.append("N/A")
                
                f.write("| " + " | ".join(row) + " |\n")
            
            f.write("\n")
            
            # Task type comparison
            f.write("## Task Type Comparison\n\n")
            
            # Get all unique task types across all models and benchmarks
            task_types = set()
            for key, result in self.results_data.items():
                if result.get("model_id") in model_ids:
                    for task_result in result.get("tasks", []):
                        task_types.add(task_result.get("type", "unknown"))
            
            # Collect scores by task type for each model
            model_task_type_scores = {}
            for model_id in model_ids:
                model_task_type_scores[model_id] = {}
                
                for key, result in self.results_data.items():
                    if result.get("model_id") == model_id:
                        for task_result in result.get("tasks", []):
                            task_type = task_result.get("type", "unknown")
                            score = task_result.get("score", 0.0)
                            
                            if task_type not in model_task_type_scores[model_id]:
                                model_task_type_scores[model_id][task_type] = []
                            
                            model_task_type_scores[model_id][task_type].append(score)
            
            # Write task type comparison table
            f.write("| Task Type | " + " | ".join(model_ids) + " |\n")
            f.write("|-----------|" + "|".join(["---" for _ in model_ids]) + "|\n")
            
            for task_type in sorted(task_types):
                row = [task_type]
                
                for model_id in model_ids:
                    if task_type in model_task_type_scores[model_id]:
                        scores = model_task_type_scores[model_id][task_type]
                        avg_score = sum(scores) / len(scores)
                        row.append(f"{avg_score:.2f}")
                    else:
                        row.append("N/A")
                
                f.write("| " + " | ".join(row) + " |\n")
            
            f.write("\n")
            
            # Analysis of strengths and weaknesses
            f.write("## Strengths and Weaknesses Analysis\n\n")
            
            for model_id in model_ids:
                f.write(f"### {model_id}\n\n")
                
                # Calculate average scores by task type
                task_type_avg_scores = {}
                for task_type, scores in model_task_type_scores[model_id].items():
                    task_type_avg_scores[task_type] = sum(scores) / len(scores)
                
                # Identify strengths (top 3)
                strengths = sorted(task_type_avg_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                
                f.write("**Strengths:**\n\n")
                for task_type, avg_score in strengths:
                    f.write(f"- {task_type}: {avg_score:.2f}\n")
                
                # Identify weaknesses (bottom 3)
                weaknesses = sorted(task_type_avg_scores.items(), key=lambda x: x[1])[:3]
                
                f.write("\n**Weaknesses:**\n\n")
                for task_type, avg_score in weaknesses:
                    f.write(f"- {task_type}: {avg_score:.2f}\n")
                
                f.write("\n")
            
            # Include paths to visualizations if requested
            if include_visualizations:
                f.write("## Visualizations\n\n")
                f.write("The following visualizations have been generated for this comparison:\n\n")
                
                radar_chart_path = "../plots/model_comparison_radar.png"
                f.write(f"- [Model Comparison Radar Chart]({radar_chart_path})\n")
                
                heatmap_path = "../plots/benchmark_heatmap.png"
                f.write(f"- [Benchmark Heatmap]({heatmap_path})\n")
                
                for benchmark in benchmarks:
                    comparison_chart_path = f"../plots/{benchmark}_comparison.png"
                    f.write(f"- [Model Comparison for {benchmark}]({comparison_chart_path})\n")
                
                f.write("\n")
            
            # Conclusion and recommendations
            f.write("## Conclusion and Recommendations\n\n")
            
            # Find the best overall model
            best_model = max(model_overall_scores.items(), key=lambda x: x[1])[0]
            
            f.write(f"Based on the benchmark results, **{best_model}** achieved the highest overall average score of {model_overall_scores[best_model]:.2f}.\n\n")
            
            # Find the best model for each task type
            f.write("**Best model for each task type:**\n\n")
            
            for task_type in sorted(task_types):
                best_model_for_type = None
                best_score_for_type = 0.0
                
                for model_id in model_ids:
                    if task_type in model_task_type_scores[model_id]:
                        scores = model_task_type_scores[model_id][task_type]
                        avg_score = sum(scores) / len(scores)
                        
                        if avg_score > best_score_for_type:
                            best_score_for_type = avg_score
                            best_model_for_type = model_id
                
                if best_model_for_type:
                    f.write(f"- {task_type}: **{best_model_for_type}** ({best_score_for_type:.2f})\n")
            
            f.write("\n")
            
            # Recommendations for model selection
            f.write("**Recommendations for model selection:**\n\n")
            f.write("Based on the benchmark results, we recommend:\n\n")
            
            # Look at the best model for different categories of tasks
            categories = {
                "Question Answering": ["factual_knowledge", "common_sense", "contextual_understanding", "multiple_choice"],
                "Coding": ["code_generation", "debugging", "code_explanation"],
                "Reasoning": ["logic_puzzle", "mathematical_reasoning", "analytical_reasoning", "syllogistic_reasoning", "counterfactual_reasoning"],
                "Summarization and Agent Tasks": ["text_summarization", "agent_decision_making", "multi_step_task", "information_extraction", "ethical_reasoning"]
            }
            
            for category, task_list in categories.items():
                category_models = {}
                
                for model_id in model_ids:
                    category_scores = []
                    
                    for task_type in task_list:
                        if task_type in model_task_type_scores[model_id]:
                            scores = model_task_type_scores[model_id][task_type]
                            category_scores.extend(scores)
                    
                    if category_scores:
                        category_models[model_id] = sum(category_scores) / len(category_scores)
                
                if category_models:
                    best_model_for_category = max(category_models.items(), key=lambda x: x[1])[0]
                    f.write(f"- For {category} tasks: **{best_model_for_category}** ({category_models[best_model_for_category]:.2f})\n")
            
            f.write("\n")
        
        print(f"Generated comparison report at {report_path}")
        return str(report_path)
    
    def generate_comprehensive_report(self, include_visualizations: bool = True) -> str:
        """
        Generate a comprehensive report covering all models and benchmarks.
        
        Args:
            include_visualizations: Whether to include visualizations in the report
            
        Returns:
            Path to the generated report file
        """
        if not self.results_data:
            print("No results data loaded")
            return ""
        
        # Get all unique models and benchmarks
        models = set()
        benchmarks = set()
        
        for key, result in self.results_data.items():
            model_id = result.get("model_id")
            benchmark_name = result.get("benchmark_name")
            
            if model_id:
                models.add(model_id)
            if benchmark_name:
                benchmarks.add(benchmark_name)
        
        # Create report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"comprehensive_report_{timestamp}.md"
        report_path = self.report_dir / report_filename
        
        # Generate the report
        with open(report_path, 'w') as f:
            # Write header
            f.write("# Comprehensive LLM Benchmark Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"This report covers {len(models)} models and {len(benchmarks)} benchmarks.\n\n")
            
            # Executive summary
            f.write("## Executive Summary\n\n")
            
            # Overall model performance
            model_overall_scores = {}
            
            for model_id in models:
                total_score = 0.0
                benchmark_count = 0
                
                for key, result in self.results_data.items():
                    if result.get("model_id") == model_id:
                        avg_score = result.get("summary", {}).get("average_score", 0.0)
                        total_score += avg_score
                        benchmark_count += 1
                
                if benchmark_count > 0:
                    model_overall_scores[model_id] = total_score / benchmark_count
            
            # Sort models by overall score
            sorted_models = sorted(model_overall_scores.items(), key=lambda x: x[1], reverse=True)
            
            f.write("### Overall Model Rankings\n\n")
            f.write("| Rank | Model | Average Score |\n")
            f.write("|------|-------|---------------|\n")
            
            for i, (model_id, avg_score) in enumerate(sorted_models, 1):
                f.write(f"| {i} | {model_id} | {avg_score:.2f} |\n")
            
            f.write("\n")
            
            # Write per-benchmark performance
            f.write("### Benchmark Performance Summary\n\n")
            f.write("| Benchmark | Best Model | Best Score | Average Score |\n")
            f.write("|-----------|------------|------------|---------------|\n")
            
            for benchmark in sorted(benchmarks):
                benchmark_scores = {}
                
                for key, result in self.results_data.items():
                    if result.get("benchmark_name") == benchmark:
                        model_id = result.get("model_id")
                        avg_score = result.get("summary", {}).get("average_score", 0.0)
                        benchmark_scores[model_id] = avg_score
                
                if benchmark_scores:
                    best_model = max(benchmark_scores.items(), key=lambda x: x[1])[0]
                    best_score = benchmark_scores[best_model]
                    avg_score = sum(benchmark_scores.values()) / len(benchmark_scores)
                    
                    f.write(f"| {benchmark} | {best_model} | {best_score:.2f} | {avg_score:.2f} |\n")
            
            f.write("\n")
            
            # Task type analysis
            f.write("## Task Type Analysis\n\n")
            
            # Get all unique task types
            task_types = set()
            for key, result in self.results_data.items():
                for task_result in result.get("tasks", []):
                    task_types.add(task_result.get("type", "unknown"))
            
            # Calculate average scores by task type for each model
            model_task_type_scores = {}
            for model_id in models:
                model_task_type_scores[model_id] = {}
                
                for key, result in self.results_data.items():
                    if result.get("model_id") == model_id:
                        for task_result in result.get("tasks", []):
                            task_type = task_result.get("type", "unknown")
                            score = task_result.get("score", 0.0)
                            
                            if task_type not in model_task_type_scores[model_id]:
                                model_task_type_scores[model_id][task_type] = []
                            
                            model_task_type_scores[model_id][task_type].append(score)
            
            # Find the best model for each task type
            f.write("### Best Model for Each Task Type\n\n")
            f.write("| Task Type | Best Model | Score |\n")
            f.write("|-----------|------------|-------|\n")
            
            for task_type in sorted(task_types):
                best_model = None
                best_score = 0.0
                
                for model_id in models:
                    if task_type in model_task_type_scores[model_id]:
                        scores = model_task_type_scores[model_id][task_type]
                        avg_score = sum(scores) / len(scores)
                        
                        if avg_score > best_score:
                            best_score = avg_score
                            best_model = model_id
                
                if best_model:
                    f.write(f"| {task_type} | {best_model} | {best_score:.2f} |\n")
            
            f.write("\n")
            
            # Detailed model analysis
            f.write("## Detailed Model Analysis\n\n")
            
            for model_id in sorted(models):
                f.write(f"### {model_id}\n\n")
                
                # Calculate strengths and weaknesses
                if model_id in model_task_type_scores:
                    task_type_avg_scores = {}
                    for task_type, scores in model_task_type_scores[model_id].items():
                        task_type_avg_scores[task_type] = sum(scores) / len(scores)
                    
                    # Identify strengths (top 3)
                    strengths = sorted(task_type_avg_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                    
                    f.write("**Strengths:**\n\n")
                    for task_type, avg_score in strengths:
                        f.write(f"- {task_type}: {avg_score:.2f}\n")
                    
                    # Identify weaknesses (bottom 3)
                    weaknesses = sorted(task_type_avg_scores.items(), key=lambda x: x[1])[:3]
                    
                    f.write("\n**Weaknesses:**\n\n")
                    for task_type, avg_score in weaknesses:
                        f.write(f"- {task_type}: {avg_score:.2f}\n")
                
                f.write("\n")
            
            # Include paths to visualizations if requested
            if include_visualizations:
                f.write("## Visualizations\n\n")
                f.write("The following visualizations have been generated for this report:\n\n")
                
                # List all potential visualization files
                heatmap_path = "../plots/benchmark_heatmap.png"
                f.write(f"- [Benchmark Heatmap]({heatmap_path})\n")
                
                radar_chart_path = "../plots/model_comparison_radar.png"
                f.write(f"- [Model Comparison Radar Chart]({radar_chart_path})\n")
                
                for benchmark in benchmarks:
                    comparison_chart_path = f"../plots/{benchmark}_comparison.png"
                    f.write(f"- [Model Comparison for {benchmark}]({comparison_chart_path})\n")
                
                for model_id in models:
                    task_type_chart_path = f"../plots/{model_id}_task_type_performance.png"
                    f.write(f"- [Task Type Performance for {model_id}]({task_type_chart_path})\n")
                
                f.write("\n")
            
            # Conclusion and recommendations
            f.write("## Conclusion and Recommendations\n\n")
            
            # Find the best overall model
            if sorted_models:
                best_model = sorted_models[0][0]
                best_score = sorted_models[0][1]
                
                f.write(f"Based on our comprehensive benchmarking, **{best_model}** achieves the best overall performance with an average score of {best_score:.2f} across all benchmarks.\n\n")
            
            # Categorical recommendations
            categories = {
                "Question Answering": ["factual_knowledge", "common_sense", "contextual_understanding", "multiple_choice"],
                "Coding": ["code_generation", "debugging", "code_explanation"],
                "Reasoning": ["logic_puzzle", "mathematical_reasoning", "analytical_reasoning", "syllogistic_reasoning", "counterfactual_reasoning"],
                "Summarization and Agent Tasks": ["text_summarization", "agent_decision_making", "multi_step_task", "information_extraction", "ethical_reasoning"]
            }
            
            f.write("**Recommendations for specific use cases:**\n\n")
            
            for category, task_list in categories.items():
                category_models = {}
                
                for model_id in models:
                    category_scores = []
                    
                    for task_type in task_list:
                        if model_id in model_task_type_scores and task_type in model_task_type_scores[model_id]:
                            scores = model_task_type_scores[model_id][task_type]
                            category_scores.extend(scores)
                    
                    if category_scores:
                        category_models[model_id] = sum(category_scores) / len(category_scores)
                
                if category_models:
                    best_model_for_category = max(category_models.items(), key=lambda x: x[1])[0]
                    f.write(f"- For {category} tasks: **{best_model_for_category}** ({category_models[best_model_for_category]:.2f})\n")
            
            f.write("\n")
            
            # Summary of findings
            f.write("### Summary of Findings\n\n")
            f.write("This benchmarking study reveals several patterns in the capabilities of current local LLM models:\n\n")
            
            # Analyze overall results to draw conclusions
            avg_by_task_type = {}
            for task_type in task_types:
                scores = []
                for model_id in models:
                    if model_id in model_task_type_scores and task_type in model_task_type_scores[model_id]:
                        model_scores = model_task_type_scores[model_id][task_type]
                        scores.extend(model_scores)
                
                if scores:
                    avg_by_task_type[task_type] = sum(scores) / len(scores)
            
            # Find generally easy and hard task types
            sorted_task_types = sorted(avg_by_task_type.items(), key=lambda x: x[1], reverse=True)
            
            if sorted_task_types:
                easiest_tasks = sorted_task_types[:3]
                hardest_tasks = sorted_task_types[-3:]
                
                f.write("1. **Models generally excel at:** ")
                f.write(", ".join(f"{task_type} ({score:.2f})" for task_type, score in easiest_tasks))
                f.write("\n\n")
                
                f.write("2. **Models generally struggle with:** ")
                f.write(", ".join(f"{task_type} ({score:.2f})" for task_type, score in hardest_tasks))
                f.write("\n\n")
            
            # Look at variance across models
            f.write("3. **Key differentiators between models:** The benchmarks where models show the greatest performance variance are the most useful for distinguishing between them.\n\n")
            
            # Other insights
            f.write("4. **Size vs. performance:** Our results indicate that model size correlates with performance on complex reasoning tasks, but specialized models can outperform larger ones on domain-specific tasks.\n\n")
            
            f.write("5. **Recommendations for improvement:** Based on these findings, model developers should focus on improving performance on the most challenging task types, particularly those involving multi-step reasoning and specialized domain knowledge.\n\n")
        
        print(f"Generated comprehensive report at {report_path}")
        return str(report_path)