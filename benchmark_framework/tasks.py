"""
Task loaders and evaluation methods for different benchmark types.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable


class TaskLoader:
    """
    Loads tasks from benchmark files and provides methods for task evaluation.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the task loader.
        
        Args:
            data_dir: Directory containing benchmark JSON files
        """
        self.data_dir = Path(data_dir)
        self.benchmarks = {}
        self.evaluators = self._register_evaluators()
    
    def _register_evaluators(self) -> Dict[str, Callable]:
        """
        Register evaluation functions for different task types.
        
        Returns:
            Dictionary mapping task types to evaluation functions
        """
        return {
            # Question Answering tasks
            "factual_knowledge": self.evaluate_qa,
            "common_sense": self.evaluate_qa,
            "contextual_understanding": self.evaluate_qa,
            "multiple_choice": self.evaluate_multiple_choice,
            
            # Coding tasks
            "code_generation": self.evaluate_code_generation,
            "debugging": self.evaluate_debugging,
            "code_explanation": self.evaluate_code_explanation,
            
            # Reasoning tasks
            "logic_puzzle": self.evaluate_reasoning,
            "mathematical_reasoning": self.evaluate_reasoning,
            "analytical_reasoning": self.evaluate_reasoning,
            "syllogistic_reasoning": self.evaluate_reasoning,
            "counterfactual_reasoning": self.evaluate_reasoning,
            
            # Summarization and Agent tasks
            "text_summarization": self.evaluate_summarization,
            "agent_decision_making": self.evaluate_agent_decision,
            "multi_step_task": self.evaluate_multi_step_task,
            "information_extraction": self.evaluate_information_extraction,
            "ethical_reasoning": self.evaluate_ethical_reasoning
        }
    
    def load_all_benchmarks(self) -> Dict[str, Any]:
        """
        Load all benchmark files from the data directory.
        
        Returns:
            Dictionary mapping benchmark names to benchmark data
        """
        for json_file in self.data_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    benchmark_data = json.load(f)
                
                benchmark_name = json_file.stem
                self.benchmarks[benchmark_name] = benchmark_data
                print(f"Loaded benchmark: {benchmark_name} with {len(benchmark_data['tasks'])} tasks")
            except Exception as e:
                print(f"Error loading benchmark file {json_file}: {e}")
        
        return self.benchmarks
    
    def get_task_by_id(self, benchmark_name: str, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific task by ID from a benchmark.
        
        Args:
            benchmark_name: Name of the benchmark
            task_id: ID of the task
            
        Returns:
            Task data or None if not found
        """
        if benchmark_name not in self.benchmarks:
            print(f"Benchmark '{benchmark_name}' not found")
            return None
        
        for task in self.benchmarks[benchmark_name]["tasks"]:
            if task["id"] == task_id:
                return task
        
        print(f"Task '{task_id}' not found in benchmark '{benchmark_name}'")
        return None
    
    def evaluate_task(self, task: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate a model's response to a task.
        
        Args:
            task: Task data
            response: Model's response
            
        Returns:
            Evaluation results
        """
        task_type = task["type"]
        
        if task_type in self.evaluators:
            return self.evaluators[task_type](task, response)
        else:
            print(f"No evaluator registered for task type '{task_type}'")
            return {
                "score": 0.0,
                "max_score": 1.0,
                "evaluation": f"No evaluator available for task type '{task_type}'"
            }
    
    def evaluate_qa(self, task: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate a question answering task.
        
        Args:
            task: Task data
            response: Model's response
            
        Returns:
            Evaluation results
        """
        reference = task["reference_answer"]
        criteria = task.get("evaluation_criteria", {})
        
        # Clean up response and reference
        response_clean = response.strip().lower()
        reference_clean = reference.strip().lower()
        
        # Check for exact match if required
        if criteria.get("exact_match", False):
            if not criteria.get("case_sensitive", False):
                exact_match = response_clean == reference_clean
            else:
                exact_match = response.strip() == reference.strip()
            
            if exact_match:
                score = 1.0
                evaluation = "Exact match to reference answer."
            else:
                # Calculate similarity to reference
                similarity = self._calculate_text_similarity(response_clean, reference_clean)
                score = similarity
                evaluation = f"Response similarity to reference: {similarity:.2f}"
        else:
            # For non-exact matches, check for keywords or similarity
            if "keywords" in criteria:
                keywords = [k.lower() for k in criteria["keywords"]]
                matched_keywords = [k for k in keywords if k in response_clean]
                keyword_score = len(matched_keywords) / len(keywords)
                
                score = keyword_score
                evaluation = f"Matched {len(matched_keywords)}/{len(keywords)} keywords."
            else:
                # Calculate similarity to reference
                similarity = self._calculate_text_similarity(response_clean, reference_clean)
                score = similarity
                evaluation = f"Response similarity to reference: {similarity:.2f}"
        
        return {
            "score": score,
            "max_score": 1.0,
            "evaluation": evaluation
        }
    
    def evaluate_multiple_choice(self, task: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate a multiple-choice task.
        
        Args:
            task: Task data
            response: Model's response
            
        Returns:
            Evaluation results
        """
        reference = task["reference_answer"]
        options = task.get("options", [])
        
        # Clean response
        response_clean = response.strip().lower()
        reference_clean = reference.strip().lower()
        
        # Check if response contains the correct option
        correct = False
        
        # Check for exact option match
        if response_clean == reference_clean:
            correct = True
        
        # Check if response includes the option number
        elif str(task.get("correct_option", "")) in response_clean:
            correct = True
        
        # Check if the response mentions the option text
        elif any(opt.lower() in response_clean for opt in options if opt.lower() == reference_clean):
            correct = True
        
        if correct:
            score = 1.0
            evaluation = "Correct option selected."
        else:
            score = 0.0
            evaluation = "Incorrect option selected."
        
        return {
            "score": score,
            "max_score": 1.0,
            "evaluation": evaluation
        }
    
    def evaluate_code_generation(self, task: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate a code generation task.
        
        Args:
            task: Task data
            response: Model's response
            
        Returns:
            Evaluation results
        """
        # Extract code from response
        code = self._extract_code(response, task.get("language", "python"))
        
        # Get reference solution
        reference_solution = task.get("reference_solution", "")
        
        # Calculate code similarity to reference
        similarity = self._calculate_code_similarity(code, reference_solution)
        
        # Simple scoring based on similarity
        score = similarity
        evaluation = f"Code similarity to reference: {similarity:.2f}"
        
        return {
            "score": score,
            "max_score": 1.0,
            "evaluation": evaluation,
            "extracted_code": code
        }
    
    def evaluate_debugging(self, task: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate a debugging task.
        
        Args:
            task: Task data
            response: Model's response
            
        Returns:
            Evaluation results
        """
        # Extract code from response
        fixed_code = self._extract_code(response, task.get("language", "python"))
        
        # Get reference solution and buggy code
        reference_solution = task.get("reference_solution", "")
        buggy_code = task.get("buggy_code", "")
        
        # Check if the code was changed
        if fixed_code != buggy_code:
            # Compare with reference solution
            solution_similarity = self._calculate_code_similarity(fixed_code, reference_solution)
            score = solution_similarity
            evaluation = f"Fixed code similarity to reference: {solution_similarity:.2f}"
        else:
            score = 0.0
            evaluation = "No changes made to the buggy code."
        
        return {
            "score": score,
            "max_score": 1.0,
            "evaluation": evaluation,
            "extracted_code": fixed_code
        }
    
    def evaluate_code_explanation(self, task: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate a code explanation task.
        
        Args:
            task: Task data
            response: Model's response
            
        Returns:
            Evaluation results
        """
        reference_explanation = task.get("reference_explanation", "")
        
        # Compare with reference explanation
        explanation_similarity = self._calculate_text_similarity(response, reference_explanation)
        
        score = explanation_similarity
        evaluation = f"Explanation similarity to reference: {explanation_similarity:.2f}"
        
        return {
            "score": score,
            "max_score": 1.0,
            "evaluation": evaluation
        }
    
    def evaluate_reasoning(self, task: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate a reasoning task.
        
        Args:
            task: Task data
            response: Model's response
            
        Returns:
            Evaluation results
        """
        reference_answer = task.get("reference_answer", "")
        
        response_lower = response.lower()
        reference_lower = reference_answer.lower()
        
        # Check if the response contains the correct answer
        answer_correct = reference_lower in response_lower or self._calculate_text_similarity(response_lower, reference_lower) > 0.8
        
        if answer_correct:
            score = 1.0
            evaluation = "Correct answer provided."
        else:
            score = 0.0
            evaluation = "Incorrect answer provided."
        
        return {
            "score": score,
            "max_score": 1.0,
            "evaluation": evaluation
        }
    
    def evaluate_summarization(self, task: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate a text summarization task.
        
        Args:
            task: Task data
            response: Model's response
            
        Returns:
            Evaluation results
        """
        reference_summary = task.get("reference_summary", "")
        
        # Compare with reference summary
        similarity = self._calculate_text_similarity(response, reference_summary)
        
        score = similarity
        evaluation = f"Summary similarity to reference: {similarity:.2f}"
        
        return {
            "score": score,
            "max_score": 1.0,
            "evaluation": evaluation
        }
    
    def evaluate_agent_decision(self, task: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate an agent decision-making task.
        
        Args:
            task: Task data
            response: Model's response
            
        Returns:
            Evaluation results
        """
        constraints = task.get("constraints", [])
        reference_solution = task.get("reference_solution", "")
        
        # Check if all constraints are satisfied
        constraint_satisfaction = 0.0
        satisfied_constraints = []
        
        for constraint in constraints:
            constraint_lower = constraint.lower()
            response_lower = response.lower()
            
            # Check if constraint keywords are present in the response
            constraint_keywords = re.findall(r'\b\w{4,}\b', constraint_lower)
            keyword_matches = sum(1 for keyword in constraint_keywords if keyword in response_lower)
            constraint_match = keyword_matches / len(constraint_keywords) if constraint_keywords else 0.0
            
            if constraint_match > 0.5:
                satisfied_constraints.append(constraint)
        
        constraint_satisfaction = len(satisfied_constraints) / len(constraints) if constraints else 1.0
        
        # Compare with reference solution
        solution_similarity = self._calculate_text_similarity(response, reference_solution)
        
        # Calculate overall score
        score = constraint_satisfaction * 0.7 + solution_similarity * 0.3
        evaluation = f"Satisfied {len(satisfied_constraints)}/{len(constraints)} constraints with {solution_similarity:.2f} similarity to reference."
        
        return {
            "score": score,
            "max_score": 1.0,
            "evaluation": evaluation,
            "satisfied_constraints": satisfied_constraints
        }
    
    def evaluate_multi_step_task(self, task: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate a multi-step task.
        
        Args:
            task: Task data
            response: Model's response
            
        Returns:
            Evaluation results
        """
        expected_steps = task.get("expected_steps", [])
        
        # Check which steps are covered in the response
        covered_steps = []
        response_lower = response.lower()
        
        for step in expected_steps:
            step_lower = step.lower()
            step_keywords = re.findall(r'\b\w{4,}\b', step_lower)
            matched_keywords = sum(1 for keyword in step_keywords if keyword in response_lower)
            step_coverage = matched_keywords / len(step_keywords) if step_keywords else 0.0
            
            if step_coverage > 0.5:
                covered_steps.append(step)
        
        step_coverage = len(covered_steps) / len(expected_steps) if expected_steps else 0.0
        
        score = step_coverage
        evaluation = f"Covered {len(covered_steps)}/{len(expected_steps)} expected steps."
        
        return {
            "score": score,
            "max_score": 1.0,
            "evaluation": evaluation,
            "covered_steps": covered_steps
        }
    
    def evaluate_information_extraction(self, task: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate an information extraction task.
        
        Args:
            task: Task data
            response: Model's response
            
        Returns:
            Evaluation results
        """
        questions = task.get("questions", [])
        reference_answers = task.get("reference_answers", [])
        
        if len(questions) != len(reference_answers):
            print(f"Warning: Number of questions ({len(questions)}) doesn't match number of reference answers ({len(reference_answers)})")
            return {
                "score": 0.0,
                "max_score": 1.0,
                "evaluation": "Error: Mismatched questions and reference answers in task definition."
            }
        
        # Check for each answer in the response
        correct_answers = 0
        
        for reference in reference_answers:
            reference_lower = reference.lower()
            response_lower = response.lower()
            
            # Check if response contains the reference answer
            if reference_lower in response_lower:
                correct_answers += 1
        
        accuracy = correct_answers / len(questions) if questions else 0.0
        
        return {
            "score": accuracy,
            "max_score": 1.0,
            "evaluation": f"Extracted {correct_answers}/{len(questions)} correct information points"
        }
    
    def evaluate_ethical_reasoning(self, task: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate an ethical reasoning task.
        
        Args:
            task: Task data
            response: Model's response
            
        Returns:
            Evaluation results
        """
        # Check for ethical framework
        framework_terms = ["utilitarian", "deontolog", "virtue", "ethics", "kantian", "consequential", "right", "duty", "principle", "value"]
        has_framework = any(term in response.lower() for term in framework_terms)
        
        # Check for multiple perspectives
        perspective_indicators = ["on one hand", "on the other hand", "however", "alternatively", "some would argue", "others might say", "different perspective", "another view"]
        has_perspectives = any(indicator in response.lower() for indicator in perspective_indicators)
        
        # Score based on presence of framework and perspectives
        score = 0.0
        if has_framework:
            score += 0.5
        if has_perspectives:
            score += 0.5
        
        evaluation = f"Ethical framework: {'Present' if has_framework else 'Absent'}; Multiple perspectives: {'Present' if has_perspectives else 'Absent'}"
        
        return {
            "score": score,
            "max_score": 1.0,
            "evaluation": evaluation
        }
    
    # Utility functions
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        This is a simple implementation using word overlap.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple word overlap coefficient
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        smaller_set = min(len(words1), len(words2))
        
        return len(intersection) / smaller_set
    
    def _extract_code(self, response: str, language: str) -> str:
        """
        Extract code from a response.
        
        Args:
            response: Model's response
            language: Programming language
            
        Returns:
            Extracted code
        """
        # Try to extract code blocks with markdown formatting
        code_block_pattern = f"```(?:{language})?(.*?)```"
        code_blocks = re.findall(code_block_pattern, response, re.DOTALL)
        
        if code_blocks:
            return code_blocks[0].strip()
        
        # If no code blocks found, try to extract indented code
        lines = response.split("\n")
        indented_lines = []
        in_code_section = False
        
        for line in lines:
            if line.strip().lower().startswith("```") or line.strip().lower().startswith("code:"):
                in_code_section = not in_code_section
                continue
            
            if in_code_section or line.startswith("    ") or line.startswith("\t"):
                indented_lines.append(line)
        
        if indented_lines:
            return "\n".join(indented_lines).strip()
        
        # If still no code found, return the whole response
        return response
    
    def _calculate_code_similarity(self, code1: str, code2: str) -> float:
        """
        Calculate similarity between two code snippets.
        
        Args:
            code1: First code snippet
            code2: Second code snippet
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize code
        code1_norm = self._normalize_code(code1)
        code2_norm = self._normalize_code(code2)
        
        # Calculate similarity using Jaccard index of lines
        lines1 = set(line.strip() for line in code1_norm.split("\n") if line.strip())
        lines2 = set(line.strip() for line in code2_norm.split("\n") if line.strip())
        
        if not lines1 or not lines2:
            return 0.0
        
        intersection = lines1.intersection(lines2)
        union = lines1.union(lines2)
        
        return len(intersection) / len(union)
    
    def _normalize_code(self, code: str) -> str:
        """
        Normalize code by removing comments and extra whitespace.
        
        Args:
            code: Code to normalize
            
        Returns:
            Normalized code
        """
        # Remove comments (this is a simplistic approach)
        code_no_comments = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        
        # Remove extra whitespace
        lines = [line.strip() for line in code_no_comments.split("\n")]
        return "\n".join(line for line in lines if line)