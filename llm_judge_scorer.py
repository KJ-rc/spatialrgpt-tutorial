"""
Spatial Reasoning Evaluation Script with Unified API Support

This script evaluates spatial reasoning model responses using aisuite for unified API access.
Supports multiple LLM providers (OpenAI, Google, Anthropic, etc.) through aisuite.

Features:
- Unified API support via aisuite
- Quantitative evaluation (distances, directions, sizes)  
- Qualitative evaluation (spatial relationships)
- Comprehensive metrics calculation
- Error analysis and success rates
- JSON output for results and raw data

Author: Refactored for unified API support
Date: September 2025
"""

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from tqdm.autonotebook import tqdm

# Import aisuite (installed as package)
import aisuite as ai

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

# ============================================================================
# UTILITY CLASSES FOR RUNNING STATISTICS
# ============================================================================

class RunningAverage:
    """Maintains a running average of values."""
    
    def __init__(self):
        self.avg = 0.0
        self.count = 0

    def append(self, value: float) -> None:
        """Add a new value to the running average."""
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self) -> float:
        """Get the current average value."""
        return self.avg


class RunningAverageDict:
    """A dictionary of running averages for multiple metrics."""

    def __init__(self):
        self._dict = {
            'a1': RunningAverage(),
            'a2': RunningAverage(), 
            'a3': RunningAverage(),
            'abs_rel': RunningAverage(),
            'rmse': RunningAverage(),
            'log_10': RunningAverage(),
            'rmse_log': RunningAverage(),
            'silog': RunningAverage(),
            'sq_rel': RunningAverage(),
        }

    def update(self, new_dict: Optional[Dict[str, float]]) -> None:
        """Update all running averages with new values."""
        if new_dict is None:
            return

        if self._dict is None:
            self._dict = {}
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            if key in self._dict:
                self._dict[key].append(value)

    def get_value(self) -> Dict[str, float]:
        """Get current average values for all metrics."""
        return {key: avg.get_value() for key, avg in self._dict.items()}


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_errors(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """
    Compute various error metrics between ground truth and predictions.
    
    Args:
        gt: Ground truth values
        pred: Predicted values
        
    Returns:
        Dictionary of error metrics
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err**2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    
    return {
        'a1': a1, 'a2': a2, 'a3': a3, 'abs_rel': abs_rel, 'rmse': rmse, 
        'log_10': log_10, 'rmse_log': rmse_log, 'silog': silog, 'sq_rel': sq_rel
    }


# ============================================================================
# UNIFIED API EVALUATION FUNCTIONS
# ============================================================================

class SpatialEvaluator:
    """Spatial reasoning evaluator using unified API via aisuite."""
    
    def __init__(self, model: str = "openai:gpt-4-turbo"):
        """
        Initialize evaluator with specified model.
        
        Args:
            model: Model name in aisuite format (e.g., "openai:gpt-4-turbo", "google:gemini-2.0-flash-exp")
        """
        self.model = model
        self.client = ai.Client()
        
    def _make_api_call(self, messages: List[Dict[str, str]]) -> str:
        """
        Make unified API call using aisuite.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Response content as string
        """
        try:
            # Prepare API parameters
            api_params = {
                "model": self.model,
                "messages": messages,
                "response_format": {"type": "json_object"}
            }
            
            # Handle model-specific parameters
            if "gpt-5" in self.model:
                api_params["max_completion_tokens"] = 1024
                # GPT-5 only supports temperature=1.0 (default)
                # Don't set temperature parameter for GPT-5
            else:
                api_params["max_tokens"] = 1024
                api_params["temperature"] = 0.0
                
            response = self.client.chat.completions.create(**api_params)
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call failed: {e}")
            return "{}"

    def evaluate_quantitative_direction(self, question: str, answer: str, response: str) -> Optional[Dict[str, int]]:
        """
        Evaluate quantitative direction questions using unified API.
        
        Args:
            question: The question text
            answer: Ground truth answer
            response: Model response to evaluate
            
        Returns:
            Dictionary with answer_direction and response_direction
        """
        prompt = f"""
You should help me to evaluate the response given the question and the correct answer.
You need to convert the directions to numbers based on clock positions (12 o'clock = 12, 1 o'clock = 1, etc.).
You should output two integers, one for the answer, and one for the response.
The output should be in JSON format.

Example 1:
Question: If you are at Region [0], where will you find Region [1]?
Answer: Region [0] will find Region [1] around the 10 o'clock direction.
Response: If you are at Region [0], you will find Region [1] around the 9 o'clock direction.
"answer_direction": 10, "response_direction": 9

Example 2:
Question: If you are at Region [0], where will you find Region [1]?
Answer: Region [0] will find Region [1] around the 12 o'clock direction.
Response: If you are at Region [0], you will find Region [1] around the 11 o'clock direction.
"answer_direction": 12, "response_direction": 11

Your Turn:
Question: {question}
Answer: {answer}
Response: {response}
"""

        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt}
        ]
        
        response_content = self._make_api_call(messages)
        
        try:
            json_response = json.loads(response_content)
            return json_response
        except json.JSONDecodeError:
            return None

    def evaluate_quantitative_distance(self, question: str, answer: str, response: str) -> Optional[Dict[str, float]]:
        """
        Evaluate quantitative distance/size questions using unified API.
        
        Args:
            question: The question text
            answer: Ground truth answer
            response: Model response to evaluate
            
        Returns:
            Dictionary with answer_in_meters and response_in_meters
        """
        prompt = f"""
You should help me to evaluate the response given the question and the correct answer.
You need to convert the distance of the correct answer and response to meters. The conversion factors are as follows: 1 inch = 0.0254 meters. 1 foot = 0.3048 meters. 1 centimeter (cm) = 0.01 meters.
You should output two floats in meters, one for the answer, and one for the response.
The output should be in JSON format.

Example 1:
Question: Measure the width of Region [0].?
Answer: The width of Region [0] is 1.02 meters.
Response: Region [0] is 2.17 meters wide.
"answer_in_meters": 1.02, "response_in_meters": 2.17

Example 2:
Question: What is the height of Region [1]?
Answer: The height of Region [1] is 10.0 inches.
Response: It is 48.47 centimeters.
"answer_in_meters": 0.25, "response_in_meters": 0.48

Example 3:
Question: What is the radius of Region [0]?
Answer: Region [0] is 77.56 centimeters wide.
Response: It is 35.9 inches wide.
"answer_in_meters": 0.78, "response_in_meters": 0.91

Your Turn:
Question: {question}
Answer: {answer}
Response: {response}
"""

        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt}
        ]
        
        response_content = self._make_api_call(messages)
        
        try:
            json_response = json.loads(response_content) 
            return json_response
        except json.JSONDecodeError:
            return None

    def evaluate_qualitative(self, question: str, answer: str, response: str, category: str) -> Optional[int]:
        """
        Evaluate qualitative spatial relationship questions using unified API.
        
        Args:
            question: The question text
            answer: Ground truth answer
            response: Model response to evaluate
            category: Question category for context
            
        Returns:
            Evaluation score (0 or 1)
        """
        # Category-specific examples
        examples = ""
        if "left" in category or "right" in category:
            examples = """
Example 1:
Question: Which object is on the left, Region [0] or Region [1]?
Answer: Region [0] is on the left.
Response: Region [0] is positioned to the left of Region [1].
Your mark: 1

Example 2:
Question: Between Region [0] and Region [1], which one is on the right?
Answer: Region [1] is on the right.
Response: Region [0] is on the right side.
Your mark: 0
"""
        elif "behind" in category or "front" in category:
            examples = """
Example 1:
Question: Is Region [0] behind Region [1]?
Answer: Yes, Region [0] is behind Region [1].
Response: Region [0] appears to be positioned behind Region [1].
Your mark: 1

Example 2:
Question: Which is in front, Region [0] or Region [1]?
Answer: Region [1] is in front.
Response: Region [0] is in the front.
Your mark: 0
"""
        elif "below" in category or "above" in category:
            examples = """
Example 1:
Question: Is Region [0] above Region [1]?
Answer: Yes, Region [0] is above Region [1].
Response: Region [0] is positioned above Region [1].
Your mark: 1

Example 2:
Question: Which is below, Region [0] or Region [1]?
Answer: Region [1] is below.
Response: Region [0] is below Region [1].
Your mark: 0
"""
        else:
            examples = """
Example 1:
Question: Is Region [0] bigger than Region [1]?
Answer: Yes, Region [0] is bigger than Region [1].
Response: Region [0] appears to be larger in size compared to Region [1].
Your mark: 1

Example 2:
Question: Which is smaller, Region [0] or Region [1]?
Answer: Region [1] is smaller.
Response: Region [0] is the smaller one.
Your mark: 0
"""

        post_fix = f"""
Your Turn:
Question: {question}
Answer: {answer}
Response: {response}
"""

        prompt = f"""
You should help me to evaluate the response given the question and the correct answer.
Please carefully analyze the semantic meaning of both the correct answer and the response.
If they convey the same meaning, even with different wording, mark it as 1 (correct).
If they convey different meanings, mark it as 0 (incorrect).
The output should be in JSON format with "your_mark" as the key.

{examples}

{post_fix}
"""

        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt}
        ]
        
        response_content = self._make_api_call(messages)
        
        try:
            json_response = json.loads(response_content)
            return json_response.get("your_mark")
        except json.JSONDecodeError:
            return None


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def main():
    """Main evaluation pipeline."""
    if len(sys.argv) < 2:
        print("Usage: python evaluate_spatial_unified.py <data_path> [model_name]")
        print("Example: python evaluate_spatial_unified.py responses.jsonl openai:gpt-4-turbo")
        sys.exit(1)
    
    data_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "openai:gpt-4-turbo"
    
    print(f"Using model: {model_name}")
    print(f"Evaluating: {data_path}")
    
    # Initialize evaluator
    evaluator = SpatialEvaluator(model=model_name)
    
    # Load data
    with open(data_path) as f:
        lines = f.readlines()

    total = len(lines)
    print(f"Total samples: {total}")

    # Initialize collections
    qualitative_dict = defaultdict(list)
    quantitative_success_dict = defaultdict(list)
    quantitative_error_dict = defaultdict(list)
    raw_list = []
    match_fail_count = 0

    # Process each sample
    for line in tqdm(lines, desc="Processing samples"):
        data = json.loads(line)
        data["llm_match_info"] = {}
        match_success = False

        if data["qa_info"]["type"] == "quantitative":
            category = data["qa_info"]["category"]
            
            if category in ["vertical_distance_data", "horizontal_distance_data", 
                          "distance_data", "width_data", "height_data", "direction"]:
                
                if category == "direction":
                    # Direction evaluation
                    try:
                        eval_result = evaluator.evaluate_quantitative_direction(
                            question=data["question"], 
                            answer=data["gt"], 
                            response=data["pred"]
                        )
                        
                        if eval_result:
                            data["llm_match_info"]["answer"] = eval_result["answer_direction"]
                            data["llm_match_info"]["response"] = eval_result["response_direction"]
                            
                            # Calculate direction error
                            answer_dir = eval_result["answer_direction"]
                            response_dir = eval_result["response_direction"]
                            error = min(abs(answer_dir - response_dir), 12 - abs(answer_dir - response_dir))
                            error_rate = error / 12 * 100
                            
                            success = 1 if error <= 1 else 0
                            match_success = True
                        else:
                            success = 0
                            error_rate = 100
                            
                    except Exception:
                        success = 0
                        error_rate = 100
                        match_fail_count += 1
                        
                else:
                    # Distance/size evaluation
                    try:
                        eval_result = evaluator.evaluate_quantitative_distance(
                            question=data["question"],
                            answer=data["gt"], 
                            response=data["pred"]
                        )
                        
                        if eval_result:
                            data["llm_match_info"]["answer"] = eval_result["answer_in_meters"]
                            data["llm_match_info"]["response"] = eval_result["response_in_meters"]
                            
                            # Calculate error rate
                            gt_val = eval_result["answer_in_meters"]
                            pred_val = eval_result["response_in_meters"]
                            error_rate = abs(gt_val - pred_val) / gt_val * 100
                            
                            success = 1 if error_rate <= 25 else 0
                            match_success = True
                        else:
                            success = 0
                            error_rate = 100
                            
                    except Exception:
                        success = 0
                        error_rate = 100
                        match_fail_count += 1

                # Store results by category
                category_map = {
                    "vertical_distance_data": "vertical_distance",
                    "horizontal_distance_data": "horizontal_distance", 
                    "distance_data": "direct_distance",
                    "width_data": "width",
                    "height_data": "height",
                    "direction": "direction"
                }
                
                mapped_category = category_map[category]
                quantitative_success_dict[mapped_category].append(int(success))
                quantitative_error_dict[mapped_category].append(error_rate)

        elif data["qa_info"]["type"] == "qualitative":
            category = data["qa_info"]["category"]
            
            try:
                evaluation = evaluator.evaluate_qualitative(
                    question=data["question"],
                    answer=data["gt"],
                    response=data["pred"], 
                    category=category
                )
                
                if evaluation is not None:
                    data["llm_match_info"]["evaluation"] = int(evaluation)
                    match_success = True
                else:
                    data["llm_match_info"]["evaluation"] = "N/A"
                    evaluation = 0
                    match_fail_count += 1
                    
            except Exception:
                data["llm_match_info"]["evaluation"] = "N/A"
                evaluation = 0
                match_fail_count += 1

            # Store qualitative results
            if "short" in category or "tall" in category:
                qualitative_dict["tall/short"].append(int(evaluation))
            elif "wide" in category or "thin" in category:
                qualitative_dict["wide/thin"].append(int(evaluation))
            elif "big" in category or "small" in category:
                qualitative_dict["big/small"].append(int(evaluation))
            elif "left" in category or "right" in category:
                qualitative_dict["left/right"].append(int(evaluation))
            elif "below" in category or "above" in category:
                qualitative_dict["below/above"].append(int(evaluation))
            elif "behind" in category or "front" in category:
                qualitative_dict["behind/front"].append(int(evaluation))

        raw_list.append(data)

    # Calculate final metrics
    result_dict = {}
    
    # Qualitative metrics
    correct_qualitative = 0
    total_qualitative = 0
    
    for qual_cat, scores in qualitative_dict.items():
        if scores:
            accuracy = sum(scores) / len(scores) * 100
            result_dict[f"Qual_{qual_cat}_acc"] = accuracy
            correct_qualitative += sum(scores)
            total_qualitative += len(scores)

    # Quantitative metrics
    correct_quantitative = 0 
    total_quantitative = 0
    
    for quan_cat, successes in quantitative_success_dict.items():
        if successes:
            accuracy = sum(successes) / len(successes) * 100
            result_dict[f"Quan_{quan_cat}_acc"] = accuracy
            correct_quantitative += sum(successes)
            total_quantitative += len(successes)
            
    for quan_cat, errors in quantitative_error_dict.items():
        if errors:
            avg_error = sum(errors) / len(errors)
            result_dict[f"Quan_{quan_cat}_err"] = avg_error

    # Advanced quantitative metrics using compute_errors
    final_metrics = defaultdict(RunningAverageDict)
    
    for data in raw_list:
        if data["qa_info"]["type"] == "quantitative":
            category = data["qa_info"]["category"]
            
            if category in ["vertical_distance_data", "horizontal_distance_data", 
                          "distance_data", "width_data", "height_data"]:
                
                if "answer" in data["llm_match_info"] and "response" in data["llm_match_info"]:
                    gt_val = data["llm_match_info"]["answer"]
                    pred_val = data["llm_match_info"]["response"]
                    
                    try:
                        error_metrics = compute_errors(
                            np.array([gt_val])[None], 
                            np.array([pred_val])[None]
                        )
                        final_metrics[category].update(error_metrics)
                    except Exception:
                        continue

    # Add advanced metrics to results
    for category, metrics in final_metrics.items():
        final_values = {k: round(v, 3) for k, v in metrics.get_value().items()}
        result_dict[f"Quan_{category}_absrel"] = final_values["abs_rel"]

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print("\nQualitative Results:")
    for qual_cat in qualitative_dict.keys():
        acc_key = f"Qual_{qual_cat}_acc"
        if acc_key in result_dict:
            print(f"{qual_cat}: {result_dict[acc_key]:.2f}%")
    
    print("\nQuantitative Results:")
    for quan_cat in quantitative_success_dict.keys():
        acc_key = f"Quan_{quan_cat}_acc"
        err_key = f"Quan_{quan_cat}_err"
        acc_val = result_dict.get(acc_key, 0)
        err_val = result_dict.get(err_key, 0)
        print(f"{quan_cat}: {acc_val:.2f}% / {err_val:.2f}% error")
    
    print(f"\nOverall Statistics:")
    print(f"Qualitative Overall: {correct_qualitative/total_qualitative*100:.2f}%" if total_qualitative > 0 else "N/A")
    print(f"Quantitative Overall: {correct_quantitative/total_quantitative*100:.2f}%" if total_quantitative > 0 else "N/A")
    print(f"Match Failures: {match_fail_count}")

    # Save results
    result_dict["Qual_overall_acc"] = correct_qualitative/total_qualitative*100 if total_qualitative > 0 else 0
    result_dict["Quan_overall_acc"] = correct_quantitative/total_quantitative*100 if total_quantitative > 0 else 0
    result_dict["Match_fail_count"] = match_fail_count

    data_path_parent_dir = os.path.dirname(data_path)
    
    # Save score summary
    score_path = os.path.join(data_path_parent_dir, "score.json")
    with open(score_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"\nResults saved to: {score_path}")
    
    # Save raw evaluation data
    raw_path = os.path.join(data_path_parent_dir, "raw_evaluation.json") 
    with open(raw_path, "w") as f:
        json.dump(raw_list, f, indent=2)
    print(f"Raw data saved to: {raw_path}")


if __name__ == "__main__":
    main()