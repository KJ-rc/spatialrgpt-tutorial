import argparse
import json
import os
from pathlib import Path
import re
from collections import defaultdict

import numpy as np
from tqdm.autonotebook import tqdm
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

# Import aisuite (installed as package)
import aisuite as ai

# Import functions from batch_evaluator.py
from batch_evaluator import load_image, create_messages

# Unified API for evaluation
import aisuite as ai
from typing import List, Dict, Optional


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
            content = response.choices[0].message.content
            
            # Strip markdown code blocks if present
            if content.startswith("```json") and content.endswith("```"):
                content = content[7:-3].strip()
            elif content.startswith("```") and content.endswith("```"):
                content = content[3:-3].strip()
                
            return content
        except Exception as e:
            print(f"API call failed: {e}")
            print(f"Model: {self.model}")
            print(f"Messages: {messages}")
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
        print(f"[DEBUG] Direction evaluation response: {response_content}")
        
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
Question: What is the distance between the camera and the nearest book shelf?
Answer: The distance between the camera and the nearest book shelf is 2.5 meters.
Response: The distance between the camera and the nearest book shelf is approximately 2.3 meters.
"answer_in_meters": 2.5, "response_in_meters": 2.3

Example 2:
Question: How tall is the chair in the image?
Answer: The chair in the image is about 3 feet tall.
Response: The chair appears to be approximately 36 inches tall.
"answer_in_meters": 0.9144, "response_in_meters": 0.9144

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
        print(f"[DEBUG] Distance evaluation response: {response_content}")
        
        try:
            json_response = json.loads(response_content)
            return json_response
        except json.JSONDecodeError as e:
            print(f"JSON decode error in quantitative_distance: {e}")
            print(f"Response content: {response_content}")
            return None

    def evaluate_qualitative(self, question: str, answer: str, response: str) -> Optional[Dict[str, str]]:
        """
        Evaluate qualitative spatial relationship questions using unified API.
        
        Args:
            question: The question text
            answer: Ground truth answer
            response: Model response to evaluate
            
        Returns:
            Dictionary with status (correct/incorrect) and explanation
        """
        prompt = f"""
You should help me to evaluate the response given the question and the correct answer.
For questions about spatial relationships, determine if the response is correct or incorrect.
The output should be in JSON format with "status" and "explanation".

Example 1:
Question: What is the spatial relationship between the red car and the blue house?
Answer: The red car is parked in front of the blue house.
Response: The red car is positioned in front of the blue house.
"status": "correct", "explanation": "Both answers describe the same spatial relationship - the car being in front of the house."

Example 2:
Question: Where is the lamp relative to the sofa?
Answer: The lamp is to the left of the sofa.
Response: The lamp is positioned to the right of the sofa.
"status": "incorrect", "explanation": "The ground truth says left but the response says right - opposite directions."

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
        print(f"[DEBUG] Qualitative evaluation response: {response_content}")
        
        try:
            json_response = json.loads(response_content)
            return json_response
        except json.JSONDecodeError as e:
            print(f"JSON decode error in qualitative: {e}")
            print(f"Response content: {response_content}")
            return None


def eval_single_example(args):
    # Load questions (handle both JSON and JSONL formats)
    questions = []
    with open(args.annotation_file) as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON array format
            questions = json.loads(content)
        else:
            # JSONL format
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))
        
    
    if not questions:
        print("No questions found in the annotation file.")
        return
    
    # Get the example index (default to first example)
    example_index = args.example_index if hasattr(args, 'example_index') and args.example_index < len(questions) else 0
    
    if example_index >= len(questions):
        print(f"Example index {example_index} is out of range. Dataset has {len(questions)} examples.")
        return
    
    # Process single example
    line = questions[example_index]
    
    print(f"Processing single example {example_index}: {line['id']}")
    
    # Extract metadata
    question_id = line["id"]
    image_file = line["image_info"]["file_path"]
    text_question = line["text_q"]
    qa_info = line["qa_info"]
    ground_truth = line["conversations"][1]["value"]  # only first QA pair
    
    overlay_path = line["overlay"] if "overlay" in line else None

    # Load overlay image
    image = load_image(overlay_path)
    
    # Create messages for API call
    messages = create_messages(text_question, image, args.model)
    
    # Initialize aisuite client
    client = ai.Client()
    
    try:
        # Make API call using aisuite
        # Use appropriate parameters for different models
        params = {
            "model": args.model,
            "messages": messages,
        }
        
        if 'gemini' in args.model.lower():
            # Always disable thinking (only affects Gemini models)
            params["thinking_config"] = {"thinking_budget": 0}
        
        # GPT-5 has specific parameter requirements
        if "gpt-5" in args.model.lower():
            params["max_completion_tokens"] = 1024
            # GPT-5 only supports default temperature (1.0)
        else:
            params["max_tokens"] = 1024
            params["temperature"] = 0.0
            
        response = client.chat.completions.create(**params)            
        
        # Extract response content
        pred_content = response.choices[0].message.content
        
    except Exception as e:
        print(f"Error processing question {question_id}: {e}")
        pred_content = f"ERROR: {str(e)}"
    
    # Create result with metadata
    result = {
        "question_id": question_id,
        "image": image_file,
        "question": text_question,
        "qa_info": qa_info,
        "gt": ground_truth,
        "pred": pred_content,
        "model_id": args.model,
        "used_overlay": overlay_path is not None and os.path.exists(overlay_path)
    }
    
    # Print results to console
    print("\n" + "="*60)
    print("SINGLE EXAMPLE EVALUATION RESULT")
    print("="*60)
    print(f"Question ID: {question_id}")
    print(f"Image: {image_file}")
    print(f"Overlay used: {result['used_overlay']}")
    print(f"Question: {text_question}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Model Response: {pred_content}")
    print(f"QA Info: {qa_info}")
    print("="*60)
    
    # Run evaluation if we have ground truth and prediction
    if pred_content and not pred_content.startswith("ERROR:"):
        print("\n" + "="*60)
        print("AUTOMATIC EVALUATION")
        print("="*60)
        
        # Initialize evaluator with the same model used for the main task
        evaluator = SpatialEvaluator(model=args.model)
        
        # Determine question type and run appropriate evaluation
        question_type = qa_info.get("type", "").lower()
        question_category = qa_info.get("category", "").lower()
        
        if question_type == "quantitative" and "direction" in question_category:
            print(f"Running quantitative direction evaluation...")
            eval_result = evaluator.evaluate_quantitative_direction(text_question, ground_truth, pred_content)
            
            if eval_result:
                answer_dir = eval_result.get("answer_direction")
                response_dir = eval_result.get("response_direction")
                
                if answer_dir is not None and response_dir is not None:
                    # Calculate directional error (minimum angle difference)
                    angle_diff = min(abs(answer_dir - response_dir), 12 - abs(answer_dir - response_dir))
                    is_correct = angle_diff <= 1  # Allow 1 hour tolerance
                    
                    print(f"Ground truth direction: {answer_dir} o'clock")
                    print(f"Predicted direction: {response_dir} o'clock")
                    print(f"Angle difference: {angle_diff} hours")
                    print(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")
                    
                    result["evaluation"] = {
                        "type": "quantitative_direction",
                        "answer_direction": answer_dir,
                        "response_direction": response_dir,
                        "angle_difference": angle_diff,
                        "is_correct": is_correct
                    }
                else:
                    print("Failed to extract direction values from evaluation")
            else:
                print("Direction evaluation failed")
                
        elif question_type == "quantitative" and ("distance" in question_category or "size" in question_category or "height" in question_category or "width" in question_category or "depth" in question_category):
            print(f"Running quantitative distance/size evaluation...")
            eval_result = evaluator.evaluate_quantitative_distance(text_question, ground_truth, pred_content)
            
            if eval_result:
                answer_meters = eval_result.get("answer_in_meters")
                response_meters = eval_result.get("response_in_meters")
                
                if answer_meters is not None and response_meters is not None:
                    # Calculate relative error
                    if answer_meters > 0:
                        relative_error = abs(answer_meters - response_meters) / answer_meters
                        is_accurate = relative_error <= 0.25  # 25% tolerance
                        
                        print(f"Ground truth: {answer_meters:.3f} meters")
                        print(f"Predicted: {response_meters:.3f} meters")
                        print(f"Relative error: {relative_error:.1%}")
                        print(f"Result: {'ACCURATE' if is_accurate else 'INACCURATE'}")
                        
                        result["evaluation"] = {
                            "type": "quantitative_distance",
                            "answer_meters": answer_meters,
                            "response_meters": response_meters,
                            "relative_error": relative_error,
                            "is_accurate": is_accurate
                        }
                    else:
                        print("Invalid ground truth value (zero or negative)")
                else:
                    print("Failed to extract distance values from evaluation")
            else:
                print("Distance evaluation failed")
                
        elif question_type == "qualitative":
            print(f"Running qualitative evaluation...")
            eval_result = evaluator.evaluate_qualitative(text_question, ground_truth, pred_content)
            
            if eval_result:
                status = eval_result.get("status", "").lower()
                explanation = eval_result.get("explanation", "")
                is_correct = status == "correct"
                
                print(f"Status: {status.upper()}")
                print(f"Explanation: {explanation}")
                
                result["evaluation"] = {
                    "type": "qualitative",
                    "status": status,
                    "explanation": explanation,
                    "is_correct": is_correct
                }
            else:
                print("Qualitative evaluation failed")
        else:
            print(f"Unknown question type: {question_type} (category: {question_category})")
            print("Skipping automatic evaluation")
            
        print("="*60)
    
    # Optionally save result to file if output path is provided
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / f"single_example_{example_index}_result.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\nResult saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--annotation-file", type=str, required=True,
                       help="Path to annotation file (JSON or JSONL)")
    parser.add_argument("--output-path", type=str, default="",
                       help="Optional output directory to save result")
    parser.add_argument("--model", type=str, default="google:gemini-2.0-flash-exp", 
                       help="Model name in aisuite format (e.g., google:gemini-2.0-flash-exp, openai:gpt-4o)")
    parser.add_argument("--example-index", type=int, default=0,
                       help="Index of the example to process (default: 0, first example)")
    
    args = parser.parse_args()
    eval_single_example(args)