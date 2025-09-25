import argparse
import base64
from io import BytesIO
import json
import os
import sys
from pathlib import Path

from tqdm.autonotebook import tqdm
from PIL import Image
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

# Import aisuite (installed as package)
import aisuite as ai


# Global system prompt
SYSTEM_PROMPT = """You are an intelligent question-answering agent. I will ask you questions about spatial information regarding a scene, and you must provide an answer. Given a user query, you must output `text` to answer the question asked by the user. If a question lacks sufficient information for a precise answer, make a specific guess rather than suggesting a range.

User Query: How wide is the table (Region [0])?
Your Answer: Region [0] is 65 inches wide.

User Query: Can you confirm if the car (Region [0]) is taller than the barrier (Region [1])?
Your Answer: Yes, Region [0] is taller than Region [1].

User Query: If you are at the lamp (Region [0]), where will you find the pillow (Region [1])?
Your Answer: Region [1] is roughly at 10 o'clock from Region [0]."""


def encode_image_to_base64(image):
    """Encode PIL image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def create_messages(text_question, image=None, model=None):
    """Create messages for aisuite chat completion."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": []}
    ]
    
    # Add image if provided (using OpenAI-compatible format)
    if image is not None:
        messages[1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image_to_base64(image)}"
            }
        })
    
    # Add text question
    messages[1]["content"].append({
        "type": "text", 
        "text": f"User Query: {text_question}"
    })
    
    return messages


def load_image(overlay_path):
    """Load image, preferring overlay if available."""
    return Image.open(overlay_path).convert("RGB")


def eval_model(args):
    # Load questions (handle both JSON and JSONL formats)
    questions = []
    with open(args.annotation_file) as f:
        # Try to load as JSONL first (one JSON object per line)
        try:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))
        except json.JSONDecodeError:
            # If JSONL fails, try loading as regular JSON
            f.seek(0)
            questions = json.load(f)
    
    # Create output directory 
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize aisuite client
    client = ai.Client()
    
    # Process each question
    results = []
    
    for i, line in enumerate(tqdm(questions, desc="Processing questions")):
        # print(f"Processing question {i+1}/{len(questions)}: {line['id']}")
        
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
            
            if "gpt-5" in args.model.lower():
                pass
            #     # params["max_completion_tokens"] = 1024
            #     # GPT-5 only supports default temperature (1.0)
            else:
                # pass
                params["max_tokens"] = 1024
                # params["temperature"] = 0.0
                
            response = client.chat.completions.create(**params)            
            
            # Extract response content
            pred_content = response.choices[0].message.content
            
        except Exception as e:
            print(f"Error processing question {question_id}: {e}")
            pred_content = f"ERROR: {str(e)}"
        
        # Store result with metadata
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
        results.append(result)
    
    # Write final results
    output_file = output_path / "responses_with_predictions.jsonl"
    with open(output_file, 'w') as f:
        for result in results:
            f.write(f"{json.dumps(result)}\n")
    
    print(f"Evaluation completed. Results saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--annotation-file", type=str, default="")
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--model", type=str, default="google:gemini-2.0-flash-exp", 
                       help="Model name in aisuite format (e.g., google:gemini-2.0-flash-exp, openai:gpt-4o)")
    
    args = parser.parse_args()
    eval_model(args)