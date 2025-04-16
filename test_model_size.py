# %%
import torch
from nnsight import LanguageModel
import os
import json
import time
import requests
from typing import List, Dict, Any
import anthropic

# %%

def generate_couplet_dataset(
    prompt_template: str,
    dataset_size: int = 30,
    couplets_per_call: int = 5,
    model: str = "claude-3-haiku-20240307",
    api_key: str = "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",
    max_retries: int = 3,
    retry_delay: int = 2
) -> List[str]:
    """
    Generate a dataset of potential first lines for rhyming couplets using Claude 3.5 Haiku.
    
    Args:
        prompt_template: The prompt to send to Claude. Should instruct the model to respond with JSON.
        dataset_size: Total number of couplet starts to generate.
        couplets_per_call: Number of couplet starts to request in each API call.
        model: The Claude model to use.
        api_key: Anthropic API key. If None, will try to get from ANTHROPIC_API_KEY env var.
        max_retries: Maximum number of retries for failed API calls.
        retry_delay: Delay between retries in seconds.
        
    Returns:
        List of strings, each being a potential first line for a rhyming couplet.
    """
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError("API key must be provided or set as ANTHROPIC_API_KEY environment variable")
    
    client = anthropic.Anthropic(api_key=api_key)
    all_couplet_starts = []
    calls_needed = (dataset_size + couplets_per_call - 1) // couplets_per_call  # Ceiling division
    
    for i in range(calls_needed):
        remaining = dataset_size - len(all_couplet_starts)
        current_batch_size = min(couplets_per_call, remaining)
        
        # Update the prompt to request the current batch size
        current_prompt = prompt_template.format(num_couplets=current_batch_size)
        
        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=4000,
                    temperature=0.7,
                    system="You are a helpful assistant that responds in JSON format.",
                    messages=[
                        {"role": "user", "content": current_prompt}
                    ]
                )
                
                # Extract the content from the response
                content = response.content[0].text
                
                # Try to parse JSON from the response
                # First, try to find JSON within markdown code blocks
                json_start = content.find("```json")
                if json_start != -1:
                    json_start = content.find("\n", json_start) + 1
                    json_end = content.find("```", json_start)
                    json_str = content[json_start:json_end].strip()
                else:
                    # If no markdown code block, try to parse the whole response
                    json_str = content
                
                try:
                    data = json.loads(json_str)
                    
                    # Handle different possible JSON structures
                    if isinstance(data, list):
                        couplet_starts = data
                    elif isinstance(data, dict) and "couplet_starts" in data:
                        couplet_starts = data["couplet_starts"]
                    elif isinstance(data, dict) and "couplets" in data:
                        couplet_starts = data["couplets"]
                    else:
                        # Try to find any list in the response
                        for key, value in data.items():
                            if isinstance(value, list) and len(value) > 0:
                                couplet_starts = value
                                break
                        else:
                            raise ValueError("Could not find couplet data in response")
                    
                    all_couplet_starts.extend(couplet_starts)
                    print(f"Generated {len(couplet_starts)} couplet starts. Total: {len(all_couplet_starts)}/{dataset_size}")
                    break  # Success, exit retry loop
                    
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON from response. Attempt {attempt+1}/{max_retries}")
                    if attempt == max_retries - 1:
                        raise
            
            except Exception as e:
                print(f"Error during API call: {str(e)}. Attempt {attempt+1}/{max_retries}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(retry_delay)
    
    # Trim to exact dataset size if we got more than requested
    return all_couplet_starts[:dataset_size]

# %%
prompt = "I'd like to build a small data set. I want the first line of potential rrhyming couplets like you would see in a simple, traditional poem. Here are some examples. 'He saw a carrot and had to grab it'. 'The silver moon casts its gentle light'. 'The clouds are gray, the raindrops fall'. And 'Boxes of books, a reader's delight'. Please write {num_couplets} first lines and return them in a JSON format. Return nothing else. Please don't write any intro or outro text. What you write will be read automatically into a JSON reader. Any extra writing will break the read."
couplet_first_lines = generate_couplet_dataset(prompt)


# %%

# Load the Gemma 3 4B Instruct Tuned model
# model = LanguageModel("google/gemma-3-4b-it", device_map="cuda")

# # You can verify the model is loaded correctly
# print(f"Model loaded: {model.model_name}")
# print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B")

# %%
