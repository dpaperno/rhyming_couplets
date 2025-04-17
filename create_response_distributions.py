# %%
import os
import json
import time
import requests
from typing import List, Dict, Any
import anthropic
from transformers import AutoTokenizer
import re
import sys
from functools import partial
from itertools import product
from pathlib import Path
from typing import Callable, Literal
import circuitsvis as cv
import einops
import numpy as np
import plotly.express as px
import torch as t
from IPython.display import HTML, display
from jaxtyping import Bool, Float, Int
from rich import print as rprint
from rich.table import Column, Table
from torch import Tensor
from tqdm.notebook import tqdm
from transformer_lens import ActivationCache, HookedTransformer, utils
from transformer_lens.components import MLP, Embed, LayerNorm, Unembed
from transformer_lens.hook_points import HookPoint


# %%
def load_couplet_starts(filename="couplet_starts.txt"):
    """
    Load couplet starts from a text file where each line is a couplet start.
    
    Args:
        filename (str): Path to the text file containing couplet starts
        
    Returns:
        list: List of couplet start strings
    """
    couplet_starts = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:  # Skip empty lines
                    couplet_starts.append(line.strip("'"))
        print(f"Loaded {len(couplet_starts)} couplet starts from {filename}")
    except FileNotFoundError:
        print(f"File not found: {filename}")
    except Exception as e:
        print(f"Error loading couplet starts: {str(e)}")
    
    return couplet_starts

couplet_starts = load_couplet_starts()
print(couplet_starts)


# %%
from huggingface_hub import login
login("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")

# %%

# Load the Gemma 2 9B Instruct Tuned model
model = HookedTransformer.from_pretrained(
    "gemma-2-9b-it",
    dtype=t.bfloat16,
    device="cuda:0"
)


# %%
# # You can verify the model is loaded correctly
# Verify model by checking for attributes with "name" or "model" in them
print(f"Model name: {model.cfg.model_name}")
# Alternative ways to verify the model
print(f"Model has {model.cfg.n_layers} layers and {model.cfg.d_model} hidden dimensions")


for l1 in couplet_starts:
    print(l1)


# %%

long_couplet_starts = couplet_starts
couplet_starts = ["A single rose, its petals unfold", "Beneath the oak, a squirrel scurries by",
                  "He saw a carrot and had to grab it", "He saw a wallet and had to grab it", 
                  "The silver moon cast its gentle light","Boxes of books, a reader's delight",
                  "Footsteps echoing on the schoolyard bricks", "Footsteps echoing on the prison yard bricks"]

# %%
def generate_responses(input_string, model, n_samples=100):
    """
    Generate n_samples responses from the model for a given input string.
    
    Args:
        input_string: The couplet start to use as input
        model: The loaded model to use for generation
        n_samples: Number of responses to generate (default: 100)
        
    Returns:
        A list of strings, each containing the input followed by the model's response
    """
    prompt = f""""This is the first line in a rhyming couplet in a poem:

{input_string}

Please respond only with a rhyming second line that makes a nice couplet with the first line.
Put it in a JSON format with no other text but the second line of the couplet. The JSON key should be 'line'.
The value should be the rhyming line."""
    
    responses = []
    
    print(f"Generating {n_samples} responses for: {input_string}")
    
    for i in range(n_samples):
        if i % 10 == 0:
            print(f"  Progress: {i}/{n_samples}")
        
        # Check if the response is empty or just whitespace
        max_retries = 3
        retry_count = 0
        
        output_text = ""
        while (not output_text.strip()) and retry_count < max_retries:
            if retry_count > 0:
                print(f"Empty response detected, retrying ({retry_count+1}/{max_retries})...")
            retry_count += 1
            
            # Generate a response from the model
            with t.no_grad():
                output = model.generate(
                    prompt, 
                    max_new_tokens=30,
                    temperature=0.85,
                    top_p=0.9,
                    do_sample=True,
                    top_k=50
                )
            output_text = output[len(prompt):]
            # Try to parse JSON from the response using the json library
            try:
                # Clean up the string to help with JSON parsing
                # Remove any leading/trailing whitespace
                json_str = output_text.strip()
                
                # Try to find JSON within markdown code blocks
                json_start = json_str.find("```json")
                if json_start != -1:
                    json_start = json_str.find("\n", json_start) + 1
                    json_end = json_str.find("```", json_start)
                    json_str = json_str[json_start:json_end].strip()
                
                # Parse the JSON and extract the "line" value
                data = json.loads(json_str)
                if isinstance(data, dict) and "line" in data:
                    output_text = data["line"]
                else:
                    print(f"JSON found but missing 'line' key: {data}")
            except json.JSONDecodeError:
                print(f"Failed to parse JSON from response: {output_text}")
            except Exception as e:
                print(f"Error extracting line from JSON: {str(e)}")
            # Clean up the output text by removing unwanted characters and words
            output_text = output_text.strip()  # Remove leading/trailing whitespace
            output_text = output_text.replace('*', '').replace(':', '')  # Remove * and : characters
            
            # Remove "answer" or "your answer" only if they appear at the beginning of the string (case insensitive)
            output_text = re.sub(r'(?i)^(answer|your answer)\s*', '', output_text)

            # Remove any leading/trailing whitespace again after the replacements
            output_text = output_text.strip()
        
        full_response = [input_string, output_text]
        responses.append(full_response)
    
    # Check if each response rhymes using Claude 3.5 Haiku
    rhyme_checks = []
    
    print(f"Checking rhymes for {n_samples} responses")
    
    client = anthropic.Anthropic(api_key="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")
    
    for i, response in enumerate(responses):
        if i % 10 == 0:
            print(f"  Rhyme check progress: {i}/{n_samples}")
        
        if len(response) >= 2:
            line1, line2 = response[0], response[1]
            
            # Prompt for Claude to check if the lines rhyme
            rhyme_prompt = f"""Do these two lines rhyme? Respond with ONLY 'yes' or 'no' and nothing else.

Line 1: {line1}
Line 2: {line2}"""
            
            # Call Claude API with retry logic
            max_retries = 3
            retry_delay = 2
            for attempt in range(max_retries):
                try:
                    message = client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=1,
                        temperature=0,
                        system="You are a helpful assistant that only responds with 'yes' or 'no' when asked if lines rhyme.",
                        messages=[
                            {"role": "user", "content": rhyme_prompt}
                        ]
                    )
                    
                    # Get the response and add to the list
                    rhyme_result = message.content[0].text.strip().lower()
                    rhyme_checks.append(rhyme_result)
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"  API call failed, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print(f"  Failed to check rhyme after {max_retries} attempts: {e}")
                        rhyme_checks.append("error")
        else:
            print(f"  Warning: Response {i} doesn't contain two lines")
            rhyme_checks.append("error")
    
    print(f"Completed rhyme checking for {n_samples} responses")
    
    print(f"Completed generating {n_samples} responses")
    return responses, rhyme_checks


# %%
import pandas as pd
from tqdm import tqdm

def create_response_dataframe(couplet_starts, model, n_samples=100):
    """
    Generate responses for each couplet start and store in a pandas DataFrame.
    
    Args:
        couplet_starts (list): List of first lines of couplets
        model: The language model to use for generation
        n_samples (int): Number of responses to generate for each couplet start
        
    Returns:
        pd.DataFrame: DataFrame containing all responses and rhyme checks
    """
    all_data = []
    
    for i, start in enumerate(tqdm(couplet_starts, desc="Processing couplet starts")):
        print(f"\nProcessing couplet start {i+1}/{len(couplet_starts)}: '{start}'")
        responses, rhyme_checks = generate_responses(start, model, n_samples=n_samples)
        
        for j, (response, rhyme_check) in enumerate(zip(responses, rhyme_checks)):
            if len(response) >= 2:
                all_data.append({
                    'first_line': response[0],
                    'second_line': response[1],
                    'rhymes': rhyme_check
                })
            else:
                all_data.append({
                    'first_line': start,
                    'second_line': "ERROR: Invalid response",
                    'rhymes': "error"
                })
    
    # Create DataFrame from collected data
    df = pd.DataFrame(all_data)
    
    print(f"Created DataFrame with {len(df)} rows")
    return df

# %%

# Generate responses in batches of 20 samples, for a total of 100 samples
batch_size = 20
num_batches = 5
all_dataframes = []

print(f"Generating {num_batches} batches of {batch_size} samples each...")

for batch in range(num_batches):
    print(f"\nProcessing batch {batch+1}/{num_batches}")
    batch_df = create_response_dataframe(couplet_starts, model, n_samples=batch_size)
    all_dataframes.append(batch_df)
    print(f"Completed batch {batch+1}/{num_batches} with {len(batch_df)} responses")
    
    # Save intermediate results
    batch_df.to_csv(f"couplet_responses_batch_{batch+1}.csv", index=False)
    print(f"Saved batch {batch+1} to couplet_responses_batch_{batch+1}.csv")

# Combine all dataframes
response_df = pd.concat(all_dataframes, ignore_index=True)
print(f"Combined all batches into final DataFrame with {len(response_df)} rows")

# Display the first few rows of the DataFrame
print(response_df.head())

# Save the DataFrame to a CSV file
response_df.to_csv("couplet_responses.csv", index=False)
print("Saved responses to couplet_responses.csv")

