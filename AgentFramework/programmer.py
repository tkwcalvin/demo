# This file contains code copied and modified from the following repository:
# Repository: https://github.com/huangd1999/AgentCoder/tree/main
# Original Author: Dong Huang, Jie M.Zhang, Michael Luck, Qingwen Bu, Yuhao Qing, Heming Cui
# License: MIT

"""
AgentFramework Programmer Module
================================

This module implements the programmer agent component of the AgentCoder framework.
The programmer agent is responsible for generating code completions based on given problem descriptions.
It can handle both regular code generation and clarifying question generation for ambiguous problems.

Key Features:
- Generates multiple code completions for each problem
- Supports clarifying question generation for HumanEvalComm dataset
- Uses OpenAI's GPT models for code generation
- Implements concurrent processing for efficiency
- Handles error recovery and retry mechanisms

Main Functions:
- fetch_completion(): Generates code completions for a single problem
- programmer_main(): Orchestrates the code generation process for a dataset
- call_fetch_completion_helper(): Helper function for bug fixing iterations
"""

import json
from tqdm import tqdm
import copy
import openai
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import time
import os

print("Current Working Directory:", os.getcwd())

# Load the few-shot prompt template for code generation
prompt_path = "./prompts/humaneval_prompt_update.txt"
with open(prompt_path, "r") as f:
    construct_few_shot_prompt = f.read()

def preprocess_data(completion_string):
    """
    Extract Python code from markdown code blocks in the completion string.
    
    Args:
        completion_string (str): Raw completion string that may contain markdown formatting
        
    Returns:
        str: Cleaned Python code without markdown formatting
        
    Note:
        This function specifically looks for ```python code blocks and extracts
        the code content, removing the markdown syntax.
    """
    if "```python" in completion_string:
        completion_string = completion_string[completion_string.find("```python") + len("```python"):]
        completion_string = completion_string[:completion_string.find("```")]
    else:
        print("Error: No code block found")
    return completion_string

def fetch_completion(data_entry, model, times=5):
    """
    Generate multiple code completions for a single problem using OpenAI's API.
    
    This function is the core of the programmer agent, responsible for:
    1. Checking if the problem needs reproduction (skip if already processed)
    2. Constructing the prompt with few-shot examples and clarity instructions
    3. Making multiple API calls to generate diverse completions
    4. Handling errors and retries with exponential backoff
    5. Preprocessing the generated code to extract clean Python code
    
    Args:
        data_entry (dict): Problem data containing prompt, entry_point, and optional clarity_prompt
        model (str): OpenAI model name to use for generation
        times (int): Number of completions to generate (default: 5)
        
    Returns:
        dict: Updated data_entry with 'completion_list' field containing generated code
        
    Note:
        - Uses global construct_few_shot_prompt for consistent prompting
        - Implements retry mechanism with 10-second delays on API failures
        - Supports HumanEvalComm clarity prompts for ambiguous problems
    """
    global construct_few_shot_prompt
    
    # Skip if this problem doesn't need reproduction (already processed successfully)
    if "need_reproduce" in data_entry.keys() and not data_entry["need_reproduce"]:
        return data_entry

    prompt = data_entry["prompt"]
    # Add clarity prompt if available (for HumanEvalComm ambiguous problems)
    clarity = "" if "clarity_prompt" not in data_entry else data_entry["clarity_prompt"]
    
    # Construct the full prompt with few-shot examples and clarity instructions
    text = f"""
{construct_few_shot_prompt}
{clarity}
**Input Code Snippet**:
```python
{prompt}
```
## Completion 3:
"""
    completions_code = []
    
    # Generate multiple completions for diversity
    for _ in range(times):
        while True:
            try:
                # Make API call to OpenAI
                completions = openai.ChatCompletion.create(
                    model=model,
                    stream=False,
                    messages=[
                        {"role": "system", "content": "You are a software programmer."},
                        {"role": "user", "content": text},
                    ],
                    request_timeout=100,
                )
                completion = completions.choices[0]["message"]["content"]
                completion = preprocess_data(completion)
            except Exception as e:
                print(e)
                time.sleep(10)  # Wait before retry
                completion = ""
            if completion:
                break
        completions_code.append(completion)

    # Store all generated completions in the data entry
    data_entry["completion_list"] = completions_code
    return data_entry

def programmer_main(model, language, new_dataset, api_key, task_id):
    """
    Main orchestrator function for the programmer agent.
    
    This function coordinates the code generation process for an entire dataset:
    1. Sets up OpenAI API key
    2. Uses ThreadPoolExecutor for concurrent processing (limited to 1 worker for stability)
    3. Processes each problem in the dataset through fetch_completion
    4. Updates the dataset with generated completions
    5. Saves the results to a JSON file
    
    Args:
        model (str): Model name identifier for file naming
        language (str): Programming language (e.g., 'python')
        new_dataset (list): List of problem dictionaries to process
        api_key (str): OpenAI API key for authentication
        task_id (str): Task identifier for file naming
        
    Returns:
        list: Updated dataset with completion_list fields populated
        
    Note:
        - Uses max_workers=1 to avoid API rate limiting and ensure sequential processing
        - Creates dataset directory if it doesn't exist
        - Saves intermediate results to prevent data loss
    """
    openai.api_key = api_key  # Set the API key here

    # Process all problems concurrently (but with limited workers for API stability)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_to_entry = {
            executor.submit(fetch_completion, copy.deepcopy(entry), "gpt-3.5-turbo-1106"): entry
            for entry in tqdm(new_dataset)
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_entry)):
            entry = future_to_entry[future]
            try:
                updated_entry = future.result()
                idx = new_dataset.index(entry)
                new_dataset[idx] = updated_entry
            except Exception as e:
                print(repr(e))
    
    # Create folder if it doesn't already exist
    os.makedirs(f"./dataset", exist_ok=True)
    
    # Save the processed dataset to a JSON file
    with open(f"./dataset/{model}_{language}_{task_id}.json", "w") as f:
        json.dump(new_dataset, f, indent=4)
    
    return new_dataset

def call_fetch_completion_helper(dataset, model, lg):
    """
    Helper function for bug fixing iterations in the AgentCoder framework.
    
    This function is called during the iterative improvement process where:
    1. The executor identifies problems that need fixing
    2. This function regenerates completions for those problems
    3. The process continues until satisfactory results are achieved
    
    Args:
        dataset (list): Dataset with problems that need completion regeneration
        model (str): Model name identifier
        lg (str): Language identifier
        
    Returns:
        list: Updated dataset with new completions
        
    Note:
        - This is part of the iterative improvement cycle in AgentCoder
        - Only processes problems that still need reproduction
        - Uses the same concurrent processing pattern as programmer_main
    """
    print("Fixing bug...")
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_to_entry = {
            executor.submit(fetch_completion, copy.deepcopy(entry), "gpt-3.5-turbo-1106"): entry
            for entry in tqdm(dataset)
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_entry)):
            entry = future_to_entry[future]
            try:
                updated_entry = future.result()
                idx = dataset.index(entry)
                dataset[idx] = updated_entry
            except Exception as e:
                print(repr(e))
    return dataset