# This file contains code copied and modified from the following repository:
# Repository: https://github.com/huangd1999/AgentCoder/tree/main
# Original Author: Dong Huang, Jie M.Zhang, Michael Luck, Qingwen Bu, Yuhao Qing, Heming Cui
# License: MIT

"""
AgentFramework Designer Module
==============================

This module implements the designer agent component of the AgentCoder framework.
The designer agent is responsible for generating comprehensive test cases for code problems.
It analyzes the problem description and generates multiple test cases to validate the correctness of generated code.

Key Features:
- Generates multiple test cases for each problem (default: 10)
- Uses few-shot prompting with examples of good test case design
- Implements concurrent processing for efficiency
- Handles error recovery and retry mechanisms
- Extracts clean test code from markdown formatting

Main Functions:
- fetch_completion(): Generates test cases for a single problem
- designer_main(): Orchestrates the test case generation process for a dataset
- call_fetch_test_completion_helper(): Helper function for iterative improvement
"""

import json
from tqdm import tqdm
import copy
import openai
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import time

# Load the few-shot prompt template for test case generation
prompt_path = "./prompts/test_designer_humaneval_prompt_update.txt"
with open(prompt_path, "r") as f:
    construct_few_shot_prompt = f.read()

def preprocess_data(test_case_string):
    """
    Extract Python test code from markdown code blocks in the test case string.
    
    Args:
        test_case_string (str): Raw test case string that may contain markdown formatting
        
    Returns:
        str: Cleaned Python test code without markdown formatting
        
    Note:
        This function specifically looks for ```python code blocks and extracts
        the test code content, removing the markdown syntax.
    """
    if "```python" in test_case_string:
        test_case_string = test_case_string[test_case_string.find("```python") + len("```python"):]
        test_case_string = test_case_string[:test_case_string.find("```")]
    return test_case_string

def fetch_completion(data_entry, model, times=10):
    """
    Generate multiple test cases for a single problem using OpenAI's API.
    
    This function is the core of the designer agent, responsible for:
    1. Checking if the problem needs reproduction (skip if already processed)
    2. Constructing the prompt with few-shot examples for test case design
    3. Making multiple API calls to generate diverse test cases
    4. Handling errors and retries with exponential backoff
    5. Preprocessing the generated test code to extract clean Python code
    
    Args:
        data_entry (dict): Problem data containing prompt and entry_point
        model (str): OpenAI model name to use for generation
        times (int): Number of test cases to generate (default: 10)
        
    Returns:
        dict: Updated data_entry with 'test_case_list' field containing generated test cases
        
    Note:
        - Uses global construct_few_shot_prompt for consistent test case prompting
        - Implements retry mechanism with 20-second delays on API failures
        - Generates more test cases than code completions for comprehensive testing
    """
    global construct_few_shot_prompt
    
    # Skip if this problem doesn't need reproduction (already processed successfully)
    if "need_reproduce" in data_entry.keys() and not data_entry["need_reproduce"]:
        return data_entry
    
    prompt = data_entry["prompt"]
    entry_point = data_entry["entry_point"]
    
    # Construct the full prompt with few-shot examples for test case generation
    text = f"""
{construct_few_shot_prompt}

**Input Code Snippet**:
```python
{prompt}
```
"""
    test_case_list = []
    
    # Generate multiple test cases for comprehensive testing
    for _ in range(times):
        while True:
            try:
                # Make API call to OpenAI
                completions = openai.ChatCompletion.create(
                    model=model,
                    stream=False,
                    messages=[
                        {"role": "system", "content": "You are a code developer assistant."},
                        {"role": "user", "content": text},
                    ],
                    request_timeout=100,
                )
                test_case = completions.choices[0]["message"]["content"]
                test_case = preprocess_data(test_case)
            except Exception as e:
                time.sleep(20)  # Wait before retry (longer than programmer)
                print(e)
                test_case = ""
            if test_case:
                break
        test_case_list.append(test_case)
    
    # Store all generated test cases in the data entry
    data_entry["test_case_list"] = test_case_list
    return data_entry

def designer_main(model, language, new_dataset, api_key, task_id):
    """
    Main orchestrator function for the designer agent.
    
    This function coordinates the test case generation process for an entire dataset:
    1. Sets up OpenAI API key
    2. Uses ThreadPoolExecutor for concurrent processing (limited to 1 worker for stability)
    3. Processes each problem in the dataset through fetch_completion
    4. Updates the dataset with generated test cases
    5. Saves the results to a JSON file
    
    Args:
        model (str): Model name identifier for file naming
        language (str): Programming language (e.g., 'python')
        new_dataset (list): List of problem dictionaries to process
        api_key (str): OpenAI API key for authentication
        task_id (str): Task identifier for file naming
        
    Returns:
        list: Updated dataset with test_case_list fields populated
        
    Note:
        - Uses max_workers=1 to avoid API rate limiting and ensure sequential processing
        - Saves intermediate results to prevent data loss
        - Follows the same pattern as programmer_main for consistency
    """
    openai.api_key = api_key
    
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
    
    # Save the processed dataset to a JSON file
    with open(f"./dataset/{model}_{language}_{task_id}.json", "w") as f:
        json.dump(new_dataset, f, indent=4)
    
    return new_dataset

def call_fetch_test_completion_helper(dataset, model, lg):
    """
    Helper function for bug fixing iterations in the AgentCoder framework.
    
    This function is called during the iterative improvement process where:
    1. The executor identifies problems that need better test cases
    2. This function regenerates test cases for those problems
    3. The process continues until satisfactory test coverage is achieved
    
    Args:
        dataset (list): Dataset with problems that need test case regeneration
        model (str): Model name identifier
        lg (str): Language identifier
        
    Returns:
        list: Updated dataset with new test cases
        
    Note:
        - This is part of the iterative improvement cycle in AgentCoder
        - Only processes problems that still need reproduction
        - Uses the same concurrent processing pattern as designer_main
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
