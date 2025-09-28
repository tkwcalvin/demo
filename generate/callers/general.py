"""
General Model Interface
======================

This module provides general interfaces for interacting with various foundation models
including CodeLlama base models, StarCoder, and other code generation models.

Key functions:
- get_completion_codellama: Standard CodeLlama completion
- get_completion_codellama_fim: CodeLlama with Fill-in-the-Middle (FIM)
- get_completion_starcoder: StarCoder completion
- get_completion_starcoder_fim: StarCoder with Fill-in-the-Middle (FIM)

Features:
- Support for foundation models (non-instruction-tuned)
- Fill-in-the-Middle (FIM) capabilities for code completion
- One-shot prompting with examples
- Post-processing for code extraction
"""

from generate_text import generate_text

def get_completion_codellama(
    prompt, user_input, model, tokenizer, args
):
    """
    Generate code completion using CodeLlama foundation model with one-shot prompting.
    
    This function uses one-shot prompting where the prompt contains an example,
    followed by the user's input. It extracts only the function body from the response.
    
    Args:
        prompt (str): One-shot example prompt (ONE_SHOT_HUMANEVAL or ONE_SHOT_MBPP)
        user_input (str): The user's function signature and problem description
        model: Loaded CodeLlama model
        tokenizer: CodeLlama tokenizer
        args: Command line arguments for generation parameters
    
    Returns:
        str: Extracted function body with proper indentation
    """
    # Combine one-shot prompt with user input
    user_input = prompt + user_input
    output = generate_text(model, tokenizer, user_input, args)
    completion = output[0]["generated_text"]
    
    # ============================================================================
    # POST-PROCESSING: EXTRACT FUNCTION BODY
    # ============================================================================
    # Extract only lines that start with 4 spaces (function body)
    processed_completion = ""
    completion_lines = completion.split("\n")
    for line in completion_lines:
        if line.startswith("    "):
            processed_completion += line + "\n"
        else:
            break  # Stop at first non-indented line
    
    print(processed_completion)
    return processed_completion


def get_completion_codellama_fim(
    prompt, function_signature, function_body, model, tokenizer, args
):
    """
    Generate docstring completion using CodeLlama with Fill-in-the-Middle (FIM).
    
    This function uses FIM prompting to generate docstrings for function signatures.
    The model is given the function signature and body, and asked to fill in the docstring.
    
    Args:
        prompt (str): One-shot example prompt (ONE_SHOT_HUMANEVAL or ONE_SHOT_MBPP)
        function_signature (str): The function signature
        function_body (str): The function body
        model: Loaded CodeLlama model
        tokenizer: CodeLlama tokenizer
        args: Command line arguments for generation parameters
    
    Returns:
        str: Generated docstring with proper formatting
    """
    function_signature = prompt + function_signature
    fim_prompt = ""
    
    # Construct FIM prompt with PRE, SUF, and MID tokens
    fim_prompt += (
        " <PRE>" + function_signature + "    \"\"\"\n    " + 
        " <SUF>" + "    \"\"\"\n" + function_body + " <MID>"
    )
    
    output = generate_text(model, tokenizer, fim_prompt, args)
    completion = output[0]["generated_text"]
    
    # Format the completion with proper docstring structure
    completion = "    \"\"\"\n    " + completion + "    \"\"\"\n"
    
    print(completion)
    return completion


def get_completion_starcoder(
    prompt, user_input, model, tokenizer, args
):
    """
    Generate code completion using StarCoder foundation model with one-shot prompting.
    
    This function uses one-shot prompting similar to CodeLlama but with StarCoder-specific
    processing. It extracts the function body from the response.
    
    Args:
        prompt (str): One-shot example prompt (ONE_SHOT_HUMANEVAL or ONE_SHOT_MBPP)
        user_input (str): The user's function signature and problem description
        model: Loaded StarCoder model
        tokenizer: StarCoder tokenizer
        args: Command line arguments for generation parameters
    
    Returns:
        str: Extracted function body with proper indentation
    """
    user_input = prompt + user_input
    output = generate_text(model, tokenizer, user_input.strip(), args)
    completion = output[0]["generated_text"]
    
    # ============================================================================
    # POST-PROCESSING: EXTRACT FUNCTION BODY
    # ============================================================================
    # Extract function body lines (indented with 4 spaces)
    processed_completion = ""
    completion_lines = completion.split("\n")
    start = False
    for line in completion_lines:
        if line.startswith("    "):
            start = True
            processed_completion += line + "\n"
        else:
            if start:
                break  # Stop after function body ends
    
    print(processed_completion)
    return processed_completion


def get_completion_starcoder_fim(
    prompt, function_signature, function_body, model, tokenizer, args
):
    """
    Generate docstring completion using StarCoder with Fill-in-the-Middle (FIM).
    
    This function uses StarCoder's FIM prompting format to generate docstrings
    for function signatures. It uses StarCoder-specific FIM tokens.
    
    Args:
        prompt (str): One-shot example prompt (ONE_SHOT_HUMANEVAL or ONE_SHOT_MBPP)
        function_signature (str): The function signature
        function_body (str): The function body
        model: Loaded StarCoder model
        tokenizer: StarCoder tokenizer
        args: Command line arguments for generation parameters
    
    Returns:
        str: Generated docstring with proper formatting
    """
    function_signature = prompt + function_signature
    fim_prompt = ""
    
    # Construct StarCoder FIM prompt with specific tokens
    fim_prompt += (
        "<fim_prefix>"
        + function_signature
        + "    \"\"\""
        + "<fim_suffix>"
        + "\"\"\"\n"
        + function_body
        + "<fim_middle>"
    )
    
    output = generate_text(model, tokenizer, fim_prompt, args)
    completion = output[0]["generated_text"]
    
    # Format the completion with proper docstring structure
    completion = "    \"\"\"" + completion + "\"\"\"\n"
    
    print(completion)
    return completion