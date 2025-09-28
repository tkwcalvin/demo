"""
CodeLlama Model Interface
========================

This module provides interfaces for interacting with CodeLlama models,
including instruction-tuned versions for natural language to code tasks.

Key functions:
- get_completion_codellama_instruct_nl_to_pl: Natural language to code generation
- get_completion_codellama_instruct_pl_to_nl: Code to natural language (docstring generation)

Features:
- Support for instruction-tuned CodeLlama models
- Custom prompt formatting with CodeLlama-specific tokens
- Both NL-to-code and code-to-NL tasks
"""

from config import B_INST_CLLAMA, E_INST_CLLAMA, B_SYS_CLLAMA, E_SYS_CLLAMA
from generate_text import generate_text



def get_completion_codellama_instruct_nl_to_pl(
    prompt, user_input, model, tokenizer, args
):
    """
    Generate code from natural language using CodeLlama-Instruct models.
    
    This function formats the input according to CodeLlama's instruction format
    and generates code responses. It supports both simple user input and
    structured prompt templates.
    
    Args:
        prompt (str): Prompt template or empty string for simple formatting
        user_input (str): The user's natural language input
        model: Loaded CodeLlama model
        tokenizer: CodeLlama tokenizer
        args: Command line arguments for generation parameters
    
    Returns:
        str: Generated code response from the model
    """
    formatted_prompt = ""
    
    if prompt == '':
        # Simple case: use user input directly
        formatted_prompt = user_input 
    else:
        # Complex case: format according to CodeLlama instruction template
        messages = [{"role": "user", "content": user_input}]
    
        for msg in messages:
            if msg["role"] == "user":
                content = msg["content"].strip()
                formatted_prompt += tokenizer.bos_token + f"{B_INST_CLLAMA} " + content + f" {E_INST_CLLAMA} "
            elif msg["role"] == "assistant":
                formatted_prompt += " " + msg["content"].strip() + " " + tokenizer.eos_token
            elif msg["role"] == "system":
                # Note: System prompts don't work well with CodeLlama-Instruct
                formatted_prompt += f"{B_SYS_CLLAMA}" + msg["content"] + f"{E_SYS_CLLAMA}"

    # Debug output
    print('\nformatted_prompt:\n', formatted_prompt)
    
    # Generate response using the formatted prompt
    output = generate_text(model, tokenizer, formatted_prompt, args)
    completion = output[0]["generated_text"]

    print('\ncompletion:\n', completion)
    return completion



def get_completion_codellama_instruct_pl_to_nl(prompt, user_input, model, tokenizer, args):
    """
    Generate natural language docstring from code using CodeLlama-Instruct models.
    
    This function takes code as input and generates appropriate docstrings.
    It supports different datasets with different docstring formats.
    
    Args:
        prompt (list): List of message dictionaries for in-context learning
        user_input (str): The code input
        model: Loaded CodeLlama model
        tokenizer: CodeLlama tokenizer
        args: Command line arguments containing input_path for dataset-specific formatting
    
    Returns:
        str: Generated docstring with proper formatting and indentation
    
    Raises:
        ValueError: If the input dataset format is not supported
    """
    # Combine prompt template with user input
    messages = prompt + [{"role": "user", "content": user_input}]
    formatted_prompt = ""
    
    for msg in messages:
        if msg["role"] == "user":
            # Add dataset-specific instructions for docstring generation
            if args.input_path.endswith("EvalPlus-Mini-v0.1.6_reformatted.jsonl"):
                content = (
                    msg["content"]
                    + "\n\nWhat should be the docstring of the above function? Please only write down the docstring with some examples."
                )
            elif args.input_path.endswith("MBPP-S_test_reformatted.jsonl"):
                content = (
                    msg["content"]
                    + "\n\nWhat should be the docstring of the above function? Please write down the docstring only in words without any examples!"
                )
            else:
                raise ValueError(f"Input file {args.input_path} not supported")
            
            formatted_prompt += tokenizer.bos_token + "[INST] " + content + " [/INST] "
        elif msg["role"] == "assistant":
            formatted_prompt += " " + msg["content"].strip() + " " + tokenizer.eos_token
    
    # Generate response
    output = generate_text(model, tokenizer, formatted_prompt, args)
    completion = output[0]["generated_text"]
    
    # ============================================================================
    # POST-PROCESSING
    # ============================================================================
    # Ensure proper indentation (4 spaces)
    completion_lines = completion.split("\n")
    for idx, line in enumerate(completion_lines):
        if not line.startswith("    "):
            completion_lines[idx] = "    " + line.lstrip()
    completion = "\n".join(completion_lines)
    
    # Add docstring closing quotes if missing
    if completion.startswith('    """') and not completion.endswith('"""'):
        completion = completion + '"""'
    elif completion.startswith("    '''") and not completion.endswith("'''"):
        completion = completion + "'''"
    
    print(completion)
    return completion