"""
StarChat Model Interface
=======================

This module provides interfaces for interacting with StarChat models,
which are instruction-tuned versions of StarCoder for conversational tasks.

Key functions:
- get_completion_starchat_nl_to_pl: Natural language to code generation
- get_completion_starchat_pl_to_nl: Code to natural language (docstring generation)

Features:
- Support for StarChat instruction-tuned models
- Dialogue template handling
- Post-processing for code and docstring extraction
- Support for both NL-to-code and code-to-NL tasks

Note: This module requires the DialogueTemplate class from the StarChat repository.
"""

from generate_text import generate_text




def get_completion_starchat_nl_to_pl(prompt, user_input, model, tokenizer, args):
    """
    Generate code from natural language using StarChat models.
    
    This function uses StarChat's dialogue template system to format conversations
    and generate code responses. It includes post-processing to extract clean code.
    
    Args:
        prompt (list): List of message dictionaries for in-context learning
        user_input (str): The user's natural language input
        model: Loaded StarChat model
        tokenizer: StarChat tokenizer
        args: Command line arguments containing model_name_or_path
    
    Returns:
        str: Generated and processed code with proper indentation
    """
    # Combine prompt template with user input
    messages = prompt + [{"role": "user", "content": user_input}]
    
    # Load dialogue template for StarChat formatting
    try:
        dialogue_template = DialogueTemplate.from_pretrained(args.model_name_or_path)
    except Exception:
        print("No dialogue template found in model repo. Defaulting to the `no_system` template.")
        dialogue_template = get_dialogue_template("no_system")
    
    dialogue_template.messages = messages
    formatted_prompt = dialogue_template.get_inference_prompt_nl_to_pl()
    
    # Generate response using StarChat's end token
    output = generate_text(
        model,
        tokenizer,
        formatted_prompt,
        args,
        eos_token_id=tokenizer.convert_tokens_to_ids(dialogue_template.end_token),
    )
    completion = output[0]["generated_text"]
    
    # ============================================================================
    # POST-PROCESSING: EXTRACT AND CLEAN CODE
    # ============================================================================
    # Extract lines with proper indentation (function body)
    completion_lines = completion.split("\n")
    processed_completion = ""
    for line in completion_lines:
        if line.startswith("    "):
            processed_completion += line + "\n"
    
    # Remove extra docstrings that might be generated
    # Find all occurrences of triple quotes (docstring markers)
    res = [i for i in range(len(processed_completion)) if processed_completion.startswith('"""', i)]
    if not res:
        res = [i for i in range(len(processed_completion)) if processed_completion.startswith("'''", i)]
    
    # If multiple docstrings found, remove everything before the second one
    if res and len(res) > 1:
        try:
            end_position = res[1] + 3
            processed_completion = processed_completion[end_position:]
            if processed_completion.startswith("\n"):
                processed_completion = processed_completion[1:]
        except IndexError:
            pass

    print(processed_completion)
    return processed_completion



def get_completion_starchat_pl_to_nl(prompt, user_input, model, tokenizer, args):
    """
    Generate natural language docstring from code using StarChat models.
    
    This function takes code as input and generates appropriate docstrings
    using StarChat's dialogue template system.
    
    Args:
        prompt (list): List of message dictionaries for in-context learning
        user_input (str): The code input
        model: Loaded StarChat model
        tokenizer: StarChat tokenizer
        args: Command line arguments containing model_name_or_path
    
    Returns:
        str: Generated docstring with proper formatting and indentation
    """
    # Combine prompt template with user input
    messages = prompt + [{"role": "user", "content": user_input}]
    
    # Load dialogue template for StarChat formatting
    try:
        dialogue_template = DialogueTemplate.from_pretrained(args.model_name_or_path)
    except Exception:
        print("No dialogue template found in model repo. Defaulting to the `no_system` template.")
        dialogue_template = get_dialogue_template("no_system")
    
    dialogue_template.messages = messages
    formatted_prompt = dialogue_template.get_inference_prompt_pl_to_nl()
    
    # Generate response using StarChat's end token
    output = generate_text(
        model,
        tokenizer,
        formatted_prompt,
        args,
        eos_token_id=tokenizer.convert_tokens_to_ids(dialogue_template.end_token),
    )
    completion = output[0]["generated_text"]
    
    # ============================================================================
    # POST-PROCESSING: EXTRACT AND CLEAN DOCSTRING
    # ============================================================================
    # Remove language identifiers and extract content from code blocks
    completion = completion.replace("python", "")
    completion_parts = completion.split("```")
    if len(completion_parts) > 1:
        completion = completion_parts[1]
    
    # Extract content from docstring markers
    completion_parts = completion.split('"""')
    if len(completion_parts) > 1:
        completion = completion_parts[1]
    completion_parts = completion.split("'''")
    if len(completion_parts) > 1:
        completion = completion_parts[1]

    # Ensure proper docstring formatting
    if not completion.startswith('"""'):
        completion = '"""' + completion
    if not completion.endswith('"""'):
        completion = completion + '\n"""'
    
    # Ensure proper indentation (4 spaces)
    completion_lines = completion.split("\n")
    for idx, line in enumerate(completion_lines):
        if not line.startswith("    "):
            completion_lines[idx] = "    " + line
    completion = "\n".join(completion_lines)
    
    print(completion)
    return completion