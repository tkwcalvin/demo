
"""
Text Generation Module
=====================

This module provides the core text generation functionality for open source models.
It handles tokenization, generation parameters, and post-processing of generated text.

Key functions:
- generate_text: Main function for generating text from open source models

Features:
- Support for various generation parameters (temperature, top-k, top-p, beam search)
- Automatic handling of empty prompts
- Batch processing for multiple sequences
- Post-processing to extract only the generated portion
"""

def generate_text(model, tokenizer, prompt_text, args, eos_token_id=None):
    """
    Generate text using an open source language model.
    
    Args:
        model: The loaded language model (from transformers)
        tokenizer: The corresponding tokenizer
        prompt_text (str): The input prompt text
        args: Command line arguments containing generation parameters
        eos_token_id (int, optional): End-of-sequence token ID. If None, uses tokenizer's EOS token
    
    Returns:
        list: List of dictionaries, each containing a "generated_text" field
    """
    # Set EOS token ID if not provided
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    
    # ============================================================================
    # INPUT TOKENIZATION
    # ============================================================================
    model_inputs = tokenizer(prompt_text, padding=False, add_special_tokens=False, return_tensors="pt")
    model_inputs["prompt_text"] = prompt_text
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask", None)
    
    # Handle empty prompts gracefully
    if input_ids.shape[1] == 0:
        input_ids = None
        attention_mask = None
        in_b = 1
    else:
        in_b = input_ids.shape[0]
    
    # ============================================================================
    # GENERATION PARAMETERS SETUP
    # ============================================================================
    # Determine maximum generation length
    if args.gen_length is None:
        max_length = args.seq_length
    else:
        max_length = input_ids.shape[1] + args.gen_length
    
    print("Generating text...")
    
    # ============================================================================
    # TEXT GENERATION
    # ============================================================================
    generated_sequence = model.generate(
        input_ids=input_ids.to(model.device),
        attention_mask=attention_mask.to(model.device),
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_length=max_length,
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_token_id,
    )
    
    # ============================================================================
    # POST-PROCESSING
    # ============================================================================
    # Reshape output for batch processing
    out_b = generated_sequence.shape[0]
    generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])
    generated_sequence = generated_sequence[0].cpu().numpy().tolist()
    
    # Decode generated sequences and extract only the new content
    records = []
    print("Decoding text...")
    for sequence in generated_sequence:
        text = tokenizer.decode(sequence, skip_special_tokens=True)
        prompt_length = len(tokenizer.decode(input_ids[0], skip_special_tokens=True))
        all_text = text[prompt_length:]  # Extract only the generated portion
        record = {"generated_text": all_text}
        records.append(record)
    
    return records