"""
Process Management Module
========================

This module handles the main experiment processing logic, including dataset loading,
experiment execution, and result logging for HumanEval code generation experiments.

Key functions:
- HumanEval_experiment: Main experiment runner for HumanEval datasets
- test_codellama: Testing function for CodeLlama models
- test_starcoder: Testing function for StarCoder models

Features:
- Dataset loading and filtering
- Phase-based evaluation support
- Caching and resumption of experiments
- Comprehensive logging and result storage
"""

from utils import string_to_int, get_ith_element
from prompt import load_prompt_from_config, create_prompt
from handle import description_2_code_one_round, description_2_code_multi_rounds
from config import ORIGINAL_PROMPT_START_0
import os
import json
import torch
import time




def HumanEval_experiment(dataset, dataset_loc, option, model, topn, temperature, args, open_source_model, tokenizer):
    """
    Main experiment runner for HumanEval code generation experiments.
    
    This function orchestrates the entire experiment process including:
    - Dataset loading and filtering
    - Caching and resumption support
    - Multi-prompt processing (for HumanEvalComm)
    - Result logging and storage
    
    Args:
        dataset (str): Dataset name (HumanEval, HumanEvalComm, etc.)
        dataset_loc (str): Path to the dataset file
        option (str): Experiment option (original, randRemoveX, etc.)
        model (str): Model identifier
        topn (int): Number of candidate responses to generate
        temperature (float): Sampling temperature
        args: Command line arguments
        open_source_model: Loaded model (if applicable)
        tokenizer: Model tokenizer (if applicable)
    """
    # ============================================================================
    # EXPERIMENT SETUP AND LOGGING
    # ============================================================================
    remove_percentage = 0
    
    # Determine log file name based on experiment configuration
    if option == 'original':
        log_file = './log/dataset_%s_model_%s_topn_%s_temperature_%s.log_%s' % \
                   (dataset, model, topn, temperature, str(args.log_phase_input))
    else:
        log_file = './log/%s_dataset_%s_model_%s_topn_%s_temperature_%s.log_%s' % \
                   (option, dataset, model, topn, temperature, str(args.log_phase_input))
        remove_percentage = string_to_int(get_ith_element(option, 1))
    
    # Set up print file for detailed logging
    print_file_str = './log/print' + log_file[5:]
    global print_file
    print_file = open(print_file_str, 'a')  # Append to existing file if it exists
    
    # ============================================================================
    # DATASET LOADING AND FILTERING
    # ============================================================================
    problem_list = []
    line_cnt = 0

    # Load problems from dataset with optional filtering
    with open(dataset_loc, 'r') as f:
        for line in f.readlines():
            # Apply index filtering if specified
            if args.min_problem_idx < 0 or line_cnt >= args.min_problem_idx:
                problem_list.append(json.loads(line))
            line_cnt += 1
            # Stop if maximum number of problems is reached
            if args.max_num_problems >= 0 and line_cnt >= args.max_num_problems:
                break
    # ============================================================================
    # CACHING AND RESUMPTION SUPPORT
    # ============================================================================
    # Load cached results to support experiment resumption
    cached_names = set()
    cached_responses = {}
    cached_answers = {}
    cached_qqs = {}
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                content = json.loads(line)
                key = content['name']+'_'+content['prompt_type']
                cached_names.add(key)
                cached_responses[key] = content['response']
                cached_answers[key] = content['answer']
                cached_qqs[key] = content['question_quality']

    # Load phase 1 prompt configuration
    config_phase1_prompt = load_prompt_from_config(phase = 1)
    
    # ============================================================================
    # MAIN EXPERIMENT LOOP
    # ============================================================================
    response_list = []
    for problem in problem_list:
        print('----------------------problem name: %s--------------------------------' % (problem['name']), flush=True)
        print('using %s to generate response' % (model), flush=True)
        
        # Determine which prompt fields to process based on dataset type
        if dataset == "HumanEvalComm":
            # HumanEvalComm has multiple prompt variants for different modification types
            input_prompt_fields = ['prompt1a','prompt1c','prompt1p','prompt2ac','prompt2ap','prompt2cp','prompt3acp']
        else:  
            # Standard HumanEval has only the original prompt
            input_prompt_fields = ['prompt']
        
        # Process each prompt variant for the current problem
        for input_prompt in input_prompt_fields:
            if input_prompt not in problem:
                continue
                
            key = problem['name'] + '_' + input_prompt
            
            # Skip if already processed (caching mechanism)
            if args.log_phase_input == args.log_phase_output and key in cached_names:
                continue
                
            print("********************************************************************", file=print_file)
            print("****** new problem (name="+problem['name']+" input_prompt="+input_prompt+") ******", file=print_file)
            print("********************************************************************", file=print_file)
            
            description = problem[input_prompt]
            try:
                # Create the prompt based on experiment option
                prompt = create_prompt(description, option, remove_percentage)
                
                if option.startswith('randRemove'):
                    # Legacy single-round evaluation for randRemove experiments
                    response_list, code_list, qq_list = description_2_code_one_round(prompt, model, topn, temperature, args, open_source_model, tokenizer)
                else:
                    # Multi-round evaluation with clarifying questions
                    original_prompt = problem['prompt']
                    entry_point = problem['entry_point']
                    task_id = problem['name']
                    
                    # Determine if this is a modified prompt (triggers question mode)
                    prompt_modified = False if input_prompt == 'prompt' else True
                    prompt_start = ORIGINAL_PROMPT_START_0 if input_prompt == 'prompt' else config_phase1_prompt
                    
                    response_list, code_list, qq_list, ans_list = description_2_code_multi_rounds(
                        prompt_modified, task_id, entry_point, prompt_start, description, 
                        original_prompt, model, topn, temperature, args, open_source_model, 
                        tokenizer, cached_responses.get(key, ''), cached_qqs.get(key, 0), 
                        cached_answers.get(key, '')
                    )
            except Exception as e:
                print('%s---------%s' % (problem['name'], e), flush=True)
                continue
            # ============================================================================
            # RESULT LOGGING AND STORAGE
            # ============================================================================
            for i in range(len(response_list)):
                if args.log_phase_output >= 1:
                    # Phase-based logging format (for multi-phase experiments)
                    res = {
                        'key': key,
                        'name': problem['name'],
                        'prompt_type': input_prompt,
                        'index': i,
                        'response': response_list[i],
                        'answer': ans_list[i] if i < len(ans_list) else '',
                        'question_quality': qq_list[i] if i < len(qq_list) else '0',
                        'code': code_list[i] if i < len(code_list) else '',
                    }
                    print('response %s is writting into file' % (i), flush=True)
                    json_str = json.dumps(res)

                    # Generate output log file name for specific phase
                    last_index = log_file.rfind('.log_')
                    log_file_output = log_file[:last_index] + '.log_' + str(args.log_phase_output)
                    
                    with open(log_file_output, 'a') as f:
                        f.write(json_str + '\n')
                else:
                    # Standard logging format (for complete experiments)
                    res = {
                        'name': problem['name'],
                        'index': i,
                        'response': response_list[i],
                        'original_prompt': description,
                        'modified_prompt': prompt,
                        'prompt_type': input_prompt,
                        'code': code_list[i],
                        'question_quality': qq_list[i],
                        'answer': ans_list[i],
                    }
                    print('response %s is writting into file' % (i), flush=True)
                    json_str = json.dumps(res)
                    with open(log_file, 'a') as f:
                        f.write(json_str + '\n')
            
            print('%s finish!' % (problem['name']), flush=True)
            # Debug option: uncomment to stop after first prompt
            # break
    
    print('Done!', flush=True)







# ============================================================================
# MODEL TESTING FUNCTIONS
# ============================================================================
# These functions provide simple testing capabilities for loaded models

def test_starcoder(tokenizer, model, user_input, max_length):
    """
    Test function for StarCoder models.
    
    Args:
        tokenizer: StarCoder tokenizer
        model: Loaded StarCoder model
        user_input (str): Input text to generate from
        max_length (int): Maximum generation length
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer.encode(
        input_ids=user_input,
        # compute a + b 
        #"def print_hello_world():", 
        return_tensors="pt",
        ).to(device)
    print('device=', device)
    outputs = model.generate(
        inputs,
        max_length=max_length,
        )
    print('!!!!!!!!!!')
    print(tokenizer.decode(outputs[0]))
    print('!!!!!!!!!!')




def test_codellama(tokenizer, model, user_input, max_length):
    """
    Test function for CodeLlama models with timing.
    
    Args:
        tokenizer: CodeLlama tokenizer
        model: Loaded CodeLlama model
        user_input (str): Input text to generate from
        max_length (int): Maximum number of new tokens to generate
    """
    timea = time.time()
    input_ids = tokenizer(user_input, return_tensors="pt")["input_ids"].to(model.device)
    generated_ids = model.generate(input_ids, max_new_tokens=max_length)
    filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
    
    print('!!!!!!!!!!')
    print(filling)
    print('!!!!!!!!!!')
    print("timea = time.time()", -timea + time.time())