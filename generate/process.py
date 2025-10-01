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
from model.experiment import *
from model.log import *

from utils import string_to_int, get_ith_element
from prompt import load_prompt_from_config, create_prompt
from handle import description_2_code_one_round, description_2_code_multi_rounds
from config import *
import os
import json
import torch
import time
from handle import Handler

from model.cache import Cache

class Experiment:
    def __init__(self, model, tokenizer, exp_config: CoreExpConfig, log_config: LogConfig):
        
        self.model = model
        self.tokenizer = tokenizer
        self.exp_config = exp_config
        self.log_config = log_config
        self.problem_list = []
        self.cache = None
        self.prepare_experiment()
    
    def prepare_experiment(self):
        # ============================================================================
        # EXPERIMENT SETUP AND LOGGING
        # ============================================================================
        self.remove_percentage = self.init_file_log_and_print()
        # ============================================================================
        # DATASET LOADING AND FILTERING
        # ============================================================================
        self.problem_list = self.load_dataset()
        # ============================================================================
        # CACHING AND RESUMPTION SUPPORT
        # ============================================================================
        # Load cached results to support experiment resumption
        self.cache = self.load_cached_results()

    def init_file_log_and_print(self):
        remove_percentage = 0
        # Determine log file name based on experiment configuration
        if self.exp_config.option == 'original':
            self.log_config.log_file = './log/dataset_%s_model_%s_topn_%s_temperature_%s.log_%s' % \
                    (self.exp_config.dataset, self.exp_config.model, self.exp_config.topn, self.exp_config.temperature, str(self.log_config.log_phase_input))
        else:
            self.log_config.log_file = './log/%s_dataset_%s_model_%s_topn_%s_temperature_%s.log_%s' % \
                    (self.exp_config.option, self.exp_config.dataset, self.exp_config.model, self.exp_config.topn, self.exp_config.temperature, str(self.log_config.log_phase_input))
            remove_percentage = string_to_int(get_ith_element(self.exp_config.option, 1))
        
       # write printed output to a file (print_file)
        print_file_str = './log/print' + self.log_config.log_file[5:]
        self.log_config.print_file = open(print_file_str, 'a') # append new content if exists already
        return remove_percentage


    def load_dataset(self):
        dataset_loc = self.exp_config.dataset_loc
        line_cnt = 0
        problem_list = []
        # Load problems from dataset with optional filtering
        with open(dataset_loc, 'r') as f:
            for line in f.readlines():
                # Apply index filtering if specified
                if self.exp_config.min_problem_idx < 0 or line_cnt >= self.exp_config.min_problem_idx:
                    self.problem_list.append(json.loads(line))
                line_cnt += 1
                # Stop if maximum number of problems is reached
                if self.exp_config.max_num_problems >= 0 and line_cnt >= self.exp_config.max_num_problems:
                    break
        return problem_list


    def load_cached_results(self):
        cached_names = set()
        cached_responses = {}
        cached_answers = {}
        cached_qqs = {}
        if os.path.exists(self.log_config.log_file):
            with open(self.log_config.log_file, 'r') as f:
                for line in f:
                    content = json.loads(line)
                    key = content['name']+'_'+content['prompt_type']
                    cached_names.add(key)
                    cached_responses[key] = content['response']
                    cached_answers[key] = content['answer']
                    cached_qqs[key] = content['question_quality']
        cache = Cache(cached_names, cached_responses, cached_answers, cached_qqs)
        return cache
        
        


    def HumanEval_experiment(self):
        # ============================================================================
        # MAIN EXPERIMENT LOOP
        # ============================================================================
        response_list = []
        for problem in self.problem_list:
            print('----------------------problem name: %s--------------------------------' % (problem['name']), flush=True)
            print('using %s to generate response' % (self.model), flush=True)
            
            # Determine which prompt fields to process based on dataset type
            dataset = self.exp_config.dataset
            input_prompt_fields = HumanEvalComm_prompts if dataset == "HumanEvalComm" else HumanEval_prompt
            # Process each prompt variant for the current problem
            for input_prompt in input_prompt_fields:
                if input_prompt not in problem:
                    continue
                    
                key = problem['name'] + '_' + input_prompt
                
                # Skip if already processed (caching mechanism)
                if self.log_config.log_phase_input == self.log_config.log_phase_output and key in self.cache.cached_names:
                    continue
                    
                print("********************************************************************", file=self.log_config.print_file)
                print("****** new problem (name="+problem['name']+" input_prompt="+input_prompt+") ******", file=self.log_config.print_file)
                print("********************************************************************", file=self.log_config.print_file)
                
                description = problem[input_prompt]
                try:
                    # Multi-round evaluation with clarifying questions
                    original_prompt = problem['prompt']
                    entry_point = problem['entry_point']
                    task_id = problem['name']
                    
                    # Determine if this is a modified prompt (triggers question mode)
                    prompt_modified = False if input_prompt == 'prompt' else True
                    prompt_start = ORIGINAL_PROMPT_START_0 if input_prompt == 'prompt' else self.exp_config.phase1_prompt
                    
                    handler = Handler(self.model, self.tokenizer, self.cache, self.exp_config, self.log_config)
                    response_list, code_list, qq_list, ans_list = handler.description_2_code_multi_rounds(
                        prompt_modified= prompt_modified,
                        task_id=task_id,
                        entry_point=entry_point,
                        prompt_start=prompt_start,
                        description=description,
                        original_prompt=original_prompt
                    )
                except Exception as e:
                    print('%s---------%s' % (problem['name'], e), flush=True)
                    continue
                # ============================================================================
                # RESULT LOGGING AND STORAGE
                # ============================================================================
                # Create the prompt based on experiment option
                prompt = create_prompt(description, self.exp_config.option, self.remove_percentage)
                self.log_result_(problem=problem,
                    input_prompt=input_prompt,
                    key=key,
                    prompt=prompt,
                    description=description,
                    response_list=response_list,
                    code_list=code_list,
                    qq_list=qq_list,
                    ans_list=ans_list
                )
                
                print('%s finish!' % (problem['name']), flush=True)
                # Debug option: uncomment to stop after first prompt
                # break
        
        print('Done!', flush=True)

    def log_result_(self, problem, key, prompt, input_prompt, description, response_list, code_list, qq_list, ans_list):
        for i in range(len(response_list)):
            res = {
                'name': problem['name'],
                'prompt_type': input_prompt,
                'index': i,
                'response': response_list[i],
                'answer': ans_list[i] if i < len(ans_list) else '',
                'question_quality': qq_list[i] if i < len(qq_list) else '0',
                'code': code_list[i] if i < len(code_list) else '',
            }

            if self.log_config.log_phase_output >= 1:
                # Phase-based logging format (for multi-phase experiments)
                res.update({'key': key,})
                last_index = self.log_config.log_file.rfind('.log_')
                log_file_output = self.log_config.log_file[:last_index] + '.log_' + str(self.log_config.log_phase_output)
                
            else:
                # Standard logging format (for complete experiments)
                res.update({
                    'original_prompt': description,
                    'modified_prompt': prompt,
                })
                log_file_output = self.log_config.log_file
                
            print('response %s is writting into file' % (i), flush=True)
            json_str = json.dumps(res)
            with open(log_file_output, 'a') as f:
                f.write(json_str + '\n')
            



