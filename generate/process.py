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
from prompt import create_prompt
from handle import Handler
from config import *
import os
import json
import torch
import time
from handle import Handler

from model.cache import Cache
from model.experiment import CoreExpConfig

class Experiment:
    def __init__(self, exp_config: CoreExpConfig):
        self.exp_config = exp_config
        self.problem_list = []
        self.cache = None
        self.log_config = None
        self.remove_percentage = 0
        self.prepare_experiment()
    
    def prepare_experiment(self):
        # EXPERIMENT SETUP AND LOGGING
        # ============================================================================
        cache_path, print_file, self.remove_percentage = self.init_file_log_and_print(self.exp_config)
        self.log_config = LogConfig(print_file=print_file)
        self.cache = self.load_cached_results(cache_path)
        self.problem_list = self.load_dataset(
            self.exp_config.datasetConfig.dataset_loc, 
            self.exp_config.datasetConfig.min_problem_idx, 
            self.exp_config.datasetConfig.max_num_problems
        )

    def init_file_log_and_print(self, ExpConfig: CoreExpConfig):
        cache_path = None
        print_file = None
        remove_percentage = 0

        # Determine log file name based on experiment configuration
        if ExpConfig.datasetConfig.option == 'original':
            cache_path = './log/dataset_%s_model_%s_topn_%s_temperature_%s.log_%s' % \
                    (ExpConfig.datasetConfig.dataset, ExpConfig.modelConfig.model, ExpConfig.modelConfig.topn, ExpConfig.modelConfig.temperature, str(ExpConfig.log_phase_input))
        else:
            cache_path = './log/%s_dataset_%s_model_%s_topn_%s_temperature_%s.log_%s' % \
                    (ExpConfig.datasetConfig.option, ExpConfig.datasetConfig.dataset, ExpConfig.modelConfig.model, ExpConfig.modelConfig.topn, ExpConfig.modelConfig.temperature, str(ExpConfig.log_phase_input))
            remove_percentage = string_to_int(get_ith_element(ExpConfig.datasetConfig.option, 1))
        
       # write printed output to a file (print_file)
        print_file_str = './log/print' + cache_path[5:]
        print_file = open(print_file_str, 'a') # append new content if exists already
        return cache_path, print_file, remove_percentage


    def load_dataset(self, dataset_loc, min_problem_idx, max_num_problems):
        line_cnt = 0
        problem_list = []
        # Load problems from dataset with optional filtering
        with open(dataset_loc, 'r') as f:
            for line in f.readlines():
                # Apply index filtering if specified
                if min_problem_idx < 0 or line_cnt >= min_problem_idx:
                    problem_list.append(json.loads(line))
                line_cnt += 1
                # Stop if maximum number of problems is reached
                if max_num_problems >= 0 and line_cnt >= max_num_problems:
                    break
        return problem_list


    def load_cached_results(self, cache_path):
        cached_names = set()
        cached_responses = {}
        cached_answers = {}
        cached_qqs = {}
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                for line in f:
                    content = json.loads(line)
                    key = content['name']+'_'+content['prompt_type']
                    cached_names.add(key)
                    cached_responses[key] = content['response']
                    cached_answers[key] = content['answer']
                    cached_qqs[key] = content['question_quality']
        cache = Cache(
            cached_file_path=cache_path, 
            cached_names=cached_names, 
            cached_responses=cached_responses, 
            cached_answers=cached_answers, 
            cached_qqs=cached_qqs
        )
        return cache
        
        


    def HumanEval_experiment(self, model, tokenizer):
        # ============================================================================
        # MAIN EXPERIMENT LOOP
        # ============================================================================
        response_list = []
        handler = Handler(model, tokenizer, 
            exp_config=self.exp_config, 
            log_config=self.log_config, 
            cache=self.cache
        )
        for problem in self.problem_list:
            print('----------------------problem name: %s--------------------------------' % (problem['name']), flush=True)
            print('using %s to generate response' % (self.exp_config.modelConfig.model), flush=True)
            
            # Determine which prompt fields to process based on dataset type
            prompt_types = HumanEvalComm_prompts if self.exp_config.datasetConfig.dataset == "HumanEvalComm" else HumanEval_prompt
            # Process each prompt variant for the current problem
            for prompt_type in prompt_types:
                if prompt_type not in problem:
                    continue
                key = problem['name'] + '_' + prompt_type
                
                # Skip if already processed (caching mechanism)
                if self.exp_config.log_phase_input == self.exp_config.log_phase_output and key in self.cache.cached_names:
                    continue
                    
                print("********************************************************************", file=self.log_config.print_file)
                print("****** new problem (name="+problem['name']+" prompt_type="+prompt_type+") ******", file=self.log_config.print_file)
                print("********************************************************************", file=self.log_config.print_file)
                
                
                try:
                    # Multi-round evaluation with clarifying questions
                    response_list, code_list, qq_list, ans_list = handler.description_2_code_multi_rounds(
                        prompt_type=prompt_type,
                        problem=problem
                    )
                except Exception as e:
                    print('%s---------%s' % (problem['name'], e), flush=True)
                    continue
                # ============================================================================
                # RESULT LOGGING AND STORAGE
                # ============================================================================
                # Create the prompt based on experiment option
                prompt = create_prompt(problem[prompt_type], self.exp_config.datasetConfig.option, self.remove_percentage)
                self.log_result_(
                    problem=problem,
                    prompt_type=prompt_type,
                    key=key,
                    prompt=prompt,
                    response_list=response_list,
                    code_list=code_list,
                    qq_list=qq_list,
                    ans_list=ans_list
                )
                
                print('%s finish!' % (problem['name']), flush=True)
                # Debug option: uncomment to stop after first prompt
                # break
        
        print('Done!', flush=True)

    def log_result_(self, problem, key, prompt, prompt_type, response_list, code_list, qq_list, ans_list):
        for i in range(len(response_list)):
            res = {
                'name': problem['name'],
                'prompt_type': prompt_type,
                'index': i,
                'response': response_list[i],
                'answer': ans_list[i] if i < len(ans_list) else '',
                'question_quality': qq_list[i] if i < len(qq_list) else '0',
                'code': code_list[i] if i < len(code_list) else '',
            }

            if self.exp_config.log_phase_output >= 1:
                # Phase-based logging format (for multi-phase experiments)
                res.update({'key': key,})
                last_index = self.cache.cached_file_path.rfind('.log_')
                log_file_output = self.cache.cached_file_path[:last_index] + '.log_' + str(self.exp_config.log_phase_output)
                
            else:
                # Standard logging format (for complete experiments)
                res.update({
                    'original_prompt': problem['prompt_type'],
                    'modified_prompt': prompt,
                })
                log_file_output = self.log_config.log_file
                
            print('response %s is writting into file' % (i), flush=True)
            json_str = json.dumps(res)
            with open(log_file_output, 'a') as f:
                f.write(json_str + '\n')
            



