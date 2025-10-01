"""
Response Handling Module
======================

This module handles the core logic for generating responses from models and
managing multi-round conversations for code generation with clarifying questions.

Key functions:
- description_2_code_one_round: Single-round code generation (legacy)
- description_2_code_multi_rounds: Multi-round conversation with Q&A
- generate_response: Unified response generation for different model types

Supported model types:
- OpenAI models (GPT-3.5, GPT-4)
- Open source models (CodeLlama, StarChat, DeepSeek, CodeQwen)
- Special models (Okanagan, AgentCoder)

Features:
- Multi-round conversation support
- Phase-based evaluation
- Question quality assessment
- Code extraction and validation
"""

# legacy code (randRemove) where only one-round evaluation is enabled
import openai
from utils import response_2_code, response_2_code_if_no_text
from config import *
from callers.CodeLlama import get_completion_codellama_instruct_nl_to_pl
import re
from model.cache import Cache
from model.log import LogConfig
from model.experiment import ExpConfig


class Handler:
    def __init__(self, model, tokenizer, cache: Cache, exp_config: ExpConfig, log_config: LogConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache
        self.log_config = log_config
        self.exp_config = exp_config


    def description_2_code_multi_rounds(self, prompt_modified, task_id, entry_point, prompt_start, description, original_prompt):
        # ============================================================================
        # INITIALIZATION
        # ============================================================================
        
        response_list = []
        code_list = []
        qq_list = []
        ans_list = []
        
        
        # ============================================================================
        # ROUND 1: INITIAL RESPONSE GENERATION
        # ============================================================================
        response_list, messages = self.generate_init_response()
        # Early return if only first round is needed
        if self.cache.log_phase_output == 1:
            return response_list, [], [], []
        # ============================================================================
        # PROCESSING EACH RESPONSE
        # ============================================================================
        for i in range(len(response_list)):
            response = response_list[i]
            code = response_2_code_if_no_text(response)
            
            print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=self.log_config.print_file)
            print('!!!!!!!!!!!!! 1st CodeLLM response:\n' + response, file=self.log_config.print_file)
            print('!!!!!!!!!!!!! 1st CodeLLM response code:\n' + code, file=self.log_config.print_file)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n", file=self.log_config.print_file)
            
            # Initialize default values
            question_quality = '0'
            answer = ''
            
            # ============================================================================
            # ROUND 2: QUESTION EVALUATION AND ANSWER GENERATION
            # ============================================================================
            if code == '':
                
                if self.cache.log_phase_output >= 3:
                    # Use cached evaluation results for phase-based evaluation
                    answer = self.cache.cached_answer
                    question_quality = self.cache.cached_qq
                else:
                    # Perform live evaluation of questions
                    answer, question_quality = self.evaluate_clarifying_questions()
                
                if self.cache.log_phase_output == 2:
                    # Only return evaluation results, skip final code generation
                    ans_list.append(answer)
                    qq_list.append(question_quality)
                    continue

                # ============================================================================
                # ROUND 3: FINAL CODE GENERATION WITH Q&A CONTEXT
                # ============================================================================
                code = self.generate_final_response()
            # Store results for this response
            qq_list.append(question_quality)
            code_list.append(code)
            ans_list.append(answer)
        
        return response_list, code_list, qq_list, ans_list



    def generate_init_response(self):
        messages = []
        if(self.model_config.model == 'Okanagan'):
            full_prompt = OK_PROMPT_CODEGEN + self.exp_config.user_input
        else:
            full_prompt = self.exp_config.prompt.format(problem=self.exp_config.user_input)

        print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=self.log_config.print_file)
        print('!!!!!!!!!!!!! prompt:\n' + full_prompt, file=self.log_config.print_file)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", file=self.log_config.print_file)
        
        messages.append({"role": "user","content": full_prompt})
        
        # Generate first-round response (or use cached response for phase-based evaluation)
        if self.cache.log_phase_output >= 2:
            response_list.append(self.cache.cached_response)
        else:
            response_list = generate_response()
            
        return response_list, messages





    def evaluate_clarifying_questions(self):
        """
        Evaluate the quality of clarifying questions and generate answers.
        
        Args:
            missing_information (str): The original, complete problem description
            clarifying_questions (str): The clarifying questions generated by the model
            problem (str): The modified/incomplete problem description
            eval_protocol (str): Evaluation protocol to use ('llm_metric_v2' for LLM-based evaluation)
        
        Returns:
            tuple: (answer_string, quality_score) where:
                - answer_string: Generated answer to the clarifying questions
                - quality_score: Quality rating of the questions (0-3 scale)
        """
        print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=print_file)
        print('!!!!!!! 2nd evaluate_clarifying_questions START !!!!!!!!!!!', file=print_file)
        
        # ============================================================================
        # STANDARD EVALUATION PROTOCOL (Default)
        # ============================================================================
        # Use GPT-3.5-turbo for standard evaluation with regex parsing
        
        topn = 1
        temperature = 1.0
        model = 'gpt-3.5-turbo-0125'  # Default model for evaluation
        prompt_evaluate_questions = load_prompt_from_config(phase = 2)
        
        # Format the evaluation prompt with the provided information
        content = prompt_evaluate_questions.format(
                    missing_information=missing_information,
                    clarifying_questions=clarifying_questions,
                    problem=problem
                )
        
        # Call OpenAI API for evaluation
        completion = openai.ChatCompletion.create(
            model=model,
            n=topn,
            temperature=temperature,
            messages=[{
                "role": "user",
                "content": content,
            }]
        )
        
        print('!!!!!!!PROMPT_EVALUATE_QUESTIONS='+content, file=print_file)
        print('!!!!!!!Completion='+completion['choices'][0]['message']['content'], file=print_file)
        
        # ============================================================================
        # RESPONSE PARSING
        # ============================================================================
        # Extract quality score and answers using regex patterns
        completion_content = str(completion['choices'][0]['message']['content'])

        # Extract quality score (integer following "QUALITY=")
        question_quality = re.findall(r'QUALITY\s*=?\s*(\d+)', completion_content)
        
        # Extract answers (text within triple backticks following "ANSWERS=")
        answers = re.findall(r'ANSWERS\s*=?\s*```(.+?)```', completion_content, flags=re.DOTALL)
        
        # Extract the first answer and quality score, or use empty string if not found
        answer_str = answers[0] if answers else ""
        question_quality_str = question_quality[0] if question_quality else ""
        
        print('!!!!!!!answer_str',answer_str, file=print_file)
        print('!!!!!!!question_quality_str',question_quality_str, file=print_file)
        
        print('!!!!!!! 2nd evaluate_clarifying_questions END !!!!!!!!!!!', file=print_file)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n", file=print_file)
        return answer_str, question_quality_str
    


    def generate_final_response(self, messages, response, answer):
        self.model_config.model = OK_MODEL if self.model_config.model == 'Okanagan' else self.model_config.model
        # Standard multi-turn conversation format
        msgs_i = messages.copy()
        msgs_i.append({"role":"assistant","content": response})
        msgs_i.append({"role":"user","content": answer + PROMPT_2ND_ROUND})
        
        # Generate final code with full conversation context
        response_2nd = generate_response(self.log_config, msgs_i, self.model_config, topn=1, temperature=self.model_config.temperature)
        code = response_2_code(response_2nd[0])
        
        print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=self.log_config.print_file)
        print('!!!!!!!!!!!!! 3rd CodeLLM input messages:\n', msgs_i, file=self.log_config.print_file)
        print('!!!!!!!!!!!!! 3rd CodeLLM response:\n', response_2nd, file=self.log_config.print_file)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n", file=self.log_config.print_file)
        return code


def generate_response_str(model, msgs, temperature, args, open_source_model, tokenizer):
    """
    Generate a single response string (convenience function).
    
    Args:
        model (str): Model identifier
        msgs (list): List of message dictionaries
        temperature (float): Sampling temperature
        args: Command line arguments
        open_source_model: Loaded model (if applicable)
        tokenizer: Model tokenizer (if applicable)
    
    Returns:
        str: Single generated response
    """
    response_list = generate_response(model, msgs, 1, temperature, args, open_source_model, tokenizer)
    return response_list[0]
    
def generate_response(log_config,msgs,model_config, topn, temperature):
    response_list = []
    
    # ============================================================================
    # OPEN SOURCE MODELS (CodeLlama, DeepSeek, CodeQwen)
    # ============================================================================
    # if 'Llama' in args.model or 'deepseek' in args.model or 'CodeQwen' in args.model:
    #     # Apply chat template for instruction-tuned models
    #     user_input = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, return_tensors="pt")
        
    #     for i in range(topn):
    #         if 'two-shot' in args.model:
    #             # Two-shot prompting with predefined examples
    #             response_list.append(get_completion_codellama_instruct_nl_to_pl(CODELLAMA_NL_2_PL_HUMANEVAL, user_input_without_prompt, open_source_model, tokenizer, args))
    #         elif 'Llama' in args.model:
    #             # Custom prompt for CodeLlama models
    #             CODELLAMA_NL_2_PL_PROMPT = [
    #                 {  # Instructions
    #                     "role": "system",
    #                     "content": prompt,
    #                 },
    #             ]
    #             response_list.append(get_completion_codellama_instruct_nl_to_pl('CODELLAMA', prompt + user_input_without_prompt, open_source_model, tokenizer, args))
    #         else:
    #             # Standard generation for other open source models
    #             response_list.append(get_completion_codellama_instruct_nl_to_pl('', user_input, open_source_model, tokenizer, args))
    #     return response_list
    # ============================================================================
    # SPECIAL MODELS
    # ============================================================================
    if model_config.model == 'Okanagan':
        """
        Okanagan model: Two-step process with reflection.
        First generates code, then reflects on whether questions are needed.
        """
        # Note: This assumes topn=1 (single response)
        coder_response = generate_response_str(msgs, model_config)

        # Reflection step: determine if clarifying questions are needed
        reflect_messages = [{"role": "user","content": OK_PROMPT_CLARIFY_Q.format(code=coder_response, problem=user_input_without_prompt)}]
        communicator_response = generate_response_str(reflect_messages, model_config)
        
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=log_config.print_file)
        print("!!!!!!!!!!!!!!! Okanagan !!!!!! communicator_response: \n" + communicator_response, file=log_config.print_file)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", file=log_config.print_file)
        
        # Decide final response based on reflection
        if re.search('no_questions', communicator_response, re.IGNORECASE):
            response_list.append(coder_response)  # Use the original code
        else:
            response_list.append(communicator_response)  # Use the questions
        return response_list  
        
        
    # ============================================================================
    # STANDARD OPENAI MODELS
    # ============================================================================
    else:
        # Standard OpenAI API call for GPT models
        completion = openai.ChatCompletion.create(
            model=model,
            n=topn,
            temperature=temperature,
            messages=msgs
        )
        for i in completion['choices']:
            response_list.append(i['message']['content'])
        return response_list