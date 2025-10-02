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
from utils import response_2_code, response_2_code_if_no_text, call_azure_openai, get_azure_openai_client
from config import *
import re
from model.cache import Cache
from model.log import LogConfig
from model.experiment import CoreExpConfig
from prompt import load_prompt_from_config


class Handler:
    def __init__(self, model, tokenizer, cache: Cache, exp_config: CoreExpConfig, log_config: LogConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache
        self.log_config = log_config
        self.exp_config = exp_config
        # Determine if this is a modified prompt (triggers question mode)
        self.prompt_phase1 = load_prompt_from_config(phase = 1, prompt_config = self.exp_config.promptConfig)
        self.prompt_phase2 = load_prompt_from_config(phase = 2, prompt_config = self.exp_config.promptConfig)
        self.client = get_azure_openai_client()


    

    def description_2_code_multi_rounds(self, prompt_type, problem):
        code_list = []
        qq_list = []
        ans_list = []
        #the key for cache access
        key = problem['name'] + '_' + prompt_type
        # ============================================================================
        # ROUND 1: INITIAL RESPONSE GENERATION
        # ============================================================================
        #generate initial response for clarifying questions or codes
        messages, response_list = self.generate_init_response(problem[prompt_type], key)
        # Early return if only first round is needed
        if self.exp_config.log_phase_output == 1:
            return response_list, [], [], []
        
        # print("response_list: ", response_list)
        print("type of response_list: ", type(response_list[0]))
        for response in response_list:
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
                
                if self.exp_config.log_phase_input >= 2:
                    # Use cached evaluation results for phase-based evaluation
                    answer = self.cache.cached_answers.get(key, '')
                    question_quality = self.cache.cached_qqs.get(key, '')
                else:
                    # Perform live evaluation of clarifying questions
                    answer, question_quality = self.evaluate_clarifying_questions(problem['prompt'], response, problem[prompt_type])
                
                if self.exp_config.log_phase_output == 2:
                    # Only return evaluation results, skip final code generation
                    ans_list.append(answer)
                    qq_list.append(question_quality)
                    continue

                # ============================================================================
                # ROUND 3: FINAL CODE GENERATION WITH Q&A CONTEXT
                # ============================================================================
                code = self.generate_final_response(messages, response, answer)
            # Store results for this response
            qq_list.append(question_quality)
            code_list.append(code)
            ans_list.append(answer)
        
        return response_list, code_list, qq_list, ans_list



    def generate_init_response(self, description, key):
        response_list = []
        messages = []
        full_prompt = OK_PROMPT_CODEGEN + description
        print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=self.log_config.print_file)
        print('!!!!!!!!!!!!! prompt:\n' + full_prompt, file=self.log_config.print_file)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", file=self.log_config.print_file)
        messages.append({"role": "user","content": full_prompt})

        if self.exp_config.log_phase_input >= 1:
            response_list = [self.cache.cached_responses.get(key, '')]
            return messages, response_list
        # Generate first-round response (or use cached response for phase-based evaluation)
        response_list = self.generate_response(messages, 1, 1.0, description)
        return messages, response_list





    def evaluate_clarifying_questions(self, missing_information, clarifying_questions, problem_description):
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
        print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=self.log_config.print_file)
        print('!!!!!!! 2nd evaluate_clarifying_questions START !!!!!!!!!!!', file=self.log_config.print_file)
        prompt_evaluate_questions = self.prompt_phase2
        # Format the evaluation prompt with the provided information
        content = prompt_evaluate_questions.format(
                    missing_information=missing_information,
                    clarifying_questions=clarifying_questions,
                    problem=problem_description
                )
        
        messages = []
        messages.append({"role": "user","content": content})
        response_list, error = call_azure_openai(messages, self.client, 1, 1)
        if error:
            print("call azure_openai error: ", error, file=self.log_config.print_file)
            return "", ""
        
        print('!!!!!!!PROMPT_EVALUATE_QUESTIONS='+content, file=self.log_config.print_file)
        print('!!!!!!!Completion='+response_list[0], file=self.log_config.print_file)
        
        # ============================================================================
        # RESPONSE PARSING
        # ============================================================================
        # Extract quality score and answers using regex patterns
        response_str = str(response_list[0])

        # Extract quality score (integer following "QUALITY=")
        question_quality = re.findall(r'QUALITY\s*=?\s*(\d+)', response_str)
        
        # Extract answers (text within triple backticks following "ANSWERS=")
        answers = re.findall(r'ANSWERS\s*=?\s*```(.+?)```', response_str, flags=re.DOTALL)
        
        # Extract the first answer and quality score, or use empty string if not found
        answer_str = answers[0] if answers else ""
        question_quality_str = question_quality[0] if question_quality else ""
        
        print('!!!!!!!answer_str',answer_str, file=self.log_config.print_file)
        print('!!!!!!!question_quality_str',question_quality_str, file=self.log_config.print_file)
        
        print('!!!!!!! 2nd evaluate_clarifying_questions END !!!!!!!!!!!', file=self.log_config.print_file)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n", file=self.log_config.print_file)
        return answer_str, question_quality_str
    


    def generate_final_response(self, messages, response, answer):
        # Standard multi-turn conversation format
        msgs_i = messages.copy()
        msgs_i.append({"role":"assistant","content": response})
        msgs_i.append({"role":"user","content": answer + PROMPT_2ND_ROUND})
        
        # Generate final code with full conversation context
        # response_2nd = self.generate_response(msgs_i, topn=1, temperature=1.0)
        response_2nd, error = call_azure_openai(msgs_i, self.client, 1, 1.0)
        if error:
            print("call azure_openai error: ", error, file=self.log_config.print_file)
            return ''
        code = response_2_code(response_2nd[0])
        
        print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=self.log_config.print_file)
        print('!!!!!!!!!!!!! 3rd CodeLLM input messages:\n', msgs_i, file=self.log_config.print_file)
        print('!!!!!!!!!!!!! 3rd CodeLLM response:\n', response_2nd, file=self.log_config.print_file)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n", file=self.log_config.print_file)
        return code




        
    def generate_response(self, messages, topn = 1, temperature = 1.0, promblem_description = ''):
        response_list = []
        
        """
        Okanagan model: Two-step process with reflection.
        First generates code, then reflects on whether questions are needed.
        """
        
        # Note: This assumes topn=1 (single response)
        coder_response, error = call_azure_openai(messages, self.client, 1, temperature)
        if error:
            print("call azure_openai error: ", error, file=self.log_config.print_file)
            return []

        # Reflection step: determine if clarifying questions are needed
        reflect_messages = [{"role": "user","content": OK_PROMPT_CLARIFY_Q_V1.format(code=coder_response[0], problem=promblem_description)}]
        communicator_response, error = call_azure_openai(reflect_messages, self.client, 1, temperature)
        if error:
            print("call azure_openai error: ", error, file=self.log_config.print_file)
            return []
        
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=self.log_config.print_file)
        print("!!!!!!!!!!!!!!! Okanagan !!!!!! communicator_response: \n" + communicator_response[0], file=self.log_config.print_file)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", file=self.log_config.print_file)
        
        # Decide final response based on reflection
        if re.search('no_questions', communicator_response[0], re.IGNORECASE):
            response_list.append(coder_response[0])  # Use the original code
        else:
            response_list.append(communicator_response[0])  # Use the questions
        return response_list  
            
        