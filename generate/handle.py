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
from callers.generate_text import generate_response
from utils import response_2_code, response_2_code_if_no_text
from evaluate import evaluate_clarifying_questions
from config import PROMPT_START_3_v4, CODELLAMA_NL_2_PL_HUMANEVAL, OK_MODEL, OK_PROMPT_CLARIFY_Q, OK_PROMPT_CODEGEN, PROMPT_2ND_ROUND
from callers.CodeLlama import get_completion_codellama_instruct_nl_to_pl
from AgentFramework.programmer import programmer_main
from AgentFramework.designer import designer_main
from AgentFramework.executor import executor_main
from process import print_file
import re




def description_2_code_one_round(prompt, model, topn, temperature, args, open_source_model, tokenizer):
    """
    Generate code in a single round (legacy function for randRemove experiments).
    
    This function is used for single-round evaluation where models generate code
    directly without asking clarifying questions. It's primarily used for
    legacy experiments with the 'randRemove' option.
    
    Args:
        prompt (str): The problem description prompt
        model (str): The model identifier
        topn (int): Number of candidate responses to generate
        temperature (float): Sampling temperature for generation
        args: Command line arguments
        open_source_model: Loaded open source model (if applicable)
        tokenizer: Model tokenizer (if applicable)
    
    Returns:
        tuple: (response_list, code_list, qq_list) where:
            - response_list: List of raw model responses
            - code_list: List of extracted code from responses
            - qq_list: List of question quality scores (always '0' for single round)
    """
    if model=='comm':
        # Special handling for 'comm' model with two-step process
        completion = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            n=1,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        first_response_list = []
        for i in completion['choices']:
            first_response_list.append(i['message']['content'])

        # Create a follow-up prompt with the initial response
        new_prompt = "You are an expert in software engineering. You will be given the problem description and current code of a coding task. You will decide whether to ask clarifying questions or return the code with markup. \n ### Problem Description: \n"+ prompt + "\n ### Generated Code From Previous Iteration:\n" + first_response_list[0]
        
        completion = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            n=topn,
            temperature=temperature,
            messages=[{"role": "user", "content": new_prompt}]
        )
        response_list = []
        for i in completion['choices']:
            response_list.append(i['message']['content'])

    else:
        # Standard single-round generation for other models
        messages=[{"role": "user", "content": prompt}]
        response_list = generate_response(model, messages, topn, temperature, args, open_source_model, tokenizer)
    
    # Extract code from responses and set question quality to 0 (no questions in single round)
    code_list = []
    qq_list = []
    for i in range(len(response_list)):
        code = response_2_code(response_list[i])
        code_list.append(code)
        qq_list.append('0')  # No questions in single round
    
    return response_list, code_list, qq_list





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
    
def generate_response(model, msgs, topn, temperature, args, open_source_model, tokenizer, user_input_without_prompt = '', prompt = ''):
    """
    Unified response generation function for different model types.
    
    This function handles response generation for various model types including
    open source models (CodeLlama, StarChat, DeepSeek, CodeQwen), special
    models (Okanagan, AgentCoder), and standard OpenAI models.
    
    Args:
        model (str): Model identifier
        msgs (list): List of message dictionaries for the conversation
        topn (int): Number of candidate responses to generate
        temperature (float): Sampling temperature for generation
        args: Command line arguments
        open_source_model: Loaded open source model (if applicable)
        tokenizer: Model tokenizer (if applicable)
        user_input_without_prompt (str): User input without system prompt
        prompt (str): System prompt template
    
    Returns:
        list: List of generated response strings
    """
    response_list = []
    
    # ============================================================================
    # OPEN SOURCE MODELS (CodeLlama, DeepSeek, CodeQwen)
    # ============================================================================
    if 'Llama' in args.model or 'deepseek' in args.model or 'CodeQwen' in args.model:
        # Apply chat template for instruction-tuned models
        user_input = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, return_tensors="pt")
        
        for i in range(topn):
            if 'two-shot' in args.model:
                # Two-shot prompting with predefined examples
                response_list.append(get_completion_codellama_instruct_nl_to_pl(CODELLAMA_NL_2_PL_HUMANEVAL, user_input_without_prompt, open_source_model, tokenizer, args))
            elif 'Llama' in args.model:
                # Custom prompt for CodeLlama models
                CODELLAMA_NL_2_PL_PROMPT = [
                    {  # Instructions
                        "role": "system",
                        "content": prompt,
                    },
                ]
                response_list.append(get_completion_codellama_instruct_nl_to_pl('CODELLAMA', prompt + user_input_without_prompt, open_source_model, tokenizer, args))
            else:
                # Standard generation for other open source models
                response_list.append(get_completion_codellama_instruct_nl_to_pl('', user_input, open_source_model, tokenizer, args))
        return response_list
    # ============================================================================
    # SPECIAL MODELS
    # ============================================================================
    elif model == 'Okanagan':
        """
        Okanagan model: Two-step process with reflection.
        First generates code, then reflects on whether questions are needed.
        """
        # Note: This assumes topn=1 (single response)
        coder_response = generate_response_str(OK_MODEL, msgs, temperature, args, open_source_model, tokenizer)

        # Reflection step: determine if clarifying questions are needed
        reflect_messages = [{"role": "user","content": OK_PROMPT_CLARIFY_Q.format(code=coder_response, problem=user_input_without_prompt)}]
        communicator_response = generate_response_str(OK_MODEL, reflect_messages, temperature, args, open_source_model, tokenizer)
        
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=print_file)
        print("!!!!!!!!!!!!!!! Okanagan !!!!!! communicator_response: \n" + communicator_response, file=print_file)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", file=print_file)
        
        # Decide final response based on reflection
        if re.search('no_questions', communicator_response, re.IGNORECASE):
            response_list.append(coder_response)  # Use the original code
        else:
            response_list.append(communicator_response)  # Use the questions
        return response_list  
        
    elif model == 'AgentCoder':
        """
        AgentCoder model: Multi-agent system with programmer, designer, and executor.
        """
        task_id = msgs[0]["task_id"]
        task_id = task_id.replace("/", "_")
        
        print("Running Programmer")
        responses = programmer_main(model, "python", msgs, openai.api_key, task_id)

        if msgs[0]["clarity_prompt"]=="":
            # Standard flow: programmer -> designer -> executor
            print("Running Designer")
            test_cases = designer_main(model, "python", responses, openai.api_key, task_id)
            
            print("Running Executor")
            results = executor_main(task_id)
            
            response_list.append(str(results[0]['completion']))
        else:
            # Clarifying questions flow: return programmer's response
            response_list.append(str(responses[0]['completion_list']))
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





def description_2_code_multi_rounds(prompt_modified, task_id, entry_point, prompt, user_input, original_prompt, model, topn, temperature, args, open_source_model, tokenizer, cached_response, cached_qq, cached_answer):
    """
    Generate code through multi-round conversation with clarifying questions.
    
    This is the main function for multi-round code generation experiments.
    It implements a 3-round process:
    1. First round: Initial code generation or question asking
    2. Second round: Question evaluation and answer generation (if questions were asked)
    3. Third round: Final code generation with Q&A context
    
    Args:
        prompt_modified (bool): Whether the prompt is modified (triggers question mode)
        task_id (str): Unique identifier for the task
        entry_point (str): Function name/entry point for the problem
        prompt (str): System prompt template
        user_input (str): User's problem description
        original_prompt (str): Original unmodified problem description
        model (str): Model identifier
        topn (int): Number of candidate responses
        temperature (float): Sampling temperature
        args: Command line arguments
        open_source_model: Loaded model (if applicable)
        tokenizer: Model tokenizer (if applicable)
        cached_response (str): Cached first-round response (for phase-based evaluation)
        cached_qq (str): Cached question quality score (for phase-based evaluation)
        cached_answer (str): Cached answer to questions (for phase-based evaluation)
    
    Returns:
        tuple: (response_list, code_list, qq_list, ans_list) where:
            - response_list: List of first-round responses
            - code_list: List of final generated code
            - qq_list: List of question quality scores
            - ans_list: List of answers to clarifying questions
    """
    
    # ============================================================================
    # INITIALIZATION
    # ============================================================================
    messages = []
    response_list = []
    model_2nd_round = OK_MODEL if model == 'Okanagan' else model
    
    # ============================================================================
    # ROUND 1: INITIAL RESPONSE GENERATION
    # ============================================================================
    if model == "AgentCoder":
        # Special handling for AgentCoder with custom message format
        if prompt_modified == False:
            messages.append({"task_id": task_id,"prompt": original_prompt, "entry_point": entry_point, "clarity_prompt": ""})
        else:
            messages.append({"task_id": task_id,"prompt": original_prompt, "entry_point": entry_point, "clarity_prompt": PROMPT_START_3_v4})
    else:
        # Standard message format for other models
        if(model == 'Okanagan'):
            full_prompt = OK_PROMPT_CODEGEN + user_input
        else:
            full_prompt = prompt.format(problem=user_input)

        print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=print_file)
        print('!!!!!!!!!!!!! prompt:\n' + full_prompt, file=print_file)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", file=print_file)
        
        messages.append({"role": "user","content": full_prompt})
    
    # Generate first-round response (or use cached response for phase-based evaluation)
    if args.log_phase_output >= 2:
        response_list.append(cached_response)
    else:
        response_list = generate_response(model, messages, topn, temperature, args, open_source_model, tokenizer, user_input, prompt)
        
    # Early return if only first round is needed
    if args.log_phase_output == 1:
        return response_list, [], [], []

    # ============================================================================
    # PROCESSING EACH RESPONSE
    # ============================================================================
    code_list = []
    qq_list = []
    ans_list = []
    
    for i in range(len(response_list)):
        response = response_list[i]
        code = response_2_code_if_no_text(response)
        
        print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=print_file)
        print('!!!!!!!!!!!!! 1st CodeLLM response:\n' + response, file=print_file)
        print('!!!!!!!!!!!!! 1st CodeLLM response code:\n' + code, file=print_file)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n", file=print_file)
        
        # Initialize default values
        question_quality = '0'
        answer = ''
        
        # ============================================================================
        # ROUND 2: QUESTION EVALUATION AND ANSWER GENERATION
        # ============================================================================
        if code == '':
            # No code was generated, so the response contains questions
            # Use LLM-based evaluator to:
            # 1) Generate answers to the clarifying questions
            # 2) Evaluate the quality of the clarifying questions
            # 3) Prepare for final code generation
            
            if args.log_phase_output >= 3:
                # Use cached evaluation results for phase-based evaluation
                answer = cached_answer
                question_quality = cached_qq
            else:
                # Perform live evaluation of questions
                answer, question_quality = evaluate_clarifying_questions(original_prompt, response, user_input, args.eval_protocol)
            
            if args.log_phase_output == 2:
                # Only return evaluation results, skip final code generation
                ans_list.append(answer)
                qq_list.append(question_quality)
                continue

            # ============================================================================
            # ROUND 3: FINAL CODE GENERATION WITH Q&A CONTEXT
            # ============================================================================
            if model == "AgentCoder":
                print("This is the original message:", messages)
                # Special handling for AgentCoder: combine all context into a single prompt
                new_prompt = "Original Question: " + original_prompt + " First Response: " + response + " Feedback: " + answer + " " + PROMPT_2ND_ROUND
                messages[-1]["prompt"] = new_prompt
                for message in messages:
                    message['clarity_prompt'] = ""
                msgs_i = messages.copy()
            else:
                # Standard multi-turn conversation format
                msgs_i = messages.copy()
                msgs_i.append({"role":"assistant","content": response})
                msgs_i.append({"role":"user","content": answer + PROMPT_2ND_ROUND})
            
            # Generate final code with full conversation context
            response_2nd = generate_response(model_2nd_round, msgs_i, 1, temperature, args, open_source_model, tokenizer)
            code = response_2_code(response_2nd[0])
            
            print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=print_file)
            print('!!!!!!!!!!!!! 3rd CodeLLM input messages:\n', msgs_i, file=print_file)
            print('!!!!!!!!!!!!! 3rd CodeLLM response:\n', response_2nd, file=print_file)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n", file=print_file)
        
        # Store results for this response
        qq_list.append(question_quality)
        code_list.append(code)
        ans_list.append(answer)
    
    return response_list, code_list, qq_list, ans_list




