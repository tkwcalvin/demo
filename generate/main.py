"""
HumanEval Code Generation Main Entry Point
==========================================

This module serves as the main entry point for running code generation experiments
on the HumanEval dataset using various Large Language Models (LLMs).

The system supports:
- Multiple model types (OpenAI GPT, CodeLlama, StarChat, DeepSeek, CodeQwen)
- Different experimental modes (original, modified prompts)
- Multi-round code generation with clarifying questions
- Phase-based evaluation (1st round response, 2nd round Q&A, 3rd round final code)

Usage:
    python main.py -d HumanEvalComm -m gpt-3.5-turbo -n 1 -t 1.0 -o original

Input:  HumanEval.jsonl benchmark dataset
Output: Log files in ./log/ directory with generated responses and code
"""

from process import test_codellama, HumanEval_experiment
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from model.experiment import *
from model.open_source_model import *
from model.log import *
from process import Experiment

def init():
    parser = argparse.ArgumentParser(
        description="HumanEval Code Generation Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            # Basic experiment with GPT-3.5
            python main.py -d HumanEvalComm -m gpt-3.5-turbo -n 1 -t 1.0 -o original
            
            # Multi-round experiment with phase tracking
            python main.py -d HumanEvalComm -m gpt-4 -n 3 -t 0.7 -o original -so 1
            
            # Test specific problems range
            python main.py -d HumanEvalComm -m gpt-3.5-turbo -n 1 -t 1.0 -o original -minp 10 -maxp 20
            """
    )
    
    # Core experiment parameters
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        choices=['APPS', 'code_contest', 'HumanEval', 'HumanEvalComm'],
        help="Choose the benchmark dataset to use for evaluation",
        required=True,
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        help="Model to use for code generation (OpenAI models: gpt-3.5-turbo, gpt-4, etc.; Open source: codellama/CodeLlama-7b-Instruct-hf, etc.)",
        required=True,
    )
    parser.add_argument(
        "-n", "--topn",
        type=int,
        help="Number of candidate solutions to generate per problem (beam search width)",
        required=True,
        default=1,
    )
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        help="Sampling temperature for text generation (0.0 = deterministic, 1.0 = more creative)",
        required=True,
        default=1.0,
    )
    parser.add_argument(
        "-o", "--option",
        type=str,
        help="Experimental mode: 'original' for standard prompts, 'randRemoveX' for randomly removed text, 'manualRemove' for manually modified prompts",
        required=True,
        default='original'
    )
    
    # Phase-based evaluation parameters (for multi-round experiments)
    parser.add_argument(
        "-s", "--log_phase_input",
        choices=[0,1,2,3],
        type=int,
        help="Input phase: 0=full pipeline, 1=1st round LLM response, 2=2nd round Q&A, 3=3rd round final code. Used for resuming experiments.",
        default=0
    )
    parser.add_argument(
        "-so", "--log_phase_output",
        choices=[0,1,2,3],
        type=int,
        help="Output phase: 0=full pipeline, 1=1st round LLM response, 2=2nd round Q&A, 3=3rd round final code. Used for phase-specific evaluation.",
        default=0
    )
    
    # Problem range filtering
    parser.add_argument(
        "-maxp", "--max_num_problems",
        type=int,
        help="Maximum number of problems to process (-1 = no limit, process all problems)",
        default=-1,
    )
    parser.add_argument(
        "-minp", "--min_problem_idx",
        type=int,
        help="Starting index of problems to process (-1 = start from beginning)",
        default=-1,
    )
    
    # ============================================================================
    # OPEN SOURCE MODEL PARAMETERS
    # ============================================================================
    # These parameters are used when working with open source models (CodeLlama, StarChat, etc.)
    
    # Model paths and configuration
    parser.add_argument('--model_name_or_path', type=str, help='HuggingFace model path or local path to the model')
    parser.add_argument('--saved_model_path', type=str, help='Local path to save the model files')
    parser.add_argument('--finetuned_model_path', type=str, help='Path to the fine-tuned model files (for PEFT models)')
    parser.add_argument('--hf_dir', type=str, help='Path to the HuggingFace cache directory')
    parser.add_argument('--input_path', type=str, help='Path to the input dataset file')
    parser.add_argument('--user_input', type=str, help='Custom user input for testing the model (debug mode)')
    parser.add_argument('--output_dir', type=str, help='Directory path for output files')
    
    # Generation parameters
    parser.add_argument('--chain_length', type=int, default=5, help='Number of steps in the Identity Chain method')
    parser.add_argument('--seq_length', type=int, default=8192, help='Maximum sequence length for input tokens')
    parser.add_argument('--gen_length', type=int, default=None, help='Maximum length for generated sequences')
    parser.add_argument('--do_sample', action='store_true', help='Enable sampling during generation (vs greedy decoding)')
    parser.add_argument('--top_k', type=int, default=0, help='Top-k sampling parameter (0 = disabled)')
    parser.add_argument('--top_p', type=float, default=1, help='Top-p (nucleus) sampling parameter')
    parser.add_argument('--num_return_sequences', type=int, default=1, help='Number of sequences to return per generation')
    parser.add_argument('--num_beams', type=int, default=1, help='Number of beams for beam search decoding')
    
    # Model optimization and quantization
    parser.add_argument('--use_int8', action='store_true', help='Use 8-bit quantization to reduce memory usage')
    parser.add_argument('--use_fp16', action='store_true', help='Use 16-bit floating point precision')
    parser.add_argument('--greedy_early_stop', action='store_true', help='Stop generation early when reaching fixed point')
    
    # Testing and debugging options
    parser.add_argument('--do_test_only', action='store_true', help='Run model in test mode only (no full experiment)')
    parser.add_argument('--do_save_model', action='store_true', help='Save the loaded model to specified directory')
    parser.add_argument('--pass_only', action='store_true', help='Pass input through without generation (debug mode)')
    parser.add_argument('--mask_func_name', action='store_true', help='Mask function names in prompts (experimental)')
    
    # Advanced experiment control
    parser.add_argument('--bootstrap_method', type=str, default='problem', help='Method for bootstrapping the experiment chain')
    parser.add_argument('--resume_task_bs', type=int, default=0, help='Task index to resume bootstrapping from')
    parser.add_argument('--resume_task_run', type=int, default=0, help='Task index to resume experiment from')
    parser.add_argument('--skip_bootstrap', action='store_true', help='Skip the bootstrap stage entirely')
    parser.add_argument('--version', type=str, default='v1', help='Version identifier for the experiment chain')
    
    # Prompt configuration (loaded from config.yaml)
    parser.add_argument('--phase1_prompt', type=str, default='prompt1', help='Prompt template for phase 1 (from config.yaml)')
    parser.add_argument('--phase2_prompt', type=str, default='prompt1', help='Prompt template for phase 2 (from config.yaml)')
    parser.add_argument("--eval_protocol", type=str, help="Evaluation protocol: 'llm_metric_v2' for LLM-based evaluation", default='')
    # ============================================================================
    # ARGUMENT PARSING AND MODEL INITIALIZATION
    # ============================================================================
    
    return parser




def load_model(args):
    model = None
    tokenizer = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('device: ', device)
    # ============================================================================
    # OPEN SOURCE MODEL LOADING
    # ============================================================================
    # Load open source models (CodeLlama, StarChat, DeepSeek, CodeQwen) when not in phase 2
    if ('Llama' in args.model 
        or args.model.startswith('starcoder')
        or args.model.startswith('deepseek')
        or args.model.startswith('CodeQwen')
        ) and args.log_phase_output != 2:
        
        # Set up HuggingFace cache directory for model storage
        HF_HOME = args.hf_dir
        offload_folder = "offload_folder"
        print("Loading model...")
        
        # Model loading with different precision options
        if args.do_save_model:
            # Standard model loading for saving purposes
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path      
            )
        elif args.use_int8:
            # 8-bit quantization for memory efficiency
            print("**********************************")
            print("**** Using 8-bit quantization ****")
            print("**********************************")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                load_in_8bit=True,
                device_map="auto",
                cache_dir=HF_HOME,
                offload_folder=offload_folder,     
            )
        elif args.use_fp16:
            # 16-bit floating point for faster inference
            print("**********************************")
            print("****** Using fp16 precision ******")
            print("**********************************")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                device_map="auto",
                torch_dtype=torch.float16,
                cache_dir=HF_HOME,
                offload_folder=offload_folder,     
            )
        else:
            # Default precision (usually fp32)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                device_map="auto",
                cache_dir=HF_HOME,
                offload_folder=offload_folder,            
            )
        
        # Load fine-tuned model if specified
        if 'finetuned' in args.model:
            model = PeftModel.from_pretrained(model, args.finetuned_model_path)

        # Multi-GPU support (currently disabled)
        # if torch.cuda.device_count() > 1:
        #     model = torch.nn.DataParallel(model)
        print('model device: ', model.device)

        # ============================================================================
        # TOKENIZER CONFIGURATION
        # ============================================================================
        # Configure tokenizer based on model type
        if (args.model.startswith('Meta-Llama')
            or args.model.startswith('deepseek')
            or args.model.startswith('CodeQwen')):
            # Models that require trust_remote_code=True
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path,
                trust_remote_code=True,
            )
        else:
            # Standard tokenizer configuration
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path,
                model_max_length=args.seq_length,
                # Note: Right-padding warning is expected for decoder-only models
                # but changing to left-padding doesn't resolve the issue
                padding_side="right",
                use_fast=False,
                trust_remote_code=True,
                cache_dir=HF_HOME,
                offload_folder=offload_folder,
            )

    return model, tokenizer


def load_configs(args):
    exp_config = ExpConfig(
        coreExpConfig=CoreExpConfig(
            dataset=args.dataset,
            model=args.model,
            topn=args.topn,
            temperature=args.temperature,
            option=args.option,
            log_phase_input=args.log_phase_input,
            log_phase_output=args.log_phase_output,
            min_problem_idx=args.min_problem_idx,
            max_num_problems=args.max_num_problems,
            dataset_loc=args.dataset_loc,
            phase1_prompt=args.phase1_prompt,
            phase2_prompt=args.phase2_prompt,
        ),
        advancedConfig=AdvancedConfig(
            bootstrap_method=args.bootstrap_method,
            resume_task_bs=args.resume_task_bs,
            resume_task_run=args.resume_task_run,
            skip_bootstrap=args.skip_bootstrap,
            version=args.version,
        ),
        testingConfig=TestingConfig(
            do_test_only=args.do_test_only,
            do_save_model=args.do_save_model,
            pass_only=args.pass_only,
            mask_func_name=args.mask_func_name,
        ),
    )
    open_source_model_config = openSourceModelConfig(
        pathConfig=pathConfig(
            model_name_or_path=args.model_name_or_path,
            saved_model_path=args.saved_model_path,
            finetuned_model_path=args.finetuned_model_path,
            hf_dir=args.hf_dir,
        ),
        generationConfig=generationConfig(
            chain_length=args.chain_length,
            seq_length=args.seq_length,
            gen_length=args.gen_length,
            do_sample=args.do_sample,
        ),
        optimizationConfig=optimizationConfig(
            use_int8=args.use_int8,
            use_fp16=args.use_fp16,
            greedy_early_stop=args.greedy_early_stop,
        ),
    )


    log_config = LogConfig(
        log_phase_input=args.log_phase_input,
        log_phase_output=args.log_phase_output,
    )

    

    return exp_config, open_source_model_config, log_config
    


if __name__ == "__main__":
    # ============================================================================
    # ARGUMENT PARSING SETUP
    # ============================================================================
    parser = init()
    args = parser.parse_args()
    model, tokenizer = load_model(args)
    exp_config, open_source_model_config, log_config = load_configs(args)
    
    # ============================================================================
    # EXECUTION MODE SELECTION
    # ============================================================================
    if args.do_test_only:
        # Test mode: Run a simple inference test with the loaded model
        print("Running model test mode...")
        test_codellama(tokenizer, model, args.user_input, args.seq_length)
    elif args.do_save_model:
        # Save mode: Save the loaded model and tokenizer to specified path
        print(f"Saving model and tokenizer to {args.saved_model_path}...")
        tokenizer.save_pretrained(args.saved_model_path)
        model.save_pretrained(args.saved_model_path)
    elif args.dataset.startswith('HumanEval'):
        # Main experiment mode: Run the full HumanEval experiment
        print(f"Starting HumanEval experiment with dataset: {args.dataset}")
        print(f"Model: {args.model}, Temperature: {args.temperature}, Top-N: {args.topn}")
        print(f"Option: {args.option}")
        HumanEval_experiment(
            exp_config.coreExpConfig,
            model, 
            tokenizer
        )
    else:
        print(f"Unsupported dataset: {args.dataset}")
        print("Supported datasets: HumanEval, HumanEvalComm, APPS, code_contest")



