from dataclasses import dataclass

@dataclass
class pathConfig:
    model_name_or_path: str
    saved_model_path: str
    finetuned_model_path: str
    hf_dir: str
    input_path: str
    user_input: str
    output_dir: str

@dataclass
class generationConfig:
    chain_length: int
    seq_length: int
    gen_length: int
    do_sample: bool
    top_k: int
    top_p: float
    num_return_sequences: int
    num_beams: int

@dataclass
class optimizationConfig:
    use_int8: bool
    use_fp16: bool
    greedy_early_stop: bool
    

@dataclass
class testingConfig:
    do_test_only: bool
    do_save_model: bool
    pass_only: bool
    mask_func_name: bool
    bootstrap_method: str
    resume_task_bs: int
    resume_task_run: int
    skip_bootstrap: bool
    version: str

@dataclass
class openSourceModelConfig:
    pathConfig: pathConfig
    generationConfig: generationConfig
    optimizationConfig: optimizationConfig
    testingConfig: testingConfig












    











        
        
        
        