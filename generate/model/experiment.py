from dataclasses import dataclass

@dataclass
class datasetConfig:
    dataset: str
    dataset_loc: str
    min_problem_idx: int
    max_num_problems: int
    option: str

@dataclass
class modelConfig:
    model: str
    topn: int
    temperature: float

    
    
@dataclass
class promptConfig:
    phase1_prompt: str
    phase2_prompt: str

@dataclass
class CoreExpConfig:
    datasetConfig: datasetConfig
    modelConfig: modelConfig
    promptConfig: promptConfig
    log_phase_input: int
    log_phase_output: int
    

@dataclass
class AdvancedConfig:
    bootstrap_method: str
    resume_task_bs: int
    resume_task_run: int
    skip_bootstrap: bool
    version: str


@dataclass
class TestingConfig:
    do_test_only: bool
    do_save_model: bool
    pass_only: bool
    mask_func_name: bool


@dataclass
class PromptConfig:
    phase1_prompt: str
    phase2_prompt: str


@dataclass
class ExpConfig:
    coreExpConfig: CoreExpConfig
    advancedConfig: AdvancedConfig
    testingConfig: TestingConfig






        