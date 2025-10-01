from dataclasses import dataclass



@dataclass
class CoreExpConfig:
    dataset: str
    model: str
    topn: int
    temperature: float
    option: str
    log_phase_input: int
    log_phase_output: int
    min_problem_idx: int
    max_num_problems: int
    dataset_loc: str
    phase1_prompt: str
    phase2_prompt: str

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
class ExpConfig:
    coreExpConfig: CoreExpConfig
    advancedConfig: AdvancedConfig
    testingConfig: TestingConfig






        