from dataclasses import dataclass


@dataclass
class LogConfig:
    log_phase_input: int
    log_phase_output: int
    print_file: str
    log_file: str
