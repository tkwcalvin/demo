from dataclasses import dataclass

@dataclass
class Cache:
    cached_names: set[str]
    cached_responses: dict[str, str]
    cached_qqs: dict[str, str]
    cached_answers: dict[str, str]
    cached_file_path: str

