from dataclasses import dataclass

@dataclass
class Cache:
    cached_names: set[str]
    cached_response: dict[str, str]
    cached_qq: dict[str, str]
    cached_answer: dict[str, str]

