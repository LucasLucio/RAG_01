from dataclasses import dataclass, field
from typing import List

@dataclass
class Redirect:
    final_type: str
    define_type: str

@dataclass
class FilesInRag:
    files_available: List[str] = field(default_factory=list)
    files_defined: List[str] = field(default_factory=list)

@dataclass
class ExecutionRag:
    question: str
    response: str
    files_used: FilesInRag = field(default_factory=FilesInRag)
    context: any
    type_question: Redirect = field(default_factory=Redirect)

