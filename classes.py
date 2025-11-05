from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Redirect:
    final_type: Optional[str] = None
    define_type: Optional[str] = None

@dataclass
class FilesInRag:
    files_available: List[str] = field(default_factory=list)
    files_defined: List[str] = field(default_factory=list)

@dataclass
class ExecutionRag:
    question: Optional[str] = None
    response: Optional[str] = None
    context: Optional[any] = None
    files_used: FilesInRag = field(default_factory=FilesInRag)
    type_question: Redirect = field(default_factory=Redirect)

