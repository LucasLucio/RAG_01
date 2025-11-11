from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

@dataclass
class Redirect:
    datetime_start: Optional[datetime] = None
    datetime_end: Optional[datetime] = None
    final_type: Optional[str] = None
    define_type: Optional[str] = None

@dataclass
class FilesInRag:
    datetime_start: Optional[datetime] = None
    datetime_end: Optional[datetime] = None
    files_available: List[str] = field(default_factory=list)
    files_defined: List[str] = field(default_factory=list)
    pseudo_code: Optional[str] = None
    datetime_pseudo_code: Optional[datetime] = None

@dataclass
class ExecutionRag:
    question: Optional[str] = None
    response: Optional[str] = None
    datetime_start: Optional[datetime] = None
    datetime_end: Optional[datetime] = None
    context: Optional[any] = None
    files_used: FilesInRag = field(default_factory=FilesInRag)
    type_question: Redirect = field(default_factory=Redirect)

@dataclass
class JudgeResult:
    evaluation: Optional[float] = None
    metric: Optional[float] = None
    is_correct: Optional[bool] = None
    occurred_error: Optional[bool] = None
    judge_response: Optional[str] = None
    datetime_start: Optional[datetime] = None
    datetime_end: Optional[datetime] = None

