from dataclasses import dataclass, field, asdict
import json
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
    pseudo_code: Optional[str] = None
    datetime_pseudo_code: Optional[datetime] = None
    files_available: List[str] = field(default_factory=list)
    files_defined: List[str] = field(default_factory=list)

@dataclass
class JudgeResult:
    evaluation: Optional[float] = None
    metric: Optional[float] = None
    is_correct: Optional[bool] = None
    occurred_error: Optional[bool] = None
    judge_response: Optional[str] = None
    datetime_start: Optional[datetime] = None
    datetime_end: Optional[datetime] = None


@dataclass
class ExecutionRag:
    question: Optional[str] = None
    response: Optional[str] = None
    datetime_start: Optional[datetime] = None
    datetime_end: Optional[datetime] = None
    context: Optional[any] = None
    returned: Optional[bool] = None
    attempt: Optional[int] = None
    type_question: Optional[Redirect] = None
    files_used: FilesInRag = field(default_factory=FilesInRag)
    judge_result: JudgeResult = field(default_factory=JudgeResult)


@dataclass
class Executions:
    question: Optional[str] = None
    datetime_start: Optional[datetime] = None
    datetime_end: Optional[datetime] = None
    default_return: Optional[bool] = None
    response: Optional[str] = None
    executions_rag: List[ExecutionRag] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)

    def to_json(self):
        return json.dumps(asdict(self), default=str, indent=4)
