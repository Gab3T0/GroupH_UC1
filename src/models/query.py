from dataclasses import dataclass, field


@dataclass
class Query:
    question_text: str
    query_type: str = field(default="unknown")
