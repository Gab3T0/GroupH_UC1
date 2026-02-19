from dataclasses import dataclass


@dataclass
class RetrievedContext:
    context_text: str
    source_course_id: str = ""
    relevance_score: float = 0.0
