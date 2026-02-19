from dataclasses import dataclass


@dataclass
class Course:
    course_id: str
    title: str
    description: str

    def __str__(self) -> str:
        return f"{self.course_id}: {self.title}\n{self.description}"
