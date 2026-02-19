import json
import os
from typing import Dict, List, Optional

from src.models.course import Course


class CourseCatalog:
    """Loads courses from a JSONL file and provides lookup methods."""

    def __init__(self, source_file: str):
        self.source_file: str = source_file
        self.courses: Dict[str, Course] = {}
        self._load()

    def _load(self) -> None:
        """Read each line of the JSONL file, parse JSON, create Course objects."""
        if not os.path.exists(self.source_file):
            raise FileNotFoundError(f"Course catalog file not found: {self.source_file}")

        with open(self.source_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                course = Course(
                    course_id="CSE" + data["id"],
                    title=data["title"],
                    description=data["description"],
                )
                self.courses[course.course_id.upper()] = course

    def find_by_id(self, course_id: str) -> Optional[Course]:
        """Return Course if found, else None. Case-insensitive lookup."""
        return self.courses.get(course_id.upper().replace(" ", ""))

    def get_all_courses(self) -> List[Course]:
        """Return all courses."""
        return list(self.courses.values())
