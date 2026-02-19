import re

from src.catalog.course_catalog import CourseCatalog
from src.chain.query_handler import QueryHandler
from src.models.answer import Answer
from src.models.query import Query


class CourseIDHandler(QueryHandler):
    """First handler in the chain. Checks if the query contains a course ID
    pattern (e.g., CSE1325, CSE 4361) and performs a direct catalog lookup."""

    def __init__(self, course_catalog: CourseCatalog):
        super().__init__()
        self.course_catalog: CourseCatalog = course_catalog
        self.course_id_pattern: str = r"[A-Za-z]{2,4}\s?\d{4}"

    def handle_request(self, query: Query) -> Answer:
        """Check for a course ID in the query. If found and valid, return
        the course details. Otherwise, pass to the next handler."""
        match = re.search(self.course_id_pattern, query.question_text)

        if match:
            raw_id = match.group(0).upper().replace(" ", "")
            if self.is_valid_id(raw_id):
                query.query_type = "course_id"
                course = self.course_catalog.find_by_id(raw_id)
                return Answer(
                    generated_text=(
                        f"[Course ID Lookup]\n"
                        f"Course: {course.course_id}\n"
                        f"Title: {course.title}\n"
                        f"Description: {course.description}"
                    )
                )

        # Pass to next handler if no valid course ID found
        if self.next_handler:
            return self.next_handler.handle_request(query)

        return Answer(
            generated_text="Sorry, I could not find information for that query."
        )

    def is_valid_id(self, course_id: str) -> bool:
        """Validate that the extracted ID exists in the catalog."""
        return self.course_catalog.find_by_id(course_id) is not None
