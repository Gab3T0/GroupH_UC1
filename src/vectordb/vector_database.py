from typing import List, Optional

import chromadb

from src.models.course import Course
from src.models.retrieved_context import RetrievedContext
from src.strategy.retrieval_strategy import RetrievalStrategy


class VectorDatabase:
    """Wraps ChromaDB. Acts as the Context in the Strategy pattern.

    Stores course data as vectorized chunks and delegates retrieval
    to the currently selected RetrievalStrategy.
    """

    def __init__(self, storage_type: str = "in-memory"):
        self.storage_type: str = storage_type
        self.current_strategy: Optional[RetrievalStrategy] = None

        if storage_type == "persistent":
            self.client = chromadb.PersistentClient(path="./chroma_db")
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name="courses",
            metadata={"hnsw:space": "cosine"},
        )

    def set_strategy(self, strategy: RetrievalStrategy) -> None:
        """Set the retrieval strategy."""
        self.current_strategy = strategy

    def index_courses(self, courses: List[Course]) -> None:
        """Index all courses into ChromaDB.

        For each course, creates:
        - A summary chunk: "COURSE_ID: Title"
        - A detail chunk: full description
        ChromaDB's default embedding function handles vectorization.
        """
        documents = []
        metadatas = []
        ids = []

        for course in courses:
            # Summary chunk (for hierarchical retrieval)
            summary = f"{course.course_id}: {course.title}"
            documents.append(summary)
            metadatas.append({
                "course_id": course.course_id,
                "chunk_index": 0,
                "chunk_type": "summary",
            })
            ids.append(f"{course.course_id}_chunk_0")

            # Detail chunk (full description)
            detail = f"{course.course_id}: {course.title} - {course.description}"
            documents.append(detail)
            metadatas.append({
                "course_id": course.course_id,
                "chunk_index": 1,
                "chunk_type": "detail",
            })
            ids.append(f"{course.course_id}_chunk_1")

        # Upsert to handle re-indexing gracefully
        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def retrieve(self, query_text: str) -> List[RetrievedContext]:
        """Delegate retrieval to the current strategy."""
        if self.current_strategy is None:
            raise ValueError("No retrieval strategy set. Call set_strategy() first.")
        return self.current_strategy.retrieve(self.collection, query_text)

    def get_collection(self):
        """Expose the ChromaDB collection for direct access if needed."""
        return self.collection
