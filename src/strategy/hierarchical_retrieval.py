from typing import List

from src.models.retrieved_context import RetrievedContext
from src.strategy.retrieval_strategy import RetrievalStrategy


class HierarchicalRetrieval(RetrievalStrategy):
    """Two-pass retrieval:
    1. Coarse pass: query against course-level summaries to narrow candidates.
    2. Fine pass: query against detail chunks only within candidate courses.
    """

    def __init__(self, coarse_n: int = 5, fine_n: int = 3):
        self.coarse_n = coarse_n
        self.fine_n = fine_n

    def retrieve(self, collection, query_text: str) -> List[RetrievedContext]:
        # Coarse pass: search only summary chunks (chunk_type == "summary")
        coarse_results = collection.query(
            query_texts=[query_text],
            n_results=self.coarse_n,
            where={"chunk_type": "summary"},
        )

        # Gather candidate course IDs from coarse results
        candidate_ids = []
        for meta in coarse_results["metadatas"][0]:
            cid = meta.get("course_id", "")
            if cid and cid not in candidate_ids:
                candidate_ids.append(cid)

        if not candidate_ids:
            return []

        # Fine pass: search detail chunks only within candidate courses
        fine_results = collection.query(
            query_texts=[query_text],
            n_results=self.fine_n,
            where={
                "$and": [
                    {"course_id": {"$in": candidate_ids}},
                    {"chunk_type": "detail"},
                ]
            },
        )

        contexts = []
        for doc, meta, dist in zip(
            fine_results["documents"][0],
            fine_results["metadatas"][0],
            fine_results["distances"][0],
        ):
            contexts.append(
                RetrievedContext(
                    context_text=doc,
                    source_course_id=meta.get("course_id", ""),
                    relevance_score=dist,
                )
            )

        return contexts
