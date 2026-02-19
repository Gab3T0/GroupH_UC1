from dataclasses import dataclass
from typing import List


@dataclass
class EmbeddingVector:
    vector_data: List[float]
