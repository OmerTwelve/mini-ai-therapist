# src/__init__.py
# Makes src a proper Python package so imports work as `from src.pipeline import ...`

from .data_prep   import prepare_data
from .pipeline    import MentalHealthPipeline, PipelineResult, SimilarStatement

__all__ = [
    "prepare_data",
    "MentalHealthPipeline",
    "PipelineResult",
    "SimilarStatement",
]
