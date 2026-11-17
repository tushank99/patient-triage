from backend.app.llm.assessor import assess_with_llm
from backend.app.services.triage_pipeline import LLMFn


def get_llm_fn() -> LLMFn:
    """Dependency so tests can swap in a fake LLM (app.dependency_overrides[get_llm_fn])."""
    return assess_with_llm
