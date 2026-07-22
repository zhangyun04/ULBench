from ulbench.models.base import (
    MODEL_INFO_KEYS,
    ModelAdapter,
    check_adapter_contract,
)

__all__ = ["ModelAdapter", "MODEL_INFO_KEYS", "check_adapter_contract"]


def create_adapter(model_id: str, **kwargs):
    """Route a model ID to its adapter class.

    Heavy adapter modules import lazily so this stays cheap for tooling.
    """
    lowered = model_id.lower()
    if "internvl" in lowered:
        from ulbench.models.internvl import InternVLAdapter
        return InternVLAdapter(model_id, **kwargs)
    from ulbench.models.huggingface import HuggingFaceAdapter
    return HuggingFaceAdapter(model_id, **kwargs)
