from google.adk.models.lite_llm import LiteLlm as _LiteLlm

if not getattr(_LiteLlm, "_llm_client_dump_patched", False):
    _original_model_dump = _LiteLlm.model_dump

    def _patched_model_dump(self, **kwargs):
        exclude = kwargs.get("exclude")
        if exclude is None:
            kwargs["exclude"] = {"llm_client"}
        elif isinstance(exclude, set):
            kwargs["exclude"] = exclude | {"llm_client"}
        elif isinstance(exclude, dict):
            kwargs["exclude"] = {**exclude, "llm_client": True}
        return _original_model_dump(self, **kwargs)

    _LiteLlm.model_dump = _patched_model_dump
    _LiteLlm._llm_client_dump_patched = True

from . import agent
