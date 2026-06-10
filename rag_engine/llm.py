from typing import Any, Callable

from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import GOOGLE_API_KEY, MODEL_NAME

# Patch langchain_google_genai retry behavior so Gemini quota errors fail fast
try:
    import langchain_google_genai.chat_models as _chat_models
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
    import logging
    import google.api_core.exceptions as _gae

    def _create_retry_decorator() -> Callable[[Any], Any]:
        return retry(
            reraise=True,
            stop=stop_after_attempt(1),
            wait=wait_exponential(multiplier=1, min=1, max=2),
            retry=(
                retry_if_exception_type(_gae.ResourceExhausted)
                | retry_if_exception_type(_gae.ServiceUnavailable)
                | retry_if_exception_type(_gae.GoogleAPIError)
            ),
            before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        )

    _chat_models._create_retry_decorator = _create_retry_decorator
except Exception:
    pass

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)
