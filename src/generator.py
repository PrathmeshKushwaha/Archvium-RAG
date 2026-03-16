"""
generator.py - Text generation via the Hugging Face Inference API.

Wraps huggingface_hub.InferenceClient with:
  • Exponential back-off for 503 (model loading) and 429 (rate limit) errors.
  • A strict prompt template that grounds answers in retrieved context.
  • Conversation history injection (last N turns).
"""

import logging
import time
from dataclasses import dataclass
import os

from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

_RETRYABLE_HTTP_CODES = {429, 503}

_SYSTEM_PROMPT = """\
You are a precise question-answering assistant. You will be given:
1. A set of CONTEXT passages retrieved from technical documents.
2. A conversation HISTORY (previous turns).
3. A USER QUESTION.

Your rules:
- Answer ONLY using information present in the CONTEXT passages below.
- If the answer is not present in the context, respond with:
  "I do not have enough information in the provided context to answer that question."
- Do NOT speculate, hallucinate, or use outside knowledge.
- Be concise and cite which part of the context supports your answer when helpful.
"""


@dataclass
class GenerationConfig:
    """Tunable generation parameters."""
    max_new_tokens: int = 512
    temperature: float = 0.1     # Low temp → deterministic, factual answers
    top_p: float = 0.9
    repetition_penalty: float = 1.1


class RAGGenerator:
    """
    Generates answers using a Hugging Face-hosted LLM.

    Args:
        model_id:     HF model repository id (e.g. "mistralai/Mistral-7B-Instruct-v0.2").
        hf_token:     HF API token.  If None, reads from HF_TOKEN env var.
        max_retries:  Maximum number of retry attempts on transient errors.
        base_delay:   Initial back-off delay in seconds (doubles each retry).
        gen_config:   Generation hyper-parameters.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        hf_token: str | None = None,
        max_retries: int = 5,
        base_delay: float = 2.0,
        gen_config: GenerationConfig | None = None,
    ) -> None:
        self._model_id = model_id
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._gen_config = gen_config or GenerationConfig()

        resolved_token = hf_token or os.getenv("HF_TOKEN")

        logger.info("Initialising HF InferenceClient for model '%s'.", model_id)
        self._client = InferenceClient(model=model_id, token=resolved_token)

    def generate(
        self,
        query: str,
        context_chunks: list[str],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Generate an answer grounded in *context_chunks*.

        Args:
            query:          The user's current question.
            context_chunks: Texts of the top-k reranked passages.
            history:        Previous (role, content) dicts from ChatSession
                            (up to last 3 turns).

        Returns:
            The assistant's answer as a plain string.
        """
        messages = self._build_messages(query, context_chunks, history or [])
        return self._call_with_retry(messages)

    def _build_messages(
        self,
        query: str,
        context_chunks: list[str],
        history: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """
        Assemble the messages list for the chat-completion endpoint.

        Structure:
          [system]  → grounding instructions
          [user]    → context passages
          [history] → last N turns (alternating user/assistant)
          [user]    → current question
        """
        # Format context
        MAX_CHARS_PER_CHUNK = 1500  # ~375 tokens at 4 chars/token
        context_chunks = [c[:MAX_CHARS_PER_CHUNK] for c in context_chunks]
        context_block = "\n\n---\n\n".join(
            f"[Passage {i + 1}]\n{text}" for i, text in enumerate(context_chunks)
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"CONTEXT:\n{context_block}\n\n"
                    "Use only the passages above to answer the question that follows."
                ),
            },
            {
                "role": "assistant",
                "content": "Understood. I will answer solely from the provided context.",
            },
        ]

        # Inject conversation history (last N turns)
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})

        # Current question
        messages.append({"role": "user", "content": query})

        return messages

    def _call_with_retry(self, messages: list[dict[str, str]]) -> str:
        """
        Call the HF chat-completion endpoint with exponential back-off.

        Retries on HTTP 429 (rate limit) and 503 (model still loading).
        Raises the last exception if all retries are exhausted.
        """
        delay = self._base_delay

        for attempt in range(1, self._max_retries + 1):
            try:
                logger.debug("HF API attempt %d/%d …", attempt, self._max_retries)

                response = self._client.chat_completion(
                    messages=messages,
                    max_tokens=self._gen_config.max_new_tokens,
                    temperature=self._gen_config.temperature,
                    top_p=self._gen_config.top_p,
                )

                answer: str = response.choices[0].message.content
                logger.debug("HF API responded on attempt %d.", attempt)
                return answer.strip()

            except HfHubHTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None

                if status_code == 400:
                    logger.error("HF API rejected prompt (likely too long): %s", exc)
                    return (
                        "I was unable to generate a response because the retrieved "
                        "context was too long. Try asking a more specific question."
                    )
                
                if status_code in _RETRYABLE_HTTP_CODES:
                    reason = "model still loading" if status_code == 503 else "rate limited"
                    if attempt < self._max_retries:
                        logger.warning(
                            "HF API %s (HTTP %s). Retrying in %.1fs (attempt %d/%d).",
                            reason, status_code, delay, attempt, self._max_retries,
                        )
                        time.sleep(delay)
                        delay *= 2   # Exponential back-off
                        continue
                    else:
                        logger.error(
                            "HF API %s after %d attempts – giving up.",
                            reason, self._max_retries,
                        )

                raise  # Non-retryable or retries exhausted

            except Exception as exc:  # noqa: BLE001
                logger.error("Unexpected error calling HF API: %s", exc)
                if attempt < self._max_retries:
                    logger.info("Retrying in %.1fs …", delay)
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise

        # Should be unreachable, but satisfies type checkers
        raise RuntimeError("All HF API retry attempts exhausted.")
