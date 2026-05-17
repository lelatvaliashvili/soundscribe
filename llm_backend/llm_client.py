import ollama
from pydantic import schema
import logging

from models.intents import RemixIntent

logger = logging.getLogger(__name__)


def ask_llm(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    logger.info("Calling Ollama...")

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={
            "temperature": temperature
        }
    )
    logger.info("Ollama response received")
    return response["message"]["content"]

#exact json schema contract instead of valid json request
def ask_llm_structured(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    response = ollama.chat(
        model="llama3.1:8b",

        format=RemixIntent.model_json_schema(),

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )
    return schema.model_validate_json(
        response["message"]["content"]
    )