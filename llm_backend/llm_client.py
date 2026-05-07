import ollama
from pydantic import schema

from models.intents import RemixIntent


def ask_llm(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
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