import ollama


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
