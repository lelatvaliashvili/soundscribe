import json
import re

from models.intents import  SeparationIntent, RemixIntent, ClarificationIntent

def parse_intent_response(response: str):

    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON found in response")

    data = json.loads(response)

    intent_type = data.get("type")

    if intent_type == "separation":
        return SeparationIntent(**data)

    if intent_type == "remix":
        return RemixIntent(**data)

    if intent_type == "clarification":
        return ClarificationIntent(**data)

    raise ValueError(f"Unknown intent type: {intent_type}")