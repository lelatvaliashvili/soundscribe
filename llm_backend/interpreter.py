import json
import logging
import re
from dotenv import load_dotenv
from models.intents import UnifiedIntent
from .llm_client import ask_llm, ask_llm_structured
from .prompts import (
    CLASSIFY_PROMPT,
    FEEDBACK_PROMPT,
    INCREMENTAL_UPDATES_PROMPT,
    CLARIFICATION_PROMPT,
    USER_PROMPT_UNSUPPORTED_STEM
)
import numpy as np
from llm_backend.parser import parse_intent_response
from config.audio_config import VALID_STEMS

load_dotenv()
logger = logging.getLogger(__name__)


def extract_stem_list(prompt: str) -> list[str]:
    logger.info(f"prompt: {prompt}")
    instruction = (
        "From the following user request, extract only the valid stems (vocals, drums, bass, other). "
        "Return them as a comma-separated list and nothing else."
    )

    try:

        content = ask_llm_structured(CLASSIFY_PROMPT,
                                     prompt,
                                     UnifiedIntent, #RemixIntent,
                                     temperature=0)

        logger.info(f"Model response: {content}")

        valid_stems = [s.strip() for s in content.lower().split(",") if s.strip() in VALID_STEMS]
        logger.info(f"Filtered stems: {valid_stems}")
        return valid_stems
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return []

def classify_prompt(prompt: str) -> dict:
    response = ask_llm(CLASSIFY_PROMPT, prompt, temperature=0)
    logger.info(f"Raw response: {response}")

    try:
        parsed = parse_intent_response(response)
        logger.info(f"Parsed intent: {parsed}")
        logger.info(f"Intent class: {type(parsed)}")
        return parsed
    except json.JSONDecodeError:
        return {"type": "separation", "stems": []}  # Fallback

def parse_feedback(feedback_text: str) -> dict:

    response = ask_llm(FEEDBACK_PROMPT, feedback_text, 0)
    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL) #extracts JSON like block that starts with {.  re.DOTALL - matches newline characters
        if json_match:
            #output becomes intent.type
            return parse_intent_response(response) #json.loads(json_match.group())
        return {}
    except:
        return {} #TODO: needs more robust handling

def describe_audio_edit(task_type: str, instructions: dict = None, extracted_stems: list[str] = None) -> str:
    system_prompt = """
    You are a friendly music producer describing what you just did with the audio.

    Guidelines:
    - If task is "separation", describe which stems were extracted.
    - If task is "remix", describe only meaningful DSP adjustments.
    - Keep responses short (1–2 sentences).
    - Speak naturally like a music producer.
    """

    user_prompt = json.dumps({
        "task_type": task_type,
        "instructions": instructions,
        "extracted_stems": extracted_stems
    }) #JSON formatted string due to json.dumps

    response = ask_llm(system_prompt, user_prompt, temperature=0.5)

    return response.strip()

def describe_feedback_changes(feedback_text: str, old_instructions: dict, new_instructions: dict) -> str:
    """
    Generate a natural description of only the changes made in response to user feedback.
    This creates incremental descriptions rather than describing the entire remix state.
    """

    user_prompt = {
        "feedback_text": feedback_text,
        "old_instructions": old_instructions,
        "new_instructions": new_instructions
    }

    user_prompt = json.dumps({
        "feedback_text": feedback_text,
        "old_instructions": old_instructions,
        "new_instructions": new_instructions
    })

    response = ask_llm(INCREMENTAL_UPDATES_PROMPT, user_prompt, temperature=0.3)

    return response.strip()

def generate_clarification_response(reason: str, user_message: str, has_audio: bool) -> str:

    context = {
        "user_message": user_message,
        "reason": reason,
        "has_audio": has_audio
    }

    if reason == "unsupported_stem":
        user_prompt = json.dumps({
            "reason": reason,
            "user_message": user_message,
            "has_audio": has_audio
        })

        response = ask_llm(CLARIFICATION_PROMPT, USER_PROMPT_UNSUPPORTED_STEM, temperature=0.7)

        return response.strip()

    user_prompt = f"""
        User said: "{user_message}"
        Reason for clarification: {reason}
        User has audio uploaded: {has_audio}
    
        Respond naturally as a music producer friend would. Guide them toward separation or remixing.
    """

    user_prompt = json.dumps({
        "reason": reason,
        "user_message": user_message,
        "has_audio": has_audio
    })

    response = ask_llm(CLARIFICATION_PROMPT, user_prompt, temperature=0.7)

    return response.strip()


def apply_feedback_to_instructions(feedback_adjustments: dict, last_instructions: dict) -> dict:
    """
    Apply parsed feedback adjustments to the last instructions.
    Works with the output of parse_feedback() to create updated instructions.
    """
    updated = last_instructions.copy()

    if "volumes" not in updated:
        updated["volumes"] = {"vocals": 1.0, "drums": 1.0, "bass": 1.0, "other": 1.0}

    # Apply volume changes
    volume_map = {
        "slightly softer": -0.1,
        "softer": -0.3,
        "much softer": -0.6,
        "mute": -1.0,
        "slightly louder": +0.1,
        "louder": +0.3,
        "much louder": +0.6
    }

    for stem, change in feedback_adjustments.get("volumes", {}).items():
        if stem in updated["volumes"]:
            delta = volume_map.get(change, 0.0)
            updated["volumes"][stem] = np.clip(updated["volumes"][stem] + delta, 0.0, 2.0)

    for stem, change in feedback_adjustments.get("reverb", {}).items():
        if "reverb" not in updated:
            updated["reverb"] = {}
        current_reverb = updated["reverb"].get(stem, 0.0)
        if change == "more":
            updated["reverb"][stem] = min(current_reverb + 0.2, 1.0)
        elif change == "less":
            updated["reverb"][stem] = max(current_reverb - 0.2, 0.0)

    for stem, change in feedback_adjustments.get("pitch_shift", {}).items():
        if "pitch_shift" not in updated:
            updated["pitch_shift"] = {}
        current_pitch = updated["pitch_shift"].get(stem, 0)
        try:
            delta = int(change.replace("+", ""))
            updated["pitch_shift"][stem] = current_pitch + delta
        except:
            pass

    for stem, level in feedback_adjustments.get("compression", {}).items():
        if "compression" not in updated:
            updated["compression"] = {}
        updated["compression"][stem] = level

    return updated