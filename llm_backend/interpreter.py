import json
import logging
import re
from dotenv import load_dotenv
from .llm_client import ask_llm
from .prompts import (
    EXTRACT_STEMS_PROMPT,
    CLASSIFY_PROMPT,
    FEEDBACK_PROMPT,
    INCREMENTAL_UPDATES_PROMPT,
    CLARIFICATION_PROMPT,
    USER_PROMPT_UNSUPPORTED_STEM
)
import numpy as np
from demucs.pretrained import get_model

load_dotenv()
logger = logging.getLogger(__name__)
VALID_STEMS = {"vocals", "drums", "bass", "other"} #TODO: move somewhere in configs

def extract_stem_list(prompt: str) -> list[str]:
    logger.info(f"prompt: {prompt}")
    instruction = (
        "From the following user request, extract only the valid stems (vocals, drums, bass, other). "
        "Return them as a comma-separated list and nothing else."
    )

    try:

        content = ask_llm(EXTRACT_STEMS_PROMPT,
                          f"{instruction}\n\nUser request: {prompt}",
                          temperature=0.0)
        logger.info(f"Model response: {content}")

        valid_stems = [s.strip() for s in content.lower().split(",") if s.strip() in VALID_STEMS]
        logger.info(f"Filtered stems: {valid_stems}")
        return valid_stems
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return []

def classify_prompt(prompt: str) -> dict:
    '''
    system_prompt = """
        You are a music assistant. Based on user prompt, classify it as one of:
        1. Separation - if user asks to extract stems like 'give me vocals and drums'
        2. Remix - if user gives volume/adjustment hints like 'make vocals louder, mellow the drums'
        3. Clarification - if user's intent is unclear, too general, or outside audio processing scope
        Return a strict JSON structure as specified below.

        For separation:
        {
            "type": "separation",
            "stems": ["vocals", "drums"]
        }

        For clarification (for cases when the user's intent is unclear):
        {
            "type": "clarification",
            "reason": "unclear_intent" | "general_question" | "out_of_scope"
        }

        - Valid stems are: vocals, drums, bass, other.

        Separation examples:
        - "give me vocals" → {"type": "separation", "stems": ["vocals"]}
        - "separate bass" → {"type": "separation", "stems": ["bass"]}
        - "lets separate bass" → {"type": "separation", "stems": ["bass"]}
        - "i want to hear bass only" → {"type": "separation", "stems": ["bass"]}
        - "isolate drums" → {"type": "separation", "stems": ["drums"]}
        - "extract vocals and drums" → {"type": "separation", "stems": ["vocals", "drums"]}
        - "separate other", "give me other", "isolate other" → {"type": "separation", "stems": ["other"]}
        - "give me that other category" → {"type": "separation", "stems": ["other"]}

        - If the user requests unsupported stems (e.g. trumpet, piano, guitar), return: {"type": "clarification", "reason": "unsupported_stem", "requested_stem": "trumpet"}
        - If none are valid, return: {"type": "separation", "stems": []}

        {
          "type": "remix",
          "instructions": {
            "volumes": {
              "vocals": 1.2,
              "drums": 0.7,
              "bass": 1.0,
              "other": 1.0
            },
            "reverb": {
              "vocals": 0.5
            },
            "global_reverb": 0.3,
            "pitch_shift": {
              "vocals": 2
            },
            "compression": {
              "vocals": "low"
            }
          }
        }

        Instructions:
        - **Instructions.volumes**: Always include all four stems with float multipliers (default 1.0 if not mentioned).
        - **Instructions.reverb**: ONLY include if user specifically mentions reverb for individual stems. Value between 0.0 (none) to 1.0 (max).
        - **Instructions.global_reverb**: ONLY include if user specifically mentions reverb for "whole mix" or "everything". Value between 0.0 (none) to 1.0 (max).
        - **Instructions.pitch_shift**: ONLY include if user specifically mentions pitch changes. Integer in semitones (+ for up, - for down).
        - **Instructions.compression**: ONLY include if user specifically mentions compression. Choose among the "low", "medium", "high".
        - **Instructions.eq**: ONLY include if user specifically mentions EQ/frequency adjustments.
        - **Instructions.filter**: ONLY include if user specifically mentions filters.

        Very Important: Do NOT add effects that the user didn't explicitly request. If the user only asks for reverb, do NOT add compression, pitch, or other effects.
        - Valid stems are: vocals, drums, bass, other. If the user asks for anything else (e.g., trumpet, guitar), return only valid stems and ignore the rest.
        - If the user requests an unsupported stem (e.g. "give me trumpet"), return: {"type": "separation", "stems": []}
        - For remixing, always include **all four stems** and only adjust volumes based on the prompt. If no volume is mentioned, use default 1.0.
        - If the user’s intent is unclear, default to {"type": "separation", "stems": []}
        For volumes:
            - If user says "slightly louder" or "a bit louder": set to 1.1
            - If user says "louder": set to 1.3
            - If user says "much louder": set to 1.6
            - If user says "extremely louder" or "max volume": set to 2.0
            - "slightly softer" or "a bit softer": 0.9
            - "softer": 0.7
            - "much softer": 0.5
            - "mute": 0.0
        For pitch_shift:
            - If user says "raise pitch by X semitones" or "increase pitch by X", set to +X.
            - If user says "lower pitch by X semitones" or "decrease pitch by X", set to -X.
            - If unspecified, set to 0.

        For reverb:
            - If user says "slight reverb" or "a bit of reverb": 0.2
            - If user says "reverb" or "add reverb": 0.5
            - If user says "heavy reverb" or "a lot of reverb": 0.8
            - If user says "maximum reverb" or "max reverb": 1.0
            - If user says "reverb to the whole mix" or "global reverb" or "reverb to everything": use "global_reverb" key instead
            - If unspecified, set to 0.0.

        For global_reverb (applies to final mix, not individual stems):
            - If user says "apply reverb to the whole mix" or "add reverb to everything": 0.5
            - If user says "make the whole mix more ambient" or "add reverb to make more ambient": 0.6
            - If user says "heavy reverb to the whole mix": 0.8
            - If user mentions "ambient", "spacious", "atmospheric" with reverb: use global_reverb
            - Use same scale as regular reverb (0.0-1.0)
        
        For compression:
            - If user says "light compression" or "slight compression": "low"
            - If user says "compression" or "add compression": "medium"
            - If user says "strong compression" or "heavy compression": "high"
            - If unspecified, set to "medium".
        
        For filter:
            - Detect user requests like "add a low-pass filter at 4kHz" or "apply high-pass at 120Hz".
            - Return JSON with "filter" key, e.g.:
              "filter": {
                "vocals": {
                  "type": "lowpass",
                  "cutoff": 4000
                }
              }
            
            For eq:
            - Parse phrases like "boost 3kHz by 5dB with width 1.0" or "cut 100 Hz by -3dB".
            - Return JSON with frequency (Hz), width (Q), and gain_db (dB).

        Very Important:: Use "clarification" type for unclear, general, or out-of-scope requests:
        - General questions: "What can you do?", "Help me", "How does this work?"
        - Vague requests: "Make it better", "Fix this", "Improve the sound"
        - Out of scope: "Write lyrics", "Compose music", "What's this song called?"
        - Unclear intent: "I love this song", "This is great", "Nice track"
        - Non-audio requests: "What's the weather?", "Tell me a joke"
        - Unsupported stems: "trumpet solo", "piano only", "guitar track", "saxophone", "synthesizer" - return {"type": "clarification", "reason": "unsupported_stem", "requested_instrument": "trumpet"}
        - Empty or nonsense: "", "asdfgh", "???"

        """
    '''
    response = ask_llm(CLASSIFY_PROMPT, prompt, temperature=0)

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"type": "separation", "stems": []}  # Fallback

def parse_feedback(feedback_text: str) -> dict:
    '''
    system_prompt = """
        You are a music DSP feedback interpreter assistant.
        Based on user feedback, extract intended adjustments in structured JSON.
        Return JSON like:

        {
          "volumes": {
            "vocals": "louder",
            "drums": "softer"
          },
          "reverb": {
            "vocals": "more",
            "drums": "less"
          },
          "pitch_shift": {
            "vocals": "+2",
            "drums": "-1"
          },
          "compression": {
            "vocals": "high",
            "drums": "low"
          }
        }

        Guidelines:
        - Only include stems that are explicitly mentioned in feedback.
        - For **volumes**, use one of: "slightly softer", "softer", "much softer", "mute", "slightly louder", "louder", "much louder".
        - For **pitch_shift**, return semitone adjustments with '+' or '-' (e.g. '+2' or '-1').
        - For **reverb**, use: "less", "more".
        - For **compression**, use: "low", "medium", "high".
        - If feedback does not mention an effect for a stem, omit that effect.
        - If nothing is detected, return an empty JSON object {}.

        Special pitch cases:
        - "raise pitch back up by 4 semitones" → {"pitch_shift": {"vocals": "+4"}}
        - "shift pitch 2 semitones more" → {"pitch_shift": {"vocals": "+2"}}
        - "shift the changes up by 2 semitones" → {"pitch_shift": {"vocals": "+2"}}
        - "can you shift the pitch of vocals 2 semitones more" → {"pitch_shift": {"vocals": "+2"}}
        - "pitch shift up by 2 semitones" → {"pitch_shift": {"vocals": "+2"}}
        - "pitch shift up again by 2 semitones" → {"pitch_shift": {"vocals": "+2"}}
        - "pitch shift up again by 2 semitones more" → {"pitch_shift": {"vocals": "+2"}}
        - "undo the pitch change" → {"pitch_shift": {"vocals": "0"}}

        Special cases:
        - "raise pitch back up by 4 semitones" → {"pitch_shift": {"vocals": "+4"}}
        - "undo the pitch change" → {"pitch_shift": {"vocals": "0"}}
        - "separate other" or "isolate other" → {"type": "separation", "stems": ["other"]}
        - "raise pitch back up by 4 semitones" → {"pitch_shift": {"vocals": "+4"}}
        - "2 semitones more", "pitch shift up again by 2 semitones" or "shift 2 semitones more" → {"pitch_shift": {"vocals": "+2"}}
        - "pitch it down more" → {"pitch_shift": {"vocals": "-2"}}
        - "make it even louder" → {"volumes": {"vocals": "louder"}} (if vocals was mentioned before)
        Return only valid JSON as above. No explanations.
    """

    response = ask_llm(system_prompt, feedback_text, 0)
    '''

    response = ask_llm(FEEDBACK_PROMPT, feedback_text, 0)
    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL) #extracts JSON like block that starts with {.  re.DOTALL - matches newline characters
        if json_match:
            return json.loads(json_match.group())
        return {}
    except:
        return {} #TODO: needs more robust handling

'''
def describe_audio_edit(task_type: str, instructions: dict = None, extracted_stems: list[str] = None) -> str:
    system_prompt = """
       You are a friendly music producer describing what you just did with the audio. Be natural and varied in your responses.

       Guidelines:
       - If task is "separation", describe which stems were extracted with enthusiasm and variety
       - If task is "remix", describe only the meaningful DSP adjustments applied (volume, pitch, reverb, compression, filters, EQ)
       - Use musical and user-friendly language
       - Ignore unchanged/default stems (e.g., volumes of 1.0 or effects not applied)
       - Keep it short and clear (1-2 sentences)
       - Don’t say "the user asked for" — speak as if you applied it
       - Vary your language - don't always say the same thing

       For separation, use varied, friendly phrases like:
       - "Here you go! I've separated the [stems] for you"
       - "Got your [stems] isolated and ready to download"
       - "Perfect! Pulled out the [stems] - ready when you are"
       - "There we go - your [stems] are separated and good to go"
       - "All done! Your [stems] are ready"
       - "Nice! I've extracted the [stems] for you"
       """

    user_prompt = {
        "task_type": task_type,
        "instructions": instructions,
        "extracted_stems": extracted_stems
    }

    response = ask_llm(system_prompt, user_prompt, temperature=0.5) #this should get dictionary of requested edits

    return response
'''
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