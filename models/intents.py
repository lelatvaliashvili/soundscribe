from enum import Enum
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel #validation and typed data model library?

class IntentType(str, Enum):
    SEPARATION = "separation"
    REMIX = "remix"
    CLARIFICATION = "clarification"

class SeparationIntent(BaseModel):
    type: Literal["separation"] #Literal["value"] reinforces exact match
    stems: List[str]

class ClarificationIntent(BaseModel):
    type: Literal["clarification"]
    reason: str

class RemixInstructions(BaseModel):
    volumes: Optional[dict] = None
    reverb: Optional[dict] = None
    pitch_shift: Optional[dict] = None
    compression: Optional[dict] = None #does this do actual compression once i apply look into it

class RemixIntent(BaseModel):
    type: Literal["remix"]
    instructions: RemixInstructions


class UnifiedIntent(BaseModel):
    type: str
    stems: Optional[list[str]] = None
    instructions: Optional[RemixInstructions] = None
    reason: Optional[str] = None
