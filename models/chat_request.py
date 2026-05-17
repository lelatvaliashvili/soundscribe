from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    user_id: str = Field(min_length=1)

# Pydantic models — they represent API input, not DB storage.
