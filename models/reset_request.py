from pydantic import BaseModel, Field

class ResetRequest(BaseModel):
    session_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
