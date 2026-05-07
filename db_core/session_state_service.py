import json
from sqlmodel import select
from db_core.config import get_session
from db_core.models import SessionState


def get_session_state(session_id: str):
    with get_session() as db:
        state = db.exec(
            select(SessionState)
            .where(SessionState.session_id == session_id)
        ).first()

        return state

def set_session_state(
    session_id: str,
    active_task: str,
    instructions: dict
):
    with get_session() as db:

        state = db.exec(
            select(SessionState)
            .where(SessionState.session_id == session_id)
        ).first()

        if not state:
            state = SessionState(
                session_id=session_id
            )

        state.active_task = active_task
        state.last_instructions = json.dumps(instructions)

        db.add(state)
        db.commit()