import logging
import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile, Form, APIRouter, HTTPException
from pydub import AudioSegment
from llm_backend.session_manager import save_file_to_db
from db_core.session import ensure_session_exists
from db_core.config import get_session

router = APIRouter()
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("separated")
ALLOWED_AUDIO_EXTENSIONS = {
    ".aac",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".wav",
}
ALLOWED_CONTENT_TYPES = {
    "audio/aac",
    "audio/aiff",
    "audio/flac",
    "audio/m4a",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/ogg",
    "audio/wav",
    "audio/wave",
    "audio/x-aiff",
    "audio/x-m4a",
    "audio/x-wav",
    "application/octet-stream",
}


def validate_upload_request(file: UploadFile, session_id: str, user_id: str) -> str:
    if not session_id.strip():
        raise HTTPException(status_code=400, detail="Session ID is required")
    if not user_id.strip():
        raise HTTPException(status_code=400, detail="User ID is required")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Audio file is required")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported audio file type")

    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported audio content type")

    return suffix


@router.post("/upload")
async def upload(file: UploadFile, session_id: str = Form(...), user_id: str = Form(...)):
    suffix = validate_upload_request(file, session_id, user_id)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    upload_id = uuid.uuid4().hex
    original_path = UPLOAD_DIR / f"{upload_id}_original{suffix}"
    converted_path = UPLOAD_DIR / f"{upload_id}_converted.wav"

    try:
        with original_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if original_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        audio = AudioSegment.from_file(original_path)
        audio.export(converted_path, format="wav")

        with get_session() as db:
            ensure_session_exists(db, session_id, user_id)

        converted_path_str = str(converted_path)
        save_file_to_db(session_id, file_type="uploaded",  path=converted_path_str, stem=None)
        logger.info(
            "Stored uploaded audio for session %s by user %s at %s",
            session_id,
            user_id,
            converted_path_str,
        )

    except HTTPException:
        if converted_path.exists():
            converted_path.unlink()
        if original_path.exists():
            original_path.unlink()
        raise
    except Exception as exc:
        logger.exception("Failed to process upload for session %s by user %s", session_id, user_id)
        if converted_path.exists():
            converted_path.unlink()
        if original_path.exists():
            original_path.unlink()
        raise HTTPException(status_code=400, detail="Could not process uploaded audio") from exc

    return {
        "message": "File uploaded and converted to WAV",
        "session_id": session_id,
        "user_id": user_id,
        "converted_path": converted_path_str
    }
