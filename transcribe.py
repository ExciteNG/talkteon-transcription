import os
import time
import logging
import requests
from typing import Optional
from requests.exceptions import RequestException
from dotenv import load_dotenv

load_dotenv()

api_key=os.getenv("ola_assemblyai")

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transcribe_audio(
    audio_url: str,
    api_key: str = api_key,
    model: str = "universal",
    timeout_seconds: int = 180,
    max_retries: int = 5,
    retry_backoff: int = 2
) -> str:
    """
    Transcribes an audio or video file using AssemblyAI and returns the transcript.
    Includes retry logic, timeout, and logging for production use.
    """

    
    headers = {
        "authorization": api_key,
        "content-type": "application/json"
    }

    data = {
        "audio_url": audio_url,
        "speech_model": model
    }

    logger.info("Starting transcription request...")

    # Retry loop for starting the transcription
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post("https://api.assemblyai.com/v2/transcript", json=data, headers=headers, timeout=10)
            response.raise_for_status()
            break
        except RequestException as e:
            logger.warning(f"[Attempt {attempt}] Failed to start transcription: {e}")
            if attempt == max_retries:
                raise Exception("Max retries reached while starting transcription.")
            time.sleep(retry_backoff ** attempt)

    transcript_id = response.json().get('id')
    polling_url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

    logger.info(f"Polling for transcription completion (ID: {transcript_id})...")

    start_time = time.time()

    while True:
        try:
            poll_response = requests.get(polling_url, headers=headers, timeout=10)
            poll_response.raise_for_status()
        except RequestException as e:
            logger.error(f"Polling error: {e}")
            raise Exception(f"Error while polling transcription status: {e}")

        result = poll_response.json()
        status = result.get("status", "")

        if status == "completed":
            logger.info("Transcription completed successfully.")
            return result["text"]

        elif status == "error":
            logger.error(f"Transcription failed: {result.get('error')}")
            raise Exception(f"Transcription failed: {result.get('error')}")

        elif time.time() - start_time > timeout_seconds:
            logger.error("Polling timed out.")
            raise TimeoutError(f"Transcription polling timed out after {timeout_seconds} seconds")

        time.sleep(3)

