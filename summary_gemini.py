
import os
import google.generativeai as genai
from starlette.concurrency import run_in_threadpool
from dotenv import load_dotenv

load_dotenv()


genai.configure(api_key=os.getenv("ola_gemini"))


model = genai.GenerativeModel("gemini-2.5-flash")

def summarize_with_gemini_sync(text: str) -> str:
    prompt = f"""Summarize the following transcript, focusing on emotional tone, intent, and key messages:
{text}"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        raise RuntimeError(f"Gemini summarization failed: {e}")


async def summarize_with_gemini(text: str) -> str:
    return await run_in_threadpool(summarize_with_gemini_sync, text)
