import os
import httpx

API_TOKEN = os.getenv("ola_huggingface")
if not API_TOKEN:
    raise EnvironmentError("Missing Hugging Face API token. Set ola_huggingface environment variable.")

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}


async def summarize_text_async(text: str, max_length: int = 500, min_length: int = 200) -> str:
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": max_length,
            "min_length": min_length,
            "do_sample": False
        }
    }

    timeout = httpx.Timeout(30.0, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(API_URL, headers=HEADERS, json=payload)
            response.raise_for_status()
            result = response.json()
            return result[0]['summary_text']
        except httpx.HTTPError as http_err:
            raise RuntimeError(f"Hugging Face API error: {http_err}") from http_err
        except (KeyError, IndexError) as parse_err:
            raise RuntimeError(f"Invalid response from Hugging Face API: {response.text}") from parse_err
