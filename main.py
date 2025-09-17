from fastapi import FastAPI, HTTPException, Query
from transcribe import transcribe_audio  # still sync
from summary_huggingface import summarize_text_async
from summary_gemini import summarize_with_gemini  

# pip install google-generativeai


app = FastAPI()

@app.get("/transcribe")
def transcribe_endpoint(audio_url: str):
    try:
        transcript = transcribe_audio(audio_url)
        return {"transcript": transcript}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe-summarize-huggingface")
async def summarize_huggingface(audio_url: str = Query(...)):
    try:
        # Transcription is still synchronous
        transcript = transcribe_audio(audio_url)

        # Summary is async
        summary = await summarize_text_async(transcript)

        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# Gemini summarization (async-safe)
@app.post("/transcribe-summarize-gemini")
async def summarize_gemini(audio_url: str = Query(...)):
    try:
        transcript = transcribe_audio(audio_url)
        summary = await summarize_with_gemini(transcript)  # Async Gemini
        return { "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

   

 #audio_url = "https://assembly.ai/wildfires.mp3"

# audio_url= "https://d386pzqmzseb1g.cloudfront.net/66423e558bea2ea2cb367984-recordings/1758016800106-71b2aa70-4a38-45e6-86a9-12fc80c55ee3.webm"


