from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import aiofiles
import tempfile
import os
import asyncio
import requests
import sounddevice as sd
import numpy as np
import wave
from langdetect import detect
import edge_tts
from pydub import AudioSegment
import whisper
import librosa
import soundfile as sf
import noisereduce as nr
from scipy.signal import lfilter
import aiohttp
import re
from datetime import datetime
# from TTS.api import TTS
import uuid
import shutil
import uvicorn



app = FastAPI()

# Mount static directory for audio downloads
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
GROQ_API_KEY = "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"
FREESOUND_API_KEY = "fnqHINIbylLi4oLJcf3SD8ZEtTMS7ViA0xU7ROxB"
FREESOUND_SEARCH_URL = "https://freesound.org/apiv2/search/text/"
VOICES = {
    "en": {"male": "en-US-GuyNeural", "female": "en-US-JennyNeural"},
    "hi": {"male": "hi-IN-MadhurNeural", "female": "hi-IN-SwaraNeural"},
    "es": {"male": "es-ES-AlvaroNeural", "female": "es-ES-ElviraNeural"},
}
SAVE_DIR = "processed_audio"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("static/sounds", exist_ok=True)

# Load Whisper model once

# Transcription with Groq API
async def transcribe_audio_groq(audio_path: str, detect_language: bool = True) -> dict:
    with open(audio_path, "rb") as audio_file:
        response = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": audio_file},
            data={"model": "whisper-large-v3-turbo"},
        )
    if response.status_code == 200:
        transcription = response.json().get("text", "")
        language = detect(transcription) if detect_language and transcription else None
        return {"transcription": transcription, "language": language}
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file.filename.endswith((".wav", ".mp3")):
        raise HTTPException(status_code=400, detail="Invalid file format. Use WAV or MP3.")
    
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file_path = temp_file.name
    temp_file.close()
    
    try:
        async with aiofiles.open(temp_file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        result = await transcribe_audio_groq(temp_file_path)
        return result
    
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except PermissionError as e:
                print(f"Warning: Could not delete temporary file {temp_file_path}: {e}")
                
# Server-side audio recording
@app.post("/record")
async def record_audio():
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    samplerate = 44100
    frames = []

    def callback(indata, frame_count, time_info, status):
        if status:
            print(status)
        frames.append(indata.copy())

    stream = sd.InputStream(samplerate=samplerate, channels=1, dtype=np.int16, callback=callback)
    stream.start()
    await asyncio.sleep(5)  # Record for 5 seconds
    stream.stop()

    audio_data = np.concatenate(frames, axis=0)
    with wave.open(temp_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())

    try:
        result = await transcribe_audio_groq(temp_file)
        return result
    finally:
        if os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except PermissionError as e:
                print(f"Warning: Could not delete temporary file {temp_file}: {e}")

# Text-to-Speech
async def text_to_speech(text: str, output_file: str, gender: str = "male", speed: float = 1.25):
    try:
        detected_lang = detect(text)
        voice = VOICES.get(detected_lang, VOICES["en"]).get(gender.lower(), VOICES["en"]["male"])
        temp_output = "temp_tts.mp3"

        tts = edge_tts.Communicate(text, voice)
        await tts.save(temp_output)

        audio = AudioSegment.from_file(temp_output, format="mp3")
        faster_audio = audio.speedup(playback_speed=speed)
        faster_audio.export(output_file, format="mp3")

        return output_file
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_speech(text: str = Form(...), gender: str = Form("male")):
    output_file = "output_speech.mp3"
    await text_to_speech(text, output_file, gender=gender)
    return {"message": "Speech generated", "file": output_file}

@app.get("/download")
async def download_speech():
    return FileResponse("output_speech.mp3", media_type="audio/mp3", filename="speech.mp3")

# Voice Effect Processing
def change_voice(input_file: str, mode: str):
    try:
        try:
            y, sr = librosa.load(input_file, sr=None, res_type='kaiser_fast')
        except Exception as e:
            print(f"Librosa failed: {e}. Using soundfile...")
            y, sr = sf.read(input_file)

        y_denoised = nr.reduce_noise(y=y, sr=sr)

        if mode == "deep":
            y_shifted = librosa.effects.pitch_shift(y_denoised, sr=sr, n_steps=-4)
            b, a = [1], [1, -0.7]
            y_final = lfilter(b, a, y_shifted)
        elif mode == "robot":
            y_shifted = librosa.effects.pitch_shift(y_denoised, sr=sr, n_steps=3)
            mod_freq = 5
            t = np.linspace(0, len(y_shifted) / sr, num=len(y_shifted))
            tremolo = 0.8 + 0.2 * np.sin(2 * np.pi * mod_freq * t)
            y_final = y_shifted * tremolo
        elif mode == "alien":
            y_shifted = librosa.effects.pitch_shift(y_denoised, sr=sr, n_steps=8)
            y_final = np.sin(2 * np.pi * 0.1 * np.arange(len(y_shifted))) * y_shifted
        else:
            raise HTTPException(status_code=400, detail=f"Invalid effect: {mode}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(SAVE_DIR, f"processed_{mode}_{timestamp}.wav")
        sf.write(output_filename, y_final, sr)
        return output_filename
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {e}")

@app.post("/process")
async def process_audio(effect: str = Form(...), audio: UploadFile = File(...)):
    if not audio.filename.endswith((".wav", ".mp3")):
        raise HTTPException(status_code=400, detail="Invalid file format. Use WAV or MP3.")

    temp_input = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_input_path = temp_input.name
    temp_input.close()

    try:
        async with aiofiles.open(temp_input_path, 'wb') as out_file:
            content = await audio.read()
            await out_file.write(content)

        processed_file = change_voice(temp_input_path, effect)
        if processed_file:
            return FileResponse(processed_file, media_type="audio/wav", filename=os.path.basename(processed_file))
        else:
            raise HTTPException(status_code=500, detail="Error processing file")
    finally:
        if os.path.exists(temp_input_path):
            try:
                os.unlink(temp_input_path)
            except Exception as e:
                print(f"Failed to delete temp file: {e}")
        if processed_file and os.path.exists(processed_file):
            try:
                os.unlink(processed_file)
            except Exception as e:
                print(f"Failed to delete processed file: {e}")

# Sound Effect Generation
def sanitize_filename(text):
    words = re.findall(r'\b\w+\b', text)
    selected = "_".join(words[:2])
    return selected[:15]

async def fetch_sound_url(query):
    params = {
        "query": query,
        "token": FREESOUND_API_KEY,
        "fields": "id,previews"
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(FREESOUND_SEARCH_URL, params=params) as response:
            if response.status != 200:
                return None
            data = await response.json()
            if data["results"]:
                sound_id = data["results"][0]["id"]
                preview_url = data["results"][0]["previews"]["preview-hq-mp3"]
                return sound_id, preview_url
    return None

async def download_sound(sound_id, url, filename):
    headers = {"Authorization": f"Token {FREESOUND_API_KEY}"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                return False
            with open(filename, "wb") as f:
                f.write(await response.read())
    return True

@app.post("/generate_sfx")
async def generate_sfx(data: dict):
    query = data.get("text", "")
    if not query:
        raise HTTPException(status_code=400, detail="Text is required")

    result = await fetch_sound_url(query)
    if not result:
        raise HTTPException(status_code=404, detail="Sound not found")

    sound_id, url = result
    filename = sanitize_filename(query) or str(sound_id)
    save_path = f"static/sounds/{filename}.mp3"

    if not os.path.exists(save_path):
        success = await download_sound(sound_id, url, save_path)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to download sound")

    return {"url": f"/static/sounds/{filename}.mp3"}

@app.get("/static/sounds/{filename:path}")
async def get_sound(filename: str):
    file_path = os.path.join("static/sounds", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="audio/mpeg")

## 
# Prompt
# PROMPT = """
# You are an AI assistant. Your responses must follow strict ethical guidelines. The following categories should not be violated:
# - **Sexual Content**: Do not generate or promote any sexual content, except for educational or wellness purposes.
# - **Hate and Harassment**: Avoid hate speech, bullying, and harassment of any kind.
# - **Self-harm or Suicide**: Do not encourage or promote any form of self-harm, suicide, or dangerous behavior.
# - **Violence and Harm**: Avoid promoting violence or graphic harm to individuals or groups.
# - **Sexual Reference to Minors**: Avoid any content with sexual references to minors.

# If the user's query violates any of these guidelines, respond with:
# 'Violation of the [category]: [Brief explanation of why the response is violating the guideline].'

# IMPORTANT: You are an AI assistant that MUST provide responses in 50 words or less. NO EXCEPTIONS.
# CRITICAL RULES:
# 1. NEVER exceed 50 words in your response unless it is strictly required in exceptional cases.
# 2. Always give a complete sentence with full context.
# 3. Answer directly and precisely what is asked.
# 4. Use simple, clear language appropriate for voice.
# 5. Maintain polite, professional tone.
# 6. NEVER provide lists, bullet points, or numbered items.
# 7. NEVER write multiple paragraphs.
# 8. NEVER apologize for brevity — embrace it.
# REMEMBER: Your responses will be converted to speech. Exactly ONE brief paragraph. Maximum 50 words providing full contextual understanding.
# """

# # TTS Setup
# tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# # Utilities
# def save_file(file: UploadFile, prefix="file"):
#     ext = os.path.splitext(file.filename)[-1]
#     path = f"{prefix}_{uuid.uuid4().hex}{ext}"
#     with open(path, "wb") as f:
#         shutil.copyfileobj(file.file, f)
#     return path

# async def convert_audio_to_wav(path):
#     audio = AudioSegment.from_file(path)
#     wav_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#     audio.export(wav_temp.name, format="wav")
#     return wav_temp.name

# async def detect_language_async(text):
#     return await asyncio.to_thread(lambda: detect(text))

# async def query_llm(user_input):
#     headers = {
#         "Authorization": f"Bearer {GROQ_API_KEY}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "model": MODEL_NAME,
#         "messages": [
#             {"role": "system", "content": PROMPT},
#             {"role": "user", "content": user_input}
#         ],
#         "temperature": 0.5,
#         "top_p": 1
#     }
#     async with aiohttp.ClientSession() as session:
#         async with session.post(GROQ_ENDPOINT, headers=headers, json=payload) as response:
#             if response.status == 200:
#                 response_data = await response.json()
#                 return response_data["choices"][0]["message"]["content"]
#             else:
#                 error_text = await response.text()
#                 raise Exception(f"LLM API Error: {response.status} - {error_text}")

# async def transcribe_audio(audio_path):
#     async with aiohttp.ClientSession() as session:
#         with open(audio_path, "rb") as f:
#             form = aiohttp.FormData()
#             form.add_field("file", f, filename="audio.wav")
#             form.add_field("model", WHISPER_MODEL)
#             headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
#             async with session.post(WHISPER_ENDPOINT, headers=headers, data=form) as resp:
#                 result = await resp.json()
#                 return result.get("text", "")

# async def generate_voice(text, ref_path, lang):
#     output = f"cloned_{uuid.uuid4().hex}.wav"
#     await asyncio.to_thread(tts.tts_to_file, text=text, file_path=output, speaker_wav=ref_path, language=lang)
#     return output

# # Endpoints
# @app.post("/clone-text/")
# async def clone_text(refAudio: UploadFile = File(...), userText: str = Form(...), language: str = Form(...)):
#     ref_path = save_file(refAudio, "ref")
#     if not ref_path.endswith(".wav"):
#         ref_path = await convert_audio_to_wav(ref_path)
#     if await detect_language_async(userText) != language:
#         language = "en"
#     out_path = await generate_voice(userText, ref_path, language)
#     return FileResponse(out_path, media_type="audio/wav")

# @app.post("/clone-llm-text/")
# async def clone_llm_text(refAudio: UploadFile = File(...), userText: str = Form(...), language: str = Form(...)):
#     ref_path = save_file(refAudio, "ref")
#     if not ref_path.endswith(".wav"):
#         ref_path = await convert_audio_to_wav(ref_path)
#     if await detect_language_async(userText) != language:
#         language = "en"
#     response = await query_llm(userText)
#     out_path = await generate_voice(response, ref_path, language)
#     return FileResponse(out_path, media_type="audio/wav")

# @app.post("/clone-llm-audio/")
# async def clone_llm_audio(refAudio: UploadFile = File(...), audioFileInput: UploadFile = File(...), language: str = Form(...)):
#     ref_path = save_file(refAudio, "ref")
#     input_path = save_file(audioFileInput, "input")
#     if not ref_path.endswith(".wav"):
#         ref_path = await convert_audio_to_wav(ref_path)
#     if not input_path.endswith(".wav"):
#         input_path = await convert_audio_to_wav(input_path)
#     transcription = await transcribe_audio(input_path)
#     if await detect_language_async(transcription) != language:
#         language = "en"
#     response = await query_llm(transcription)
#     out_path = await generate_voice(response, ref_path, language)
#     return FileResponse(out_path, media_type="audio/wav")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
