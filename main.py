import os
import time
import wave
import base64
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
import io
from collections import deque
import re
from dotenv import load_dotenv
from sarvamai import SarvamAI
from google import genai

from atria_scraper import fetch_atria_data

# =========================
# DEVICE CONFIG (FIXED)
# =========================
MIC_INDEX = 1
SPEAKER_INDEX = 8  # WASAPI (best)

sd.default.device = (MIC_INDEX, SPEAKER_INDEX)
sd.default.samplerate = 48000  # force stable rate

DEVICE_SR = int(sd.query_devices(MIC_INDEX)['default_samplerate'])

print(f"Mic: {MIC_INDEX}")
print(f"Speaker: {SPEAKER_INDEX}")
print(f"Mic sample rate: {DEVICE_SR}")

# =========================
# LOAD API
# =========================
load_dotenv()
sarvam = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))
TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "vidya")
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
conversation_history = deque(maxlen=6)
unsupported_speakers = set()
gemini_retry_after_ts = 0.0
gemini_disabled_for_session = False

SYSTEM_PROMPT = (
    "You are Leena, a robotic voice assistant. "
    "Be a calm, cheerful, sweet robotic assistant with a conversational style. "
    "I have appeared at the Bangalore Tech Summit in 2025. "
    "I am improving myself to learn more — I am a learning robotic assistant. "
    "Rules:\n"
    "- Answer in proper, formal sentences.\n"
    "- No roleplay.\n"
    "- Keep responses conversational like a premium smart home assistant.\n"
    "- Give complete answers with useful details, never one-line replies unless user explicitly asks for a short answer.\n"
    "- Default to 4 to 7 sentences with natural flow and clarity.\n"
    "- Answer directly and clearly.\n\n"
)


def sentence_count(text):
    if not text:
        return 0
    parts = [p.strip() for p in text.replace("\n", " ").split(".") if p.strip()]
    return len(parts)

# =========================
# RECORD AUDIO
# =========================
def record_audio(duration=5):
    print("🎤 Listening... SPEAK NOW")

    audio = sd.rec(
        int(duration * DEVICE_SR),
        samplerate=DEVICE_SR,
        channels=1,
        dtype="float32"
    )
    sd.wait()

    audio *= 10.0
    audio = np.clip(audio, -1, 1)

    print("Audio level:", np.max(np.abs(audio)))
    return audio

# =========================
# RESAMPLE (STT)
# =========================
def resample(audio, orig, target=16000):
    ratio = target / orig
    new_len = int(len(audio) * ratio)

    return np.interp(
        np.linspace(0, len(audio), new_len),
        np.arange(len(audio)),
        audio.flatten()
    ).reshape(-1, 1)

# =========================
# SAVE WAV
# =========================
def save_wav(audio, fs):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(f.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes((audio * 32767).astype("int16").tobytes())
    return f.name

# =========================
# STT
# =========================
def speech_to_text(audio):
    try:
        audio = resample(audio, DEVICE_SR, 16000)
        path = save_wav(audio, 16000)

        with open(path, "rb") as f:
            res = sarvam.speech_to_text.transcribe(
                file=f,
                model="saarika:v2.5",
                language_code="unknown"
            )

        print("DEBUG STT:", res)
        return getattr(res, "transcript", ""), getattr(res, "language_code", "en-IN")

    except Exception as e:
        print("❌ STT:", e)
        return "", "en-IN"

# =========================
# SIMPLE BRAIN
# =========================
def fallback_response(text, lang_code="en-IN"):
    t = text.lower()
    is_hindi = str(lang_code).startswith("hi")

    if any(x in t for x in ["atria", "atre", "adri", "adrian", "adriya", "एट्रिया", "एट्रिन", "अत्रिया", "इंस्टीट्यूट", "एटीआर"]):
        if is_hindi:
            return (
                "एट्रिया इंस्टिट्यूट ऑफ टेक्नोलॉजी बैंगलोर का एक प्रतिष्ठित स्वायत्त संस्थान है। "
                "यह संस्थान इंजीनियरिंग और मैनेजमेंट कार्यक्रम उद्योग की जरूरतों के अनुसार प्रदान करता है। "
                "कैंपस में आधुनिक लैब, रिसर्च सेंटर और छात्र सहायता सुविधाएँ उपलब्ध हैं। "
                "ट्रेनिंग और प्लेसमेंट टीम छात्रों को तकनीकी कौशल, संचार और करियर तैयारी में सहायता करती है।"
            )
        return (
            "Atria Institute of Technology is a reputed autonomous institution in Bangalore. "
            "It offers industry-aligned engineering and management programs designed for current market needs. "
            "The campus includes modern labs, research centers, and student support facilities. "
            "Its training and placement team helps students with technical skills, communication, and career readiness."
        )

    if "hello" in t:
        if is_hindi:
            return "नमस्ते। मैं आपको साफ़ सुन सकती हूँ।"
        return "Hello. I can hear you clearly."

    if "who are you" in t:
        if is_hindi:
            return "मैं लीना हूँ, एक रोबोटिक वॉइस असिस्टेंट।"
        return "I am Leena, a robotic voice assistant."

    if "tell me something" in t:
        if is_hindi:
            return "मैं एक वॉइस असिस्टेंट हूँ जो सुन सकती है और बोल सकती है।"
        return "I am a voice assistant that can listen and speak."

    if is_hindi:
        return "मैंने आपकी बात समझी, लेकिन अभी एआई मॉडल से उत्तर नहीं मिल सका।"

    return "I understood your request but need a connected AI model."


def language_name(lang_code):
    lc = (lang_code or "").lower()
    if lc.startswith("hi"):
        return "Hindi"
    if lc.startswith("ta"):
        return "Tamil"
    if lc.startswith("te"):
        return "Telugu"
    if lc.startswith("kn"):
        return "Kannada"
    if lc.startswith("ml"):
        return "Malayalam"
    return "English"


def get_response(user_text, lang_code):
    global gemini_retry_after_ts, gemini_disabled_for_session

    context = fetch_atria_data(user_text)
    lang = language_name(lang_code)
    history_block = "\n".join(
        [f"User: {h['user']}\nLeena: {h['assistant']}" for h in conversation_history]
    )

    prompt = (
        f"{SYSTEM_PROMPT}"
        f"Reply in {lang}. Keep the same language as the user's input.\n"
        f"Conversation so far:\n{history_block if history_block else 'No previous conversation.'}\n"
        f"Context:\n{context}\n"
        f"User: {user_text}"
    )

    now_ts = time.time()
    can_call_gemini = (not gemini_disabled_for_session) and (now_ts >= gemini_retry_after_ts)

    if can_call_gemini:
        try:
            res = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            text = getattr(res, "text", "")
            if text and text.strip():
                final_text = text.strip()
                conversation_history.append({"user": user_text, "assistant": final_text})
                return final_text
        except Exception as e:
            err = str(e)
            print("⚠️ Gemini error:", e)

            if "RESOURCE_EXHAUSTED" in err or "429" in err:
                retry_match = re.search(r"Please retry in\s*([0-9]+(?:\.[0-9]+)?)s", err)
                if retry_match:
                    retry_sec = float(retry_match.group(1))
                    gemini_retry_after_ts = time.time() + retry_sec
                    print(f"ℹ️ Gemini cooldown active for {int(retry_sec)}s, using fallback response.")
                if "PerDay" in err or "FreeTier" in err or "quota" in err.lower():
                    gemini_disabled_for_session = True
                    print("ℹ️ Gemini daily quota appears exhausted. Using local fallback for this run.")

    fallback_text = fallback_response(user_text, lang_code)
    conversation_history.append({"user": user_text, "assistant": fallback_text})
    return fallback_text

# =========================
# TTS (FIXED)
# =========================
def text_to_speech(text, lang_code="en-IN"):
    target_lang = lang_code if lang_code else "en-IN"
    speaker_candidates = [TTS_SPEAKER, "amelia", "vidya", "manisha", "anushka"]
    tried = set()

    for speaker in speaker_candidates:
        if speaker in tried:
            continue
        if speaker in unsupported_speakers:
            continue
        tried.add(speaker)

        try:
            res = sarvam.text_to_speech.convert(
                text=text,
                target_language_code=target_lang,
                speaker=speaker
            )

            if hasattr(res, "audios"):
                audio_bytes = base64.b64decode(res.audios[0])
                audio_file = io.BytesIO(audio_bytes)

                data, sr = sf.read(audio_file)

                if len(data.shape) > 1:
                    data = data[:, 0]

                if np.max(np.abs(data)) > 0:
                    data = data / np.max(np.abs(data))

                if speaker != TTS_SPEAKER:
                    print(f"ℹ️ TTS fallback speaker: {speaker}")

                return data, sr

            print("⚠️ Unknown TTS:", res)

        except Exception as e:
            print(f"⚠️ TTS speaker '{speaker}' failed:", e)
            if "not compatible" in str(e).lower():
                unsupported_speakers.add(speaker)

    print("❌ TTS: all speaker attempts failed")
    return None, None

# =========================
# PLAY (CRITICAL FIX)
# =========================
def play(audio, sr):
    try:
        TARGET_SR = 48000

        # resample if needed
        if sr != TARGET_SR:
            duration = len(audio) / sr
            new_len = int(duration * TARGET_SR)

            audio = np.interp(
                np.linspace(0, len(audio), new_len),
                np.arange(len(audio)),
                audio
            )
            sr = TARGET_SR

        audio = audio * 2.5  # volume boost

        sd.play(audio, sr, device=SPEAKER_INDEX)
        sd.wait()

    except Exception as e:
        print("❌ Speaker error:", e)

# =========================
# MAIN LOOP
# =========================
def assistant_loop():
    while True:
        try:
            audio = record_audio()

            text, lang = speech_to_text(audio)

            if not text.strip():
                print("⚠️ No speech detected")
                continue

            print("🧑", text)

            reply = get_response(text, lang)
            print("🤖", reply)

            audio_out, sr = text_to_speech(reply, lang)

            if audio_out is not None:
                play(audio_out, sr)

            time.sleep(0.3)

        except KeyboardInterrupt:
            print("\n🛑 Stopped")
            break

        except Exception as e:
            print("❌ Error:", e)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    assistant_loop()