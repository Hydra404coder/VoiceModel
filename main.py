import os
import time
import wave
import base64
import tempfile
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from collections import deque
import numpy as np
import sounddevice as sd
import soundfile as sf
import io
from dotenv import load_dotenv
from sarvamai import SarvamAI
from google import genai

from atria_scraper import fetch_atria_data

# =========================
# DEVICE CONFIG
# =========================
MIC_INDEX = 1
SPEAKER_INDEX = 8

sd.default.device = (MIC_INDEX, SPEAKER_INDEX)
sd.default.samplerate = 48000

DEVICE_SR = int(sd.query_devices(MIC_INDEX)['default_samplerate'])

print(f"Mic: {MIC_INDEX}")
print(f"Speaker: {SPEAKER_INDEX}")
print(f"Mic sample rate: {DEVICE_SR}")

# =========================
# LOAD API
# =========================
load_dotenv()
sarvam = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))

# Better voice
TTS_SPEAKER = "anushka"

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = "gemini-3-flash-preview"
CHAT_HISTORY = deque(maxlen=6)
LAST_USER_TEXT = ""
LAST_REPLY_TEXT = ""
SARVAM_SUPPORTED_LANG_CODES = {
    "as-IN", "bn-IN", "brx-IN", "doi-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "kok-IN",
    "ks-IN", "mai-IN", "ml-IN", "mni-IN", "mr-IN", "ne-IN", "od-IN", "pa-IN", "sa-IN",
    "sat-IN", "sd-IN", "ta-IN", "te-IN", "ur-IN"
}

# =========================
# CACHE (FAST RESPONSE)
# =========================
SCRAPE_CACHE = None

def get_cached_data():
    global SCRAPE_CACHE
    if SCRAPE_CACHE is None:
        SCRAPE_CACHE = fetch_atria_data("")
    return SCRAPE_CACHE

# =========================
# NATURAL PROMPT
# =========================
SYSTEM_PROMPT = (
    "You are Leena, the official assistant for Atria Institute of Technology.\n"
    "Respond in a polite and formal tone suitable for students and parents.\n"
    "Use concise language and complete sentences.\n"
    "If specific information is unavailable, say so clearly and avoid guessing.\n"
)

# =========================
# RESPONSE CLEANUP
# =========================
def normalize_response_text(text):
    if not text:
        return ""

    text = text.replace("\n", " ")
    text = " ".join(text.split())

    sentence_chunks = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    cleaned = " ".join(sentence_chunks[:4])
    return cleaned[:420]


def language_name_from_code(lang_code):
    code = (lang_code or "en-IN").lower()
    if code.startswith("hi"):
        return "Hindi"
    if code.startswith("pa"):
        return "Punjabi"
    if code.startswith("kn"):
        return "Kannada"
    if code.startswith("ta"):
        return "Tamil"
    if code.startswith("te"):
        return "Telugu"
    if code.startswith("ml"):
        return "Malayalam"
    return "English"


def detect_preferred_lang_code(user_text, stt_lang_code):
    text = user_text or ""

    if re.search(r"[\u0900-\u097F]", text):
        return "hi-IN"
    if re.search(r"[\u0A00-\u0A7F]", text):
        return "pa-IN"
    if re.search(r"[\u0C80-\u0CFF]", text):
        return "kn-IN"
    if re.search(r"[\u0B80-\u0BFF]", text):
        return "ta-IN"
    if re.search(r"[\u0C00-\u0C7F]", text):
        return "te-IN"
    if re.search(r"[\u0D00-\u0D7F]", text):
        return "ml-IN"

    normalized = (stt_lang_code or "").strip()
    if normalized and normalized.lower() != "unknown":
        return normalized

    return "en-IN"


def normalize_lang_code(lang_code):
    code = (lang_code or "").strip()
    if not code:
        return "en-IN"

    normalized = code.lower()
    short_map = {
        "en": "en-IN",
        "hi": "hi-IN",
        "pa": "pa-IN",
        "kn": "kn-IN",
        "ta": "ta-IN",
        "te": "te-IN",
        "ml": "ml-IN",
        "mr": "mr-IN",
        "gu": "gu-IN",
        "bn": "bn-IN",
        "ur": "ur-IN",
        "od": "od-IN",
        "as": "as-IN",
    }

    if len(normalized) == 2 and normalized in short_map:
        return short_map[normalized]

    if "-" in code:
        left, right = code.split("-", 1)
        candidate = f"{left.lower()}-{right.upper()}"
        if candidate in SARVAM_SUPPORTED_LANG_CODES:
            return candidate

    candidate = short_map.get(normalized)
    if candidate:
        return candidate

    return "en-IN"


def detect_user_language(user_text, stt_lang_code="en-IN"):
    heuristic_lang = normalize_lang_code(detect_preferred_lang_code(user_text, stt_lang_code))
    text = (user_text or "").strip()

    if not text:
        return heuristic_lang

    try:
        identified = sarvam.text.identify_language(input=text)
        detected = normalize_lang_code(getattr(identified, "language_code", ""))
        if detected in SARVAM_SUPPORTED_LANG_CODES:
            return detected
    except Exception:
        pass

    return heuristic_lang


def translate_with_sarvam(text, source_lang, target_lang):
    content = (text or "").strip()
    if not content:
        return ""

    source = normalize_lang_code(source_lang)
    target = normalize_lang_code(target_lang)

    if source == target:
        return content

    if target not in SARVAM_SUPPORTED_LANG_CODES:
        return content

    try:
        res = sarvam.text.translate(
            input=content,
            source_language_code=source,
            target_language_code=target,
            mode="formal",
            model="sarvam-translate:v1",
            numerals_format="international",
        )
        translated = getattr(res, "translated_text", "")
        return translated.strip() if translated else content
    except Exception:
        try:
            res = sarvam.text.translate(
                input=content,
                source_language_code="auto",
                target_language_code=target,
            )
            translated = getattr(res, "translated_text", "")
            return translated.strip() if translated else content
        except Exception:
            return content


def out_of_scope_message(lang_code):
    code = (lang_code or "en-IN").lower()
    if code.startswith("hi"):
        return "मुझे इस प्रश्न का उत्तर उपलब्ध डेटा में नहीं मिला। कृपया एट्रिया इंस्टीट्यूट के बारे में कोई प्रश्न पूछें - जैसे कोर्स, विभाग, होस्टल, या प्लेसमेंट।"
    if code.startswith("pa"):
        return "ਮੈਨੂੰ ਇਸ ਸਵਾਲ ਦਾ ਜਵਾਬ ਉਪਲਬਧ ਜਾਣਕਾਰੀ ਵਿੱਚ ਨਹੀਂ ਮਿਲਿਆ। ਕਿਰਪਾ ਕਰਕੇ Atria ਇੰਸਟੀਚਿਊਟ ਬਾਰੇ ਪੁੱਛ - ਜਿਵੇਂ ਕੋਰਸ, ਵਿਭਾਗ, ਹੋਸਟਲ, ਜਾਂ ਪਲੇਸਮੈਂਟ।"
    if code.startswith("kn"):
        return "ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರ ಲಭ್ಯವಿರುವ ಮಾಹಿತಿಯಲ್ಲಿ ಸಿಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು Atria ಸಂಸ್ಥೆ ಬಗ್ಗೆ ಪ್ರಶ್ನೆ ಕೇಳಿ - ಕೋರ್ಸ್‌ಗಳು, ವಿಭಾಗಗಳು, ವಸತಿ, ಅಥವಾ ನೇಮಕಾತಿ."
    return "I could not find the answer to this in available data. Please ask about Atria Institute - such as courses, departments, hostel, or placements."


def generic_about_message(lang_code):
    code = (lang_code or "en-IN").lower()
    if code.startswith("hi"):
        return "एट्रिया इंस्टीट्यूट ऑफ टेक्नोलॉजी, बेंगलुरु का एक प्रतिष्ठित स्वायत्त संस्थान है। कृपया बताइए, क्या आप कोर्स, प्रवेश, विभाग, हॉस्टल या प्लेसमेंट के बारे में जानकारी चाहते हैं?"
    if code.startswith("pa"):
        return "Atria Institute of Technology, Bengaluru ਇੱਕ ਮਾਨਤਾ ਪ੍ਰਾਪਤ ਸਵੈ-ਸ਼ਾਸਿਤ ਸੰਸਥਾ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਦੱਸੋ, ਤੁਸੀਂ ਕੋਰਸ, ਦਾਖਲਾ, ਵਿਭਾਗ, ਹੋਸਟਲ ਜਾਂ ਪਲੇਸਮੈਂਟ ਬਾਰੇ ਜਾਣਕਾਰੀ ਚਾਹੁੰਦੇ ਹੋ?"
    if code.startswith("kn"):
        return "Atria Institute of Technology, Bengaluru ಒಂದು ಪ್ರತಿಷ್ಠಿತ ಸ್ವಾಯತ್ತ ಸಂಸ್ಥೆಯಾಗಿದೆ. ದಯವಿಟ್ಟು ತಿಳಿಸಿ, ನಿಮಗೆ ಕೋರ್ಸ್‌ಗಳು, ಪ್ರವೇಶ, ವಿಭಾಗಗಳು, ವಸತಿ ಅಥವಾ ಪ್ಲೇಸ್ಮೆಂಟ್ ಕುರಿತು ಮಾಹಿತಿ ಬೇಕೇ?"
    return "Atria Institute of Technology, Bengaluru is a reputed autonomous institution. Please let me know whether you need details on courses, admissions, departments, hostel, or placements."


def followup_hint(lang_code):
    code = (lang_code or "en-IN").lower()
    if code.startswith("hi"):
        return "कृपया बताइए, क्या आप प्रवेश, विभाग, सुविधाएँ या प्लेसमेंट की जानकारी चाहते हैं?"
    if code.startswith("pa"):
        return "ਕਿਰਪਾ ਕਰਕੇ ਦੱਸੋ, ਕੀ ਤੁਸੀਂ ਦਾਖਲਾ, ਵਿਭਾਗ, ਸੁਵਿਧਾਵਾਂ ਜਾਂ ਪਲੇਸਮੈਂਟ ਬਾਰੇ ਜਾਣਨਾ ਚਾਹੁੰਦੇ ਹੋ?"
    if code.startswith("kn"):
        return "ದಯವಿಟ್ಟು ತಿಳಿಸಿ, ನಿಮಗೆ ಪ್ರವೇಶ, ವಿಭಾಗಗಳು, ಸೌಕರ್ಯಗಳು ಅಥವಾ ಪ್ಲೇಸ್ಮೆಂಟ್ ಬಗ್ಗೆ ಮಾಹಿತಿ ಬೇಕೇ?"
    return "Please let me know if you need details about admissions, departments, facilities, or placements."


def is_about_atria_query(user_text):
    t = (user_text or "").lower()
    if "atria" in t and any(x in t for x in ["about", "tell", "institute", "college", "who"]):
        return True
    if re.search(r"एट्रिया|इंस्टीट्यूट|कॉलेज", user_text or "") and re.search(r"बताइए|बारे", user_text or ""):
        return True
    return False

# =========================
# RECORD AUDIO (SIMPLE & RELIABLE)
# =========================
def record_audio(duration=5):
    """Record audio for fixed duration. Simple and reliable."""
    print("🎤 Listening...")
    
    # Record for fixed duration
    num_samples = int(duration * DEVICE_SR)
    audio = sd.rec(num_samples, samplerate=DEVICE_SR, channels=1, dtype="float32")
    sd.wait()
    
    # Amplify and normalize
    audio *= 10
    audio = np.clip(audio, -1, 1)
    
    return audio

# =========================
# RESAMPLE
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

        return getattr(res, "transcript", ""), getattr(res, "language_code", "en-IN")

    except Exception as e:
        print("❌ STT:", e)
        return "", "en-IN"

# =========================
# INTENT DETECTION
# =========================
def detect_intent(text):
    t = text.lower()

    if any(x in t for x in ["canteen", "food", "eat"]):
        return "food"
    if any(x in t for x in ["course", "branch", "study"]):
        return "course"
    if any(x in t for x in ["hostel", "stay"]):
        return "hostel"

    return "general"

# =========================
# SMART FALLBACK
# =========================
def fallback_response(text, lang_code="en-IN"):
    data = get_cached_data()

    if not data:
        return out_of_scope_message(lang_code)

    intent = detect_intent(text)

    if intent == "food":
        if data.get("facilities"):
            return f"Campus facilities include {', '.join(data['facilities'][:3])}."
        return "Food options are available on campus."

    if intent == "course":
        if data.get("courses"):
            return f"Atria offers programs such as {', '.join(data['courses'][:3])}."
        return "Atria offers engineering and management programs."

    if intent == "hostel":
        return "Hostel facilities are available for students."

    if is_about_atria_query(text):
        return generic_about_message(lang_code)

    return out_of_scope_message(lang_code)


def build_rag_context(query):
    data = fetch_atria_data(query)
    if not data:
        return "", ""

    snippets = data.get("relevant_snippets") or []

    context_lines = []
    for idx, snippet in enumerate(snippets[:3], start=1):  # Top 3 most relevant
        snippet = snippet.strip()
        if snippet:
            context_lines.append(f"- {snippet}")

    if not context_lines:
        return "", ""  # Return empty context if no snippets found

    return "\n".join(context_lines), fallback_response(query)


def _llm_response_with_timeout(contents, timeout_seconds=5.0):
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                gemini_client.models.generate_content,
                model=GEMINI_MODEL,
                contents=contents,
            )
            try:
                res = future.result(timeout=timeout_seconds)
                text = getattr(res, "text", "").strip()
                if text:
                    return text
                else:
                    return None
            except TimeoutError:
                return None
    except Exception as e:
        # Silent fail - LLM unavailable (quota exhausted, network issue, etc.)
        return None


def _doc_grounded_fallback(user_text, rag_context, lang_code="en-IN"):
    if not rag_context:
        return ""

    text = rag_context.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    is_hindi = (lang_code or "").lower().startswith("hi")

    query = (user_text or "").lower()
    is_admission_query = any(k in query for k in ["admission", "admissions", "kcet", "comedk", "apply", "application"]) \
        or bool(re.search(r"प्रवेश|एडमिशन", user_text or ""))
    is_placement_query = any(k in query for k in ["placement", "training", "recruit", "job"]) \
        or bool(re.search(r"प्लेसमेंट|ट्रेनिंग|नियुक्ति|नौकरी", user_text or ""))
    is_department_query = any(k in query for k in ["department", "departments", "branch", "branches", "program", "programs"]) \
        or bool(re.search(r"विभाग|डिपार्टमेंट|शाखा|प्रोग्राम", user_text or ""))

    def _split_bullets(block):
        if not block:
            return []
        return [item.strip(" -") for item in re.split(r"\s+-\s+", block) if item.strip()]

    def _to_hindi_item(item):
        replacements = {
            "Technical training": "तकनीकी प्रशिक्षण",
            "coding": "कोडिंग",
            "core subjects": "मुख्य विषय",
            "Soft skills": "सॉफ्ट स्किल्स",
            "communication": "संचार",
            "aptitude": "एप्टीट्यूड",
            "Resume building": "रिज़्यूमे निर्माण",
            "Mock interviews": "मॉक इंटरव्यू",
            "Industry tie-ups": "उद्योग सहयोग",
            "Internship opportunities": "इंटर्नशिप अवसर",
            "Campus recruitment drives": "कैंपस भर्ती अभियान",
            "To ensure students are": "छात्रों को",
            "job-ready and industry capable": "नौकरी के लिए तैयार और उद्योग के अनुरूप",
            "Application submission": "आवेदन जमा करना",
            "Counseling / seat allocation": "काउंसलिंग / सीट आवंटन",
            "Document verification": "दस्तावेज़ सत्यापन",
            "Fee payment": "शुल्क भुगतान",
            "Management quota": "मैनेजमेंट कोटा",
        }
        translated = item
        for en, hi in replacements.items():
            translated = re.sub(re.escape(en), hi, translated, flags=re.IGNORECASE)
        return translated

    if is_admission_query:
        routes_match = re.search(r"Admission Routes:\s*(.*?)\s*Process:", text, flags=re.IGNORECASE)
        process_match = re.search(r"Process:\s*(.*?)(?:\s*Phone:|\s*Email:|\s*Website:|$)", text, flags=re.IGNORECASE)

        routes_text = routes_match.group(1).strip() if routes_match else ""
        process_text = process_match.group(1).strip() if process_match else ""

        routes_items = _split_bullets(routes_text)
        process_items = _split_bullets(process_text)

        if is_hindi:
            hindi_routes = [_to_hindi_item(item) for item in routes_items[:3]]
            hindi_process = [_to_hindi_item(item) for item in process_items[:4]]
            routes_part = " , ".join(hindi_routes) if hindi_routes else "KCET, COMEDK और मैनेजमेंट कोटा"
            process_part = " , ".join(hindi_process) if hindi_process else "आवेदन जमा करना, काउंसलिंग/सीट आवंटन, दस्तावेज़ सत्यापन और शुल्क भुगतान"
            return (
                f"Atria में प्रवेश के मुख्य मार्ग हैं: {routes_part}. "
                f"प्रवेश प्रक्रिया में ये चरण शामिल हैं: {process_part}."
            )

        routes_part = ", ".join(routes_items[:3]) if routes_items else "KCET, COMEDK, and Management quota"
        process_part = ", ".join(process_items[:4]) if process_items else "application submission, counseling/seat allocation, document verification, and fee payment"
        return (
            f"Atria admissions are available through {routes_part}. "
            f"The process includes {process_part}."
        )

    if is_placement_query:
        training_match = re.search(r"focuses on:\s*(.*?)\s*Placement Features:", text, flags=re.IGNORECASE)
        features_match = re.search(r"Placement Features:\s*(.*?)\s*Goal:", text, flags=re.IGNORECASE)
        goal_match = re.search(r"Goal:\s*(.*?)(?:\s+-\s+Atria Institute|$)", text, flags=re.IGNORECASE)

        training_items = _split_bullets(training_match.group(1).strip() if training_match else "")
        feature_items = _split_bullets(features_match.group(1).strip() if features_match else "")
        goal_text = (goal_match.group(1).strip() if goal_match else "")
        goal_text = re.sub(r"\s+-\s+Atria Institute.*$", "", goal_text, flags=re.IGNORECASE).strip()
        if " - " in goal_text:
            goal_text = goal_text.split(" - ", 1)[0].strip()

        if is_hindi:
            training_part = " , ".join([_to_hindi_item(item) for item in training_items[:4]]) if training_items else "तकनीकी प्रशिक्षण, सॉफ्ट स्किल्स, रिज़्यूमे निर्माण और मॉक इंटरव्यू"
            features_part = " , ".join([_to_hindi_item(item) for item in feature_items[:3]]) if feature_items else "उद्योग सहयोग, इंटर्नशिप अवसर और कैंपस भर्ती अभियान"
            goal_part = _to_hindi_item(goal_text) if goal_text else "छात्रों को नौकरी के लिए तैयार बनाना"
            goal_part = re.sub(r"\.+$", "", goal_part).strip()
            return (
                f"Atria के Training and Placement Cell में {training_part} पर फोकस किया जाता है. "
                f"मुख्य प्लेसमेंट सुविधाओं में {features_part} शामिल हैं. लक्ष्य: {goal_part}."
            )

        training_part = ", ".join(training_items[:4]) if training_items else "technical training, soft skills, resume building, and mock interviews"
        features_part = ", ".join(feature_items[:3]) if feature_items else "industry tie-ups, internship opportunities, and campus recruitment drives"
        goal_part = goal_text if goal_text else "to make students job-ready and industry capable"
        return (
            f"Atria's Training and Placement Cell focuses on {training_part}. "
            f"Key placement features include {features_part}. Goal: {goal_part}."
        )

    if is_department_query:
        ug_match = re.search(r"Undergraduate Engineering:\s*(.*?)\s*Postgraduate:", text, flags=re.IGNORECASE)
        pg_match = re.search(r"Postgraduate:\s*(.*?)\s*Key Focus:", text, flags=re.IGNORECASE)

        ug_items = _split_bullets(ug_match.group(1).strip() if ug_match else "")
        pg_items = _split_bullets(pg_match.group(1).strip() if pg_match else "")

        if is_hindi:
            ug_part = " , ".join([_to_hindi_item(item) for item in ug_items[:5]]) if ug_items else "CSE, ISE, ECE, Mechanical और Civil Engineering"
            pg_part = " , ".join([_to_hindi_item(item) for item in pg_items[:2]]) if pg_items else "MBA"
            return (
                f"Atria के मुख्य विभागों में {ug_part} शामिल हैं. "
                f"पोस्टग्रेजुएट स्तर पर {pg_part} उपलब्ध है."
            )

        ug_part = ", ".join(ug_items[:5]) if ug_items else "CSE, ISE, ECE, Mechanical Engineering, and Civil Engineering"
        pg_part = ", ".join(pg_items[:2]) if pg_items else "MBA"
        return (
            f"Atria's key departments include {ug_part}. "
            f"At postgraduate level, {pg_part} is available."
        )

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if sentences:
        if is_hindi:
            return "उपलब्ध जानकारी के अनुसार, कृपया प्रश्न को कोर्स, प्रवेश, सुविधाएँ या प्लेसमेंट जैसे विषय में पूछें ताकि मैं सटीक उत्तर दे सकूँ।"
        return " ".join(sentences[:2])[:380]

    return text[:380]

# =========================
# RESPONSE ENGINE (FAST)
# =========================
def get_response(user_text, lang_code="en-IN"):
    global LAST_REPLY_TEXT
    global LAST_USER_TEXT

    preferred_lang_code = detect_user_language(user_text, lang_code)

    english_query = translate_with_sarvam(user_text, preferred_lang_code, "en-IN")
    if not english_query:
        english_query = user_text

    rag_context, _ = build_rag_context(english_query)

    history_text = "\n".join(
        [f"User: {u}\nAssistant: {a}" for u, a in list(CHAT_HISTORY)[-2:]]
    )

    if rag_context:
        prompt = (
            f"{SYSTEM_PROMPT}\n"
            "Use the retrieved Atria context to answer the user's question. "
            "Do not invent facts. Do not repeat the full context. "
            "Return a short, direct, formal answer in English.\n\n"
            f"Conversation history:\n{history_text if history_text else 'No prior turns.'}\n\n"
            f"User question (English): {english_query}\n"
            "Output language: English (en-IN)\n\n"
            f"Retrieved Atria context:\n{rag_context}\n\n"
            "Answer:"
        )

        english_answer = ""
        try:
            text = _llm_response_with_timeout(prompt, timeout_seconds=5.0)
            if text:
                english_answer = normalize_response_text(text)
        except Exception:
            pass

        if not english_answer:
            grounded = _doc_grounded_fallback(english_query, rag_context, "en-IN")
            if grounded:
                english_answer = normalize_response_text(grounded)

        if english_answer:
            final_answer = english_answer
            if preferred_lang_code != "en-IN":
                translated = translate_with_sarvam(english_answer, "en-IN", preferred_lang_code)
                final_answer = normalize_response_text(translated)

            LAST_REPLY_TEXT = final_answer
            LAST_USER_TEXT = user_text
            CHAT_HISTORY.append((user_text, final_answer))
            return final_answer

    # Fallback if no context available or the model could not answer
    normalized_fallback = normalize_response_text(fallback_response(user_text, preferred_lang_code))
    LAST_REPLY_TEXT = normalized_fallback
    LAST_USER_TEXT = user_text
    CHAT_HISTORY.append((user_text, normalized_fallback))
    return normalized_fallback

# =========================
# TTS
# =========================
def text_to_speech(text, lang_code="en-IN"):
    try:
        res = sarvam.text_to_speech.convert(
            text=text,
            target_language_code=lang_code,
            speaker=TTS_SPEAKER
        )

        audio_bytes = base64.b64decode(res.audios[0])
        audio_file = io.BytesIO(audio_bytes)

        data, sr = sf.read(audio_file)

        if len(data.shape) > 1:
            data = data[:, 0]

        data = data / np.max(np.abs(data))

        return data, sr

    except Exception as e:
        print("❌ TTS:", e)
        return None, None

# =========================
# PLAY AUDIO
# =========================
def play(audio, sr):
    try:
        TARGET_SR = 48000

        if sr != TARGET_SR:
            duration = len(audio) / sr
            new_len = int(duration * TARGET_SR)

            audio = np.interp(
                np.linspace(0, len(audio), new_len),
                np.arange(len(audio)),
                audio
            )
            sr = TARGET_SR

        audio *= 2.5

        sd.play(audio, sr, device=SPEAKER_INDEX)
        sd.wait()

    except Exception as e:
        print("❌ Speaker error:", e)

# =========================
# STARTUP SPEECH
# =========================
def speak_startup():
    text = "Leena assistant is now online."
    audio, sr = text_to_speech(text)

    if audio is not None:
        play(audio, sr)

# =========================
# MAIN LOOP
# =========================
def assistant_loop():
    while True:
        try:
            audio = record_audio(duration=5)

            if len(audio) == 0:
                continue

            text, lang = speech_to_text(audio)

            if not text.strip():
                time.sleep(0.5)
                continue

            print("🧑", text)

            preferred_lang = detect_preferred_lang_code(text, lang)
            reply = get_response(text, preferred_lang)
            print("🤖", reply)

            audio_out, sr = text_to_speech(reply, preferred_lang)

            if audio_out is not None:
                play(audio_out, sr)
                time.sleep(0.5)
            else:
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
    speak_startup()
    assistant_loop()