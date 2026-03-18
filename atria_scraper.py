import os
import re
import time

DATA_FILE = "atria_structured_data.txt"

_CACHE_TTL_SECONDS = 3600
_FILE_CACHE = {"timestamp": 0, "data": {}}

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i", "in",
    "is", "it", "me", "my", "of", "on", "or", "our", "please", "tell", "that", "the",
    "their", "there", "to", "us", "was", "what", "when", "where", "which", "who", "with",
    "you", "your", "about", "this", "these", "those"
}

# Keyword expansion: maps words (transliterated Hindi or English) to relevant synonyms
_KEYWORD_EXPANSION = {
    # Hindi transliterated terms
    "kors": ["engineering", "program", "course", "degree", "undergraduate"],
    "peshksh": ["offer", "provide", "program", "engineering"],
    "prkram": ["program", "engineering", "course"],
    "vibhag": ["engineering", "department", "computer", "science"],
    "vidhyarthee": ["student", "course", "program"],
    "suvidhaaen": ["facility", "facilities", "infrastructure"],
    "sukhavilta": ["facility", "facilities"],
    "suvidha": ["facility", "facilities", "hostel"],
    "pritiplsmnt": ["placement", "training", "job"],
    "plesment": ["placement", "training", "job", "career"],
    "niyukt": ["placement", "job"],
    
    # English terms
    "department": ["departments", "program", "programs", "engineering", "cse", "ise", "ece", "mechanical", "civil"],
    "departments": ["department", "program", "programs", "engineering", "cse", "ise", "ece", "mechanical", "civil"],
    "branch": ["department", "departments", "engineering", "cse", "ise", "ece"],
    "branches": ["department", "departments", "engineering", "cse", "ise", "ece"],
    "admission": ["admissions", "kcet", "comedk", "application", "counseling", "verification"],
    "admissions": ["admission", "kcet", "comedk", "application", "counseling", "verification"],
    "courses": ["engineering", "program", "academic", "degree"],
    "program": ["engineering", "academic", "course"],
    "tell": ["information", "about", "describe"],
}

_HINDI_TRANSLITERATION_MAP = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'uu',
    'ऋ': 'ri', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v',
    'श': 'sh', 'ष': 's', 'स': 's', 'ह': 'h',
    'ा': 'aa', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'uu', 'ृ': 'ri',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ं': 'n', 'ः': 'h', '्': '',
}


def _transliterate_hindi_to_latin(text):
    """Convert Hindi text to Latin characters."""
    if not text:
        return text
    transliterated = ""
    for char in text:
        transliterated += _HINDI_TRANSLITERATION_MAP.get(char, char)
    return transliterated


def _tokenize(text):
    """Extract searchable tokens from text."""
    if not text:
        return set()
    text = _transliterate_hindi_to_latin(text)
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _load_data_from_file():
    """Load and parse the structured data file."""
    now = time.time()
    
    if _FILE_CACHE["data"] and (now - _FILE_CACHE["timestamp"] < _CACHE_TTL_SECONDS):
        return _FILE_CACHE["data"]
    
    if not os.path.exists(DATA_FILE):
        return {}
    
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    
    sections = re.split(r"={10,}", content)
    
    data = {
        "title": "Atria Institute of Technology",
        "about": "",
        "facilities": [],
        "courses": [],
        "sections": []
    }
    
    pending_heading = ""

    for section in sections:
        section = section.strip()
        if not section:
            continue
        
        lines = [line.strip() for line in section.split("\n") if line.strip()]
        if not lines:
            continue
        
        if len(lines) == 1:
            heading_candidate = lines[0].strip()
            if re.match(r"^[A-Z0-9& /:().,-]{3,}$", heading_candidate):
                pending_heading = heading_candidate.lower()
                continue

        if pending_heading:
            section_title = pending_heading
            section_text = " ".join(lines)
            pending_heading = ""
        else:
            section_title = lines[0].lower()
            section_text = " ".join(lines[1:])
        
        if "about" in section_title:
            data["about"] = section_text[:500]
        
        if "facilities" in section_title or "infrastructure" in section_title:
            facility_items = [l for l in lines[1:] if l.startswith("-")]
            data["facilities"].extend(facility_items)
        
        if "academic" in section_title or "program" in section_title:
            course_items = [l for l in lines[1:] if l.startswith("-")]
            data["courses"].extend(course_items)
        
        if section_text:
            data["sections"].append({"title": section_title, "content": section_text})
    
    data["facilities"] = list(dict.fromkeys(data["facilities"]))[:20]
    data["courses"] = list(dict.fromkeys(data["courses"]))[:20]
    
    _FILE_CACHE["timestamp"] = now
    _FILE_CACHE["data"] = data
    
    return data


def _retrieve_sections(query, sections, top_k=4):
    """Fast retrieval of relevant sections with keyword expansion."""
    if not query or not sections:
        return []
    
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []
    
    # Expand query tokens with synonyms
    expanded_tokens = set(query_tokens)
    for token in query_tokens:
        if token in _KEYWORD_EXPANSION:
            expanded_tokens.update(_KEYWORD_EXPANSION[token])
    
    ranked = []
    for section in sections:
        title = section.get("title", "")
        content = section.get("content", "")
        content_tokens = _tokenize(content)
        title_tokens = _tokenize(title)

        overlap_content = len(expanded_tokens.intersection(content_tokens))
        overlap_title = len(expanded_tokens.intersection(title_tokens))
        overlap = overlap_content + (overlap_title * 2)
        if overlap >= 1:
            coverage = overlap / max(len(expanded_tokens), 1)
            score = overlap + (coverage * 0.5)
            ranked.append((score, content))
    
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [content for _, content in ranked[:top_k]]


def fetch_atria_data(query):
    """Fetch relevant Atria information from file."""
    try:
        base_data = _load_data_from_file()
        
        result = {
            "title": base_data.get("title", ""),
            "about": base_data.get("about", ""),
            "facilities": base_data.get("facilities", []),
            "courses": base_data.get("courses", []),
            "relevant_snippets": []
        }
        
        if query and query.strip():
            sections = base_data.get("sections", [])
            result["relevant_snippets"] = _retrieve_sections(query, sections, top_k=4)
        
        return result
    
    except Exception as e:
        print(f"Error: {e}")
        return {}
