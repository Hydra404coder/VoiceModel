import requests
from bs4 import BeautifulSoup

BASE_URL = "https://atria.edu/"

def fetch_atria_data(query):
    try:
        response = requests.get(BASE_URL, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")

        paragraphs = soup.find_all("p")
        text_data = " ".join([p.get_text() for p in paragraphs])

        # Simple filter (English + common Hindi transliterations)
        query_l = query.lower()
        if any(word in query_l for word in ["atria", "college", "institute", "एट्रिया", "अत्रिया", "इंस्टीट्यूट"]):
            return text_data[:2000]  # limit context size

        return ""

    except Exception as e:
        return ""