"""
preprocess.py  –  Roman Urdu text cleaning pipeline
"""

import re
import string

# ── Roman Urdu stopwords ────────────────────────────────────────────────────
STOPWORDS = {
    "ka", "ki", "ke", "ko", "se", "mein", "mai", "hai", "hain", "tha",
    "thi", "the", "hy", "ha", "ho", "hoga", "hogi", "honge", "hote",
    "aur", "ya", "bhi", "koi", "kuch", "kab", "kahan", "kyun", "kaise",
    "yeh", "ye", "woh", "wo", "is", "us", "iss", "uss", "in", "un",
    "ne", "par", "pe", "tak", "jab", "tab", "phir", "ab", "aaj",
    "kal", "main", "mujhe", "mujhy", "mera", "meri", "mere", "humara",
    "hamara", "hamari", "hamare", "ap", "aap", "apna", "apni", "apne",
    "wala", "wali", "wale", "jo", "jو", "tum", "tu", "tera", "teri",
    "kya", "kia", "nahi", "nahin", "nah", "na", "mat", "hum", "sab",
    "hi", "to", "toh", "lekin", "magar", "liye", "lye", "agar", "jis",
    "jab", "warna", "phir", "aisa", "aisi", "aise", "or", "bhai", "ji"
}


def clean_text(text: str) -> str:
    """Clean a single Roman Urdu text string."""
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove mentions and hashtags
    text = re.sub(r"@\w+|#\w+", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation and special characters (keep only letters and spaces)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    return " ".join(tokens)


def preprocess_series(series):
    """Apply clean_text to a pandas Series. Returns cleaned Series."""
    return series.apply(clean_text)
