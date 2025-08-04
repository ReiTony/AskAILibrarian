import logging
import re
from langdetect import detect
from googletrans import Translator

logger = logging.getLogger("translation")
translator = Translator()

ALLOWED_LANGUAGES = {"en", "tl", "ceb", "ko"}

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        if lang not in ALLOWED_LANGUAGES:
            logger.warning(f"Unsupported language '{lang}' detected. Defaulting to 'en'.")
            return "en"
        return lang
    except Exception as e:
        logger.error(f"[LangDetect Error]: {e}")
        return "en"

async def translate_to_eng(text: str, src_lang: str) -> str:
    if src_lang == "en":
        return text
    try:
        translated = await translator.translate(text, src=src_lang, dest="en")
        logger.info(f"[→ EN] ({src_lang}): {text} → {translated.text}")
        return translated.text
    except Exception as e:
        logger.error(f"[Translation to English Error]: {e}")
        return text

async def translate_to_user_lang(text: str, target_lang: str) -> str:
    if target_lang == "en":
        return text

    try:
        temp_text, protected_map = extract_protected(text)

        translated = await translator.translate(temp_text, dest=target_lang)
        logger.info(f"[→ {target_lang.upper()}]: {text} → {translated.text}")

        # Restore protected segments like book titles
        restored = restore_protected(translated.text, protected_map)
        return restored
    except Exception as e:
        logger.error(f"[Translation to {target_lang} Error]: {e}")
        return text
    
# Aliases to maintain compatibility with previous function names
translate_to_english = translate_to_eng
translate_to_user_language = translate_to_user_lang

def extract_protected(text):
    protected_map = {}
    def replacer(match):
        token = f"<<PROTECTED_{len(protected_map)}>>"
        protected_map[token] = match.group(1)
        return token

    temp_text = re.sub(r"\[\[(.*?)\]\]", replacer, text)
    return temp_text, protected_map


def restore_protected(text, protected_map):
    text = re.sub(
        r"<<\s*PROTECTED\s*_?\s*(\d+)\s*>>",
        lambda m: f"<<PROTECTED_{m.group(1)}>>",
        text
    )
    text = re.sub(
        r"<<\s*Protected\s*_?\s*(\d+)\s*>>",
        lambda m: f"<<PROTECTED_{m.group(1)}>>",
        text
    )
    text = re.sub(
        r"<<\s*protected\s*_?\s*(\d+)\s*>>",
        lambda m: f"<<PROTECTED_{m.group(1)}>>",
        text
    )

    for token, original in protected_map.items():
        text = text.replace(token, original)

    return text
