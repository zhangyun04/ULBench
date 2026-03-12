def normalize_choice_text(text: str) -> str:
    return " ".join(text.strip().lower().split())
