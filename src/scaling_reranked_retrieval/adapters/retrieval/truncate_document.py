def truncate_document(doc: str, max_words: int = 500) -> str:
    """Truncate a document to a maximum number of words."""
    words = doc.split(" ")
    if len(words) <= max_words:
        return doc
    return " ".join(words[:max_words])