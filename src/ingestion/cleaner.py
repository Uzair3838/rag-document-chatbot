import re


def clean_text(text):
    # Replace PDF bullet characters with a newline
    text = text.replace("", "\n")

    # Replace multiple spaces and tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Replace multiple blank lines with a single blank line
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Remove unnecessary whitespace from the beginning and end
    text = text.strip()

    return text