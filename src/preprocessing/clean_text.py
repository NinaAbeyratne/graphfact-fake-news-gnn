"""
Text-cleaning utilities 

Pipeline applied to the 'statement' and 'context' columns:
  1. Lowercase
  2. Remove URLs
  3. Remove HTML tags
  4. Normalize whitespace

After cleaning, 'statement' and 'context' are concatenated into a single
'article' column that is truncated to MAX_WORDS words.  The original
'statement' and 'context' columns are then dropped.
"""

import re

# Individual cleaning helpers

# Remove URLs from text
def remove_urls(text: str) -> str:
    return re.sub(r'https?://\S+|www\.\S+', '', text)

# Strip HTML / XML tags from text
def remove_html(text: str) -> str:
    return re.sub(r'<[^>]+>', '', text)

# Collapse multiple whitespace characters into a single space
def normalize_spaces(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


# Full text-cleaning pipeline

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = remove_urls(text)
    text = remove_html(text)
    text = normalize_spaces(text)
    return text


# Word-level truncation
MAX_WORDS = 128

# Return at most max_words whitespace-separated tokens from text
def truncate_words(text: str, max_words: int = MAX_WORDS) -> str:
    if not isinstance(text, str):
        return text
    return " ".join(text.split()[:max_words])


# DataFrame-level helpers

# Apply clean_text() to the 'statement' and 'context' columns of df in-place and return the modified DataFrame.
def apply_text_cleaning(df):
    df["statement"] = df["statement"].apply(clean_text)
    df["context"] = df["context"].apply(clean_text)
    return df


#  Concatenate 'context' and 'statement' into an 'article' column, truncate to MAX_WORDS words, then drop the source columns.
def build_article_column(df):
    df["article"] = (
        df["context"].fillna("") + " " + df["statement"].fillna("")
    ).str.strip()

    df["article"] = df["article"].apply(truncate_words)
    df = df.drop(columns=["statement", "context"])
    return df
