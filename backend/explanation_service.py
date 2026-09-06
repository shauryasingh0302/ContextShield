"""
Deterministic, rule-based explanation and safer-caption suggestion service for ContextShield.

Generates honest, student-level explanations and suggestions based strictly on:
- Model predicted risk category (SAFE, OFFENSIVE, HATE, HARASSMENT)
- Model prediction confidence
- Accompanying caption and OCR text

Rules:
1. SAFE:
   - explanation: explain that no significant risk category was detected.
   - suggestion: None (null)
2. OFFENSIVE:
   - explanation: explain that the content may contain offensive or insulting language.
   - suggestion: neutralized version when possible; otherwise generic safe suggestion.
3. HATE:
   - explanation: explain that the content may contain hateful or identity-targeted content.
   - suggestion: suggest removing hateful/identity-targeted wording and replacing it with neutral language.
4. HARASSMENT:
   - explanation: explain that the content may target or insult an individual.
   - suggestion: suggest removing personal attacks and rewriting it in neutral language.
"""

import re
from typing import Dict, List, Optional


# Mappings of common offensive/insulting terms to milder, neutralized alternatives.
MILD_SUBSTITUTIONS: Dict[str, str] = {
    "hate": "strongly disagree with",
    "stupid": "unwise",
    "idiot": "person",
    "moron": "critic",
    "dumb": "unclear",
    "ugly": "unpleasant",
    "shut up": "please listen",
    "kill": "defeat",
    "die": "fail",
    "trash": "poor quality",
    "garbage": "unhelpful",
    "crap": "stuff",
    "damn": "darn",
    "hell": "trouble",
    "asshole": "disrespectful person",
    "bitch": "critic",
    "bastard": "opponent",
    "fuck": "mess",
    "fucking": "really",
    "shit": "trouble",
}


def find_flagged_words(text: str) -> List[str]:
    """
    Finds exact word boundary occurrences of known offensive/insulting words in the text.
    Returns only terms that are actually present.
    """
    if not text:
        return []
    found = []
    lower_text = text.lower()
    for word in MILD_SUBSTITUTIONS:
        pattern = r"\b" + re.escape(word) + r"\b"
        if re.search(pattern, lower_text):
            found.append(word)
    return found


def neutralize_text(text: str) -> Optional[str]:
    """
    Attempts to neutralize flagged words in the caption by substituting milder terms.
    Preserves the user's original structure. Returns None if no flagged terms are found.
    """
    if not text or not text.strip():
        return None

    modified = text
    changed = False
    for word, replacement in MILD_SUBSTITUTIONS.items():
        pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
        if pattern.search(modified):
            modified = pattern.sub(replacement, modified)
            changed = True

    return modified.strip() if changed else None


def generate_explanation(
    risk_label: str,
    confidence: float,
    caption: str = "",
    ocr_text: str = "",
) -> str:
    """
    Generates a short, honest, rule-based explanation based on the predicted risk label
    and confidence. Does not claim specific words were detected unless they actually appear.
    """
    risk = (risk_label or "").upper().strip()
    conf_pct = f"{confidence * 100:.1f}%"
    combined_text = f"{caption or ''} {ocr_text or ''}".strip()
    flagged = find_flagged_words(combined_text)

    if risk == "SAFE":
        return "No significant risk category was detected in the post. The content appears safe for general audiences."

    if risk == "OFFENSIVE":
        if flagged:
            flagged_str = ", ".join(f"'{w}'" for w in sorted(set(flagged)))
            return (
                f"The content may contain offensive or insulting language (detected term(s): {flagged_str}; "
                f"confidence: {conf_pct})."
            )
        return (
            f"The content may contain offensive, vulgar, or insulting language based on multimodal context "
            f"(confidence: {conf_pct})."
        )

    if risk == "HATE":
        if flagged:
            flagged_str = ", ".join(f"'{w}'" for w in sorted(set(flagged)))
            return (
                f"The content may contain hateful or identity-targeted content (detected term(s): {flagged_str}; "
                f"confidence: {conf_pct})."
            )
        return (
            f"The content may contain hateful or identity-targeted content directed against a protected group "
            f"(confidence: {conf_pct})."
        )

    if risk == "HARASSMENT":
        if flagged:
            flagged_str = ", ".join(f"'{w}'" for w in sorted(set(flagged)))
            return (
                f"The content may target or insult an individual with hostile language (detected term(s): {flagged_str}; "
                f"confidence: {conf_pct})."
            )
        return (
            f"The content may target or insult an individual or specific person based on multimodal context "
            f"(confidence: {conf_pct})."
        )

    return f"The content was evaluated with risk label {risk} (confidence: {conf_pct})."


def generate_suggestion(
    risk_label: str,
    caption: str = "",
    ocr_text: str = "",
) -> Optional[str]:
    """
    Generates a short, actionable safer-caption suggestion for risky posts.
    Returns None for SAFE posts.
    Provides a neutralized version of the caption when possible; otherwise offers
    clear, rule-based safe guidance.
    """
    risk = (risk_label or "").upper().strip()

    if risk == "SAFE":
        return None

    # Check if caption can be directly neutralized
    neutralized = neutralize_text(caption)

    if risk == "OFFENSIVE":
        if neutralized and neutralized.lower() != caption.strip().lower():
            return f"Consider rephrasing with milder wording: \"{neutralized}\""
        return "Consider rephrasing using neutral, polite language and removing provocative or insulting words."

    if risk == "HATE":
        if neutralized and neutralized.lower() != caption.strip().lower():
            return f"Consider removing identity-targeted phrasing: \"{neutralized}\""
        return "Suggest removing hateful or identity-targeted wording and replacing it with neutral, respectful language."

    if risk == "HARASSMENT":
        if neutralized and neutralized.lower() != caption.strip().lower():
            return f"Consider reframing without personal insults: \"{neutralized}\""
        return "Suggest removing personal attacks or targeted insults, and rewriting the post in neutral language."

    return "Suggest reviewing the wording and replacing aggressive or sensitive terms with neutral language."
