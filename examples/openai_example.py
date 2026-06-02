"""
AuditorAI — OpenAI API example
Demonstrates wrapping OpenAI GPT models with AuditorAI.
Run: OPENAI_API_KEY=sk-... python examples/openai_example.py
Requires: pip install auditorai[openai]

Note: This uses gpt-4o-mini to minimize API costs during testing.
Each run makes ~25 API calls (20 val + 5 test).
"""

import numpy as np
from auditorai import AuditorSystem, wrap


# ── Step 1: Define a response parser ─────────────────────────────
# The parser converts GPT's text output into (class_index, confidence).
# Your prompt should instruct the model to output in a parseable format.
def parse_sentiment(response_text: str) -> tuple:
    """
    Parse GPT response like "Class: 1, Confidence: 0.87"
    Returns (class_index, confidence_score).
    """
    response = response_text.strip().lower()

    # Extract class
    class_idx = 0
    if "class: 1" in response or "class:1" in response:
        class_idx = 1
    elif "class: 0" in response or "class:0" in response:
        class_idx = 0
    elif "positive" in response:
        class_idx = 1
    elif "negative" in response:
        class_idx = 0

    # Extract confidence
    confidence = 0.7  # default
    import re
    conf_match = re.search(r"confidence[:\s]+([0-9.]+)", response)
    if conf_match:
        try:
            confidence = float(conf_match.group(1))
            confidence = min(max(confidence, 0.0), 1.0)  # clamp to [0, 1]
        except ValueError:
            confidence = 0.7

    return (class_idx, confidence)


# ── Step 2: Wrap the model ───────────────────────────────────────
# wrap() creates an APIAdapter that:
#   - Calls the OpenAI API for each input
#   - Uses parse_sentiment() to convert text to (class, confidence)
#   - Builds probability vectors from confidence scores
#   - Handles rate limiting with exponential backoff
adapter = wrap(
    "gpt-4o-mini",
    adapter_type="openai",
    parse_response=parse_sentiment,
    n_classes=2,
    system_prompt=(
        "Classify the following text as positive (1) or negative (0) sentiment. "
        "Reply with EXACTLY this format and nothing else:\n"
        "Class: <0 or 1>, Confidence: <0.00 to 1.00>"
    ),
    batch_size=3,  # conservative to avoid rate limits
)


# ── Step 3: Prepare a small dataset ──────────────────────────────
# We use a small dataset to minimize API costs.
# In production, use your real data.
texts_val = [
    "I love this product, absolutely amazing!",
    "Terrible experience, total waste.",
    "It was okay, nothing to write home about.",
    "Best purchase I've ever made!",
    "Awful quality, completely disappointed.",
    "Pretty good, I'd buy it again.",
    "Worst customer service ever.",
    "Outstanding results, highly recommend!",
    "Not great, not terrible.",
    "Absolutely fantastic, exceeded expectations!",
    "Broken within a week, very frustrated.",
    "Decent value for the price.",
    "I regret buying this.",
    "Wonderful, my family loves it!",
    "Do not buy this, it's a scam.",
    "Solid product, works as advertised.",
    "Garbage, returning immediately.",
    "Pleasantly surprised by the quality!",
    "Meh, expected better for the price.",
    "Five stars, couldn't be happier!",
]

y_val = np.array([1, 0, 0, 1, 0, 1, 0, 1, 0, 1,
                   0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

texts_test = [
    "This is genuinely excellent!",
    "Worst thing I've ever used.",
    "It's fine, nothing special.",
    "Incredible quality!",
    "Total disappointment.",
]

y_test = np.array([1, 0, 0, 1, 0])


# ── Step 4: Create and train the auditor ─────────────────────────
print("Training auditor (this makes ~20 API calls)...")
system = AuditorSystem(adapter)
system.train(texts_val, y_val)

# ── Step 5: Predict ──────────────────────────────────────────────
print("\nMaking predictions (this makes ~5 API calls)...")
result = system.predict(texts_test)
print(f"Predictions: {result['ai_predictions']}")
print(f"P(wrong):    {np.round(result['p_wrong'], 3)}")
print(f"Show mask:   {result['show_mask']}")
print(f"Suppress:    {result['suppress_mask']}")
print("\nOpenAI example complete!")
