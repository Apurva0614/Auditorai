"""
AuditorAI — HuggingFace example
Demonstrates wrapping a HuggingFace transformers pipeline with AuditorAI.
Run: python examples/huggingface_example.py
Requires: pip install auditorai[hf]

Note: This example uses a fake dataset to run without downloading
large datasets. In production, use your real text data.
"""

try:
    from transformers import pipeline
except ImportError:
    print("This example requires transformers. Install with: pip install transformers torch")
    raise SystemExit(1)

import numpy as np
from auditorai import AuditorSystem, wrap


# ── Step 1: Create a HuggingFace pipeline ────────────────────────
# Using a small, fast sentiment analysis model
print("Loading HuggingFace pipeline...")
pipe = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1,  # CPU; use device=0 for GPU
)


# ── Step 2: Create a small dataset ───────────────────────────────
# In production you'd use your real text data. Here we use a small
# hand-crafted sentiment dataset for demonstration.
texts_val = [
    "I absolutely loved this movie, it was fantastic!",
    "This is the worst product I have ever bought.",
    "The service was okay, nothing special.",
    "Amazing experience, would highly recommend!",
    "Terrible quality, complete waste of money.",
    "It was fine, not great but not bad either.",
    "Outstanding performance, exceeded expectations!",
    "Awful, I want my money back immediately.",
    "Pretty good overall, I'm satisfied.",
    "Disappointing, did not meet expectations at all.",
    "Wonderful staff, very friendly and helpful.",
    "The food was cold and tasteless.",
    "Excellent value for the price!",
    "Not worth the price, very overpriced.",
    "Great atmosphere and lovely decor.",
    "I had a horrible experience, never again.",
    "Superb quality, I'm very impressed.",
    "Mediocre at best, expected much more.",
    "Highly recommend this to everyone!",
    "Completely useless product, total scam.",
]

# Labels: 1 = positive, 0 = negative
y_val = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 0,
                   1, 0, 1, 0, 1, 0, 1, 0, 1, 0])


# ── Step 3: Wrap the pipeline with AuditorAI ─────────────────────
# The HuggingFaceAdapter handles:
#   - Converting pipeline output ({"label": "POSITIVE", "score": 0.99})
#     to a probability matrix
#   - Batched inference
#   - Label discovery
adapter = wrap(pipe, adapter_type="huggingface")

# Check it works
probas = adapter.predict_proba(texts_val[:3])
preds = adapter.predict(texts_val[:3])
print(f"\nSample predictions: {preds}")
print(f"Sample probabilities:\n{probas}")


# ── Step 4: Create and train the auditor ─────────────────────────
# Note: In production you'd use a much larger validation set.
# The auditor needs enough data to learn failure patterns.
system = AuditorSystem(adapter)
system.train(texts_val, y_val)

# ── Step 5: Make predictions ─────────────────────────────────────
texts_test = [
    "This is genuinely great, I love it!",
    "Worst purchase of my life.",
    "It's acceptable, nothing more.",
    "Incredible quality and fast delivery!",
    "Broken on arrival, terrible experience.",
]

result = system.predict(texts_test)
print(f"\nPredictions: {result['ai_predictions']}")
print(f"P(wrong):    {result['p_wrong']}")
print(f"Show mask:   {result['show_mask']}")
print(f"Suppress:    {result['suppress_mask']}")
print("\nHuggingFace example complete!")
