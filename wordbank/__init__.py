"""Word/phrase-bank generation pipeline.

Builds a bank of audio clips two ways: the **Gemini Batch** path (`gemini_batch.py`,
the en-us word bank) and the **local OmniVoice** path (`pipeline.py` + `backends/`,
accent phrase banks, optionally Whisper-verified). See WORDBANK_PLAN.md.
"""
