"""Synchronous single-phrase Gemini TTS → raw PCM. Used to author voice-clone
references (the UI "voice studio" and the `gemini-voice` CLI). Distinct from
`gemini_batch.py`, which builds the large async word bank.

Lazily imports google-genai; needs GEMINI_API_KEY in the environment.
"""
import io
import wave

SAMPLE_RATE = 24_000  # Gemini TTS emits 24 kHz / 16-bit mono PCM
DEFAULT_MODEL = "gemini-2.5-flash-preview-tts"

# Prebuilt Gemini TTS voices: (name, style descriptor, gender). Voices are NOT
# language-specific — the model auto-detects language from the input text, so
# any voice speaks any supported language. The accent/language of a reference
# comes from the phrase you write (and optional `instruct`), not the voice.
# Gender is the community-perceived gender (Google doesn't officially assign one
# — the names are mythological); 14 female / 16 male.
# Source: https://ai.google.dev/gemini-api/docs/speech-generation
VOICES = [
    ("Zephyr", "Bright", "female"), ("Puck", "Upbeat", "male"),
    ("Charon", "Informative", "male"), ("Kore", "Firm", "female"),
    ("Fenrir", "Excitable", "male"), ("Leda", "Youthful", "female"),
    ("Orus", "Firm", "male"), ("Aoede", "Breezy", "female"),
    ("Callirrhoe", "Easy-going", "female"), ("Autonoe", "Bright", "female"),
    ("Enceladus", "Breathy", "male"), ("Iapetus", "Clear", "male"),
    ("Umbriel", "Easy-going", "male"), ("Algieba", "Smooth", "male"),
    ("Despina", "Smooth", "female"), ("Erinome", "Clear", "female"),
    ("Algenib", "Gravelly", "male"), ("Rasalgethi", "Informative", "male"),
    ("Laomedeia", "Upbeat", "female"), ("Achernar", "Soft", "female"),
    ("Alnilam", "Firm", "male"), ("Schedar", "Even", "male"),
    ("Gacrux", "Mature", "female"), ("Pulcherrima", "Forward", "female"),
    ("Achird", "Friendly", "male"), ("Zubenelgenubi", "Casual", "male"),
    ("Vindemiatrix", "Gentle", "female"), ("Sadachbia", "Lively", "male"),
    ("Sadaltager", "Knowledgeable", "male"), ("Sulafat", "Warm", "female"),
]


def synthesize_pcm(text: str, voice: str = "Kore", instruct: str = "",
                   model: str = DEFAULT_MODEL) -> bytes:
    """Speak `text` with a Gemini prebuilt voice; return raw 24 kHz PCM bytes.

    `instruct` is style/accent direction (e.g. "British accent") — prepended,
    not spoken. Raises if Gemini returns no audio.
    """
    from google import genai
    from google.genai import types

    client = genai.Client()
    prompt = f"{instruct}: {text}" if instruct else text
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))),
        ),
    )
    return resp.candidates[0].content.parts[0].inline_data.data


def pcm_to_wav_bytes(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container (in memory) — no ffmpeg."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()
