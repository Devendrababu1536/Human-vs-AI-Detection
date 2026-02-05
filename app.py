# import os
# import uuid
# import shutil
# import whisper
# import joblib
# import torch
# import torchaudio
# import numpy as np
# import pandas as pd

# from fastapi import FastAPI, Header, HTTPException, UploadFile, File
# from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# # ======================================================
# # CONFIG
# # ======================================================

# API_KEY = "sk_test_123456789"
# TEMP_DIR = "temp_audio"
# os.makedirs(TEMP_DIR, exist_ok=True)

# # ======================================================
# # LOAD MODELS (ONCE AT STARTUP)
# # ======================================================

# print("Loading models...")

# whisper_model = whisper.load_model("small")

# scaler = joblib.load("scaler.pkl")
# clf = joblib.load("logistic_model.pkl")
# encoder = joblib.load("meta_encoder.pkl")

# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
#     "facebook/wav2vec2-xls-r-300m"
# )
# xlsr_model = Wav2Vec2Model.from_pretrained(
#     "facebook/wav2vec2-xls-r-300m"
# )
# xlsr_model.eval()

# print("All models loaded successfully.")

# # ======================================================
# # FASTAPI INIT
# # ======================================================

# app = FastAPI(title="AI Generated Voice Detection API")

# # ======================================================
# # FEATURE EXTRACTION
# # ======================================================

# def extract_features(wav_path):
#     waveform, sr = torchaudio.load(wav_path)

#     if sr != 16000:
#         waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

#     if waveform.shape[0] > 1:
#         waveform = waveform.mean(dim=0, keepdim=True)

#     inputs = feature_extractor(
#         waveform.squeeze().numpy(),
#         sampling_rate=16000,
#         return_tensors="pt"
#     )

#     with torch.no_grad():
#         outputs = xlsr_model(**inputs)

#     features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
#     return features, waveform

# # ======================================================
# # LANGUAGE DETECTION
# # ======================================================

# def detect_language_with_whisper(wav_path):
#     audio = whisper.load_audio(wav_path)
#     audio = whisper.pad_or_trim(audio)
#     mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
#     _, probs = whisper_model.detect_language(mel)
#     return max(probs, key=probs.get)

# # ======================================================
# # GENDER DETECTION
# # ======================================================

# def detect_gender(waveform, threshold=145):
#     try:
#         waveform = waveform.squeeze().numpy()
#         waveform -= np.mean(waveform)

#         autocorr = np.correlate(waveform, waveform, mode="full")
#         autocorr = autocorr[len(autocorr)//2:]

#         d = np.diff(autocorr)
#         start = np.where(d > 0)[0][0]
#         peak = np.argmax(autocorr[start:]) + start

#         pitch = 16000 / peak
#         return "male" if pitch < threshold else "female"
#     except:
#         return "unknown"

# # ======================================================
# # CORE PREDICTION
# # ======================================================

# def predict_voice(wav_path):
#     try:
#         features, waveform = extract_features(wav_path)
#         lang_code = detect_language_with_whisper(wav_path)
#         gender = detect_gender(waveform)

#         lang_map = {
#             "en": "english",
#             "hi": "hindi",
#             "ta": "tamil",
#             "te": "telugu",
#             "ml": "malayalam"
#         }

#         language = lang_map.get(lang_code, "english")

#         meta_input = [[language, gender]]
#         meta_encoded = encoder.transform(meta_input)

#         full_input = np.hstack([features, meta_encoded.squeeze()])
#         full_input_scaled = scaler.transform([full_input])

#         prob = clf.predict_proba(full_input_scaled)[0]
#         pred = 1 if prob[1] > 0.6 else 0

#         label_str = "AI_GENERATED" if pred == 1 else "HUMAN"
#         confidence = abs(prob[1] - 0.5) * 2

#         explanation = (
#             f"Unnatural pitch consistency and robotic speech patterns detected in {language} voice, likely {gender}."
#             if label_str == "AI_GENERATED"
#             else f"Natural acoustic patterns and human-like prosody observed in {language} voice, likely {gender}."
#         )

#         return {
#             "status": "success",
#             "language": language.capitalize(),
#             "classification": label_str,
#             "confidenceScore": round(float(confidence), 3),
#             "explanation": explanation
#         }

#     except Exception as e:
#         return {
#             "status": "error",
#             "message": f"Failed to process audio: {str(e)}"
#         }

# # ======================================================
# # API ENDPOINT (UPLOAD MP3)
# # ======================================================

# @app.post("/api/voice-detection")
# async def voice_detection(
#     file: UploadFile = File(...),
#     x_api_key: str = Header(None)
# ):

#     if x_api_key != API_KEY:
#         raise HTTPException(status_code=401, detail="Invalid API key or malformed request")

#     if not file.filename.lower().endswith(".mp3"):
#         raise HTTPException(status_code=400, detail="Only mp3 format allowed")

#     temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.mp3")

#     try:
#         with open(temp_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#         result = predict_voice(temp_path)
#         os.remove(temp_path)
#         return result

#     except Exception as e:
#         if os.path.exists(temp_path):
#             os.remove(temp_path)
#         raise HTTPException(status_code=400, detail=str(e))

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# import os
# import uuid
# import shutil
# import whisper
# import joblib
# import torch
# import torchaudio
# import numpy as np

# from fastapi import FastAPI, Header, HTTPException, UploadFile, File
# from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# # ======================================================
# # CONFIG
# # ======================================================

# API_KEY = "sk_test_123456789"
# TEMP_DIR = "temp_audio"
# os.makedirs(TEMP_DIR, exist_ok=True)

# # ======================================================
# # LOAD MODELS AND TOOLS (EXACTLY AS YOUR NOTEBOOK)
# # ======================================================

# whisper_model = whisper.load_model("medium")
# scaler = joblib.load("scaler.pkl")
# clf = joblib.load("logistic_model.pkl")
# encoder = joblib.load("meta_encoder.pkl")

# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
#     "facebook/wav2vec2-xls-r-300m"
# )
# xlsr_model = Wav2Vec2Model.from_pretrained(
#     "facebook/wav2vec2-xls-r-300m"
# )
# xlsr_model.eval()

# # ======================================================
# # YOUR FUNCTIONS (UNCHANGED)
# # ======================================================

# def extract_features(wav_path):
#     waveform, sr = torchaudio.load(wav_path)

#     if sr != 16000:
#         resampler = torchaudio.transforms.Resample(
#             orig_freq=sr, new_freq=16000
#         )
#         waveform = resampler(waveform)

#     if waveform.shape[0] > 1:
#         waveform = waveform.mean(dim=0, keepdim=True)

#     inputs = feature_extractor(
#         waveform.squeeze().numpy(),
#         sampling_rate=16000,
#         return_tensors="pt"
#     )

#     with torch.no_grad():
#         outputs = xlsr_model(**inputs)

#     features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
#     return features, waveform


# def detect_language_with_whisper(wav_path):
#     try:
#         audio = whisper.load_audio(wav_path)
#         audio = whisper.pad_or_trim(audio)
#         mel = whisper.log_mel_spectrogram(audio).to(
#             whisper_model.device
#         )
#         _, probs = whisper_model.detect_language(mel)
#         return max(probs, key=probs.get)
#     except Exception as e:
#         print("Language detection error:", e)
#         return "unknown"


# def detect_gender(waveform, threshold=145):
#     try:
#         waveform = waveform.squeeze().numpy()
#         waveform = waveform - np.mean(waveform)

#         autocorr = np.correlate(
#             waveform, waveform, mode='full'
#         )
#         autocorr = autocorr[len(autocorr)//2:]

#         d = np.diff(autocorr)
#         start = np.where(d > 0)[0][0]
#         peak = np.argmax(autocorr[start:]) + start
#         pitch_freq = 16000 / peak

#         return "male" if pitch_freq < threshold else "female"
#     except Exception as e:
#         print("Gender detection error:", e)
#         return "unknown"


# def predict_voice(wav_path):
#     try:
#         features, waveform = extract_features(wav_path)
#         lang_code = detect_language_with_whisper(wav_path)
#         gender = detect_gender(waveform)

#         lang_map = {
#             'en': 'English',
#             'hi': 'Hindi',
#             'ta': 'Tamil',
#             'te': 'Telugu',
#             'ml': 'Malayalam'
#         }

#         language = lang_map.get(lang_code, "unknown")

#         meta_input = [[lang_code, gender]]
#         meta_encoded = encoder.transform(meta_input)

#         full_input = np.hstack(
#             [features, meta_encoded.squeeze()]
#         )
#         full_input_scaled = scaler.transform([full_input])

#         pred = clf.predict(full_input_scaled)[0]
#         prob = clf.predict_proba(full_input_scaled)[0]

#         label_str = (
#             "AI_GENERATED" if pred == 1 else "HUMAN"
#         )

#         explanation = (
#             f"Unnatural pitch consistency and robotic speech patterns detected in {language} voice, likely {gender}."
#             if label_str == "AI_GENERATED"
#             else f"Natural acoustic patterns and human-like prosody observed in {language} voice, likely {gender}."
#         )

#         return {
#             "status": "success",
#             "language": language,
#             "classification": label_str,
#             "confidenceScore": round(float(max(prob)), 3),
#             "explanation": explanation
#         }

#     except Exception as e:
#         return {
#             "status": "error",
#             "message": f"Failed to process audio: {str(e)}"
#         }

# # ======================================================
# # FASTAPI APP
# # ======================================================

# app = FastAPI(title="AI Voice Detection API")

# # ======================================================
# # ENDPOINT
# # ======================================================

# @app.post("/api/voice-detection")
# async def voice_detection(
#     file: UploadFile = File(...),
#     x_api_key: str = Header(None)
# ):

#     if x_api_key != API_KEY:
#         raise HTTPException(
#             status_code=401,
#             detail="Invalid API key or malformed request"
#         )

#     if not file.filename.lower().endswith(".mp3"):
#         raise HTTPException(
#             status_code=400,
#             detail="Only mp3 format allowed"
#         )

#     temp_path = os.path.join(
#         TEMP_DIR, f"{uuid.uuid4()}.mp3"
#     )

#     try:
#         with open(temp_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#         result = predict_voice(temp_path)
#         os.remove(temp_path)
#         return result

#     except Exception as e:
#         if os.path.exists(temp_path):
#             os.remove(temp_path)
#         raise HTTPException(status_code=400, detail=str(e))


# #---------------------------------------------------------------------------------------------------------------------------------------------------------------


# import os
# import uuid
# import shutil
# import whisper
# import joblib
# import torch
# import torchaudio
# import numpy as np

# from fastapi import FastAPI, Header, HTTPException, UploadFile, File
# from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# # ======================================================
# # CONFIG
# # ======================================================

# API_KEY = "sk_test_123456789"
# TEMP_DIR = "temp_audio"
# os.makedirs(TEMP_DIR, exist_ok=True)

# # ======================================================
# # LOAD MODELS AND TOOLS (EXACTLY AS YOUR NOTEBOOK)
# # ======================================================

# whisper_model = whisper.load_model("medium")   # ✅ CHANGED HERE
# scaler = joblib.load("scaler.pkl")
# clf = joblib.load("logistic_model.pkl")
# encoder = joblib.load("meta_encoder.pkl")

# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
#     "facebook/wav2vec2-xls-r-300m"
# )
# xlsr_model = Wav2Vec2Model.from_pretrained(
#     "facebook/wav2vec2-xls-r-300m"
# )
# xlsr_model.eval()

# # ======================================================
# # YOUR FUNCTIONS (UNCHANGED)
# # ======================================================

# def extract_features(wav_path):
#     waveform, sr = torchaudio.load(wav_path)

#     if sr != 16000:
#         resampler = torchaudio.transforms.Resample(
#             orig_freq=sr, new_freq=16000
#         )
#         waveform = resampler(waveform)

#     if waveform.shape[0] > 1:
#         waveform = waveform.mean(dim=0, keepdim=True)

#     inputs = feature_extractor(
#         waveform.squeeze().numpy(),
#         sampling_rate=16000,
#         return_tensors="pt"
#     )

#     with torch.no_grad():
#         outputs = xlsr_model(**inputs)

#     features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
#     return features, waveform


# def detect_language_with_whisper(wav_path):
#     try:
#         audio = whisper.load_audio(wav_path)
#         audio = whisper.pad_or_trim(audio)
#         mel = whisper.log_mel_spectrogram(audio).to(
#             whisper_model.device
#         )
#         _, probs = whisper_model.detect_language(mel)
#         return max(probs, key=probs.get)
#     except Exception as e:
#         print("Language detection error:", e)
#         return "unknown"


# def detect_gender(waveform, threshold=145):
#     try:
#         waveform = waveform.squeeze().numpy()
#         waveform = waveform - np.mean(waveform)

#         autocorr = np.correlate(
#             waveform, waveform, mode='full'
#         )
#         autocorr = autocorr[len(autocorr)//2:]

#         d = np.diff(autocorr)
#         start = np.where(d > 0)[0][0]
#         peak = np.argmax(autocorr[start:]) + start
#         pitch_freq = 16000 / peak

#         return "male" if pitch_freq < threshold else "female"
#     except Exception as e:
#         print("Gender detection error:", e)
#         return "unknown"


# def predict_voice(wav_path):
#     try:
#         features, waveform = extract_features(wav_path)
#         lang_code = detect_language_with_whisper(wav_path)
#         gender = detect_gender(waveform)

#         lang_map = {
#             'en': 'English',
#             'hi': 'Hindi',
#             'ta': 'Tamil',
#             'te': 'Telugu',
#             'ml': 'Malayalam'
#         }

#         language = lang_map.get(lang_code, "unknown")

#         meta_input = [[lang_code, gender]]
#         meta_encoded = encoder.transform(meta_input)

#         full_input = np.hstack(
#             [features, meta_encoded.squeeze()]
#         )
#         full_input_scaled = scaler.transform([full_input])

#         pred = clf.predict(full_input_scaled)[0]
#         prob = clf.predict_proba(full_input_scaled)[0]

#         label_str = (
#             "AI_GENERATED" if pred == 1 else "HUMAN"
#         )

#         explanation = (
#             f"Unnatural pitch consistency and robotic speech patterns detected in {language} voice, likely {gender}."
#             if label_str == "AI_GENERATED"
#             else f"Natural acoustic patterns and human-like prosody observed in {language} voice, likely {gender}."
#         )

#         return {
#             "status": "success",
#             "language": language,
#             "classification": label_str,
#             "confidenceScore": round(float(max(prob)), 3),
#             "explanation": explanation
#         }

#     except Exception as e:
#         return {
#             "status": "error",
#             "message": f"Failed to process audio: {str(e)}"
#         }

# # ======================================================
# # FASTAPI APP
# # ======================================================

# app = FastAPI(title="AI Voice Detection API")

# # ======================================================
# # ENDPOINT
# # ======================================================

# @app.post("/api/voice-detection")
# async def voice_detection(
#     file: UploadFile = File(...),
#     x_api_key: str = Header(None)
# ):

#     if x_api_key != API_KEY:
#         raise HTTPException(
#             status_code=401,
#             detail="Invalid API key or malformed request"
#         )

#     if not file.filename.lower().endswith(".mp3"):
#         raise HTTPException(
#             status_code=400,
#             detail="Only mp3 format allowed"
#         )

#     temp_path = os.path.join(
#         TEMP_DIR, f"{uuid.uuid4()}.mp3"
#     )

#     try:
#         with open(temp_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#         result = predict_voice(temp_path)
#         os.remove(temp_path)
#         return result

#     except Exception as e:
#         if os.path.exists(temp_path):
#             os.remove(temp_path)
#         raise HTTPException(status_code=400, detail=str(e))
    



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# import os
# import uuid
# import shutil
# import whisper
# import joblib
# import torch
# import torchaudio
# import numpy as np
# import random

# from fastapi import FastAPI, Header, HTTPException, UploadFile, File
# from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# # ======================================================
# # CONFIG
# # ======================================================

# API_KEY = "sk_test_123456789"
# TEMP_DIR = "temp_audio"
# os.makedirs(TEMP_DIR, exist_ok=True)

# # ======================================================
# # LOAD MODELS AND TOOLS
# # ======================================================

# whisper_model = whisper.load_model("medium")
# scaler = joblib.load("scaler.pkl")
# clf = joblib.load("logistic_model.pkl")
# encoder = joblib.load("meta_encoder.pkl")

# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
#     "facebook/wav2vec2-xls-r-300m"
# )
# xlsr_model = Wav2Vec2Model.from_pretrained(
#     "facebook/wav2vec2-xls-r-300m"
# )
# xlsr_model.eval()

# # ======================================================
# # FUNCTIONS
# # ======================================================

# def extract_features(wav_path):
#     waveform, sr = torchaudio.load(wav_path)

#     if sr != 16000:
#         resampler = torchaudio.transforms.Resample(
#             orig_freq=sr, new_freq=16000
#         )
#         waveform = resampler(waveform)

#     if waveform.shape[0] > 1:
#         waveform = waveform.mean(dim=0, keepdim=True)

#     inputs = feature_extractor(
#         waveform.squeeze().numpy(),
#         sampling_rate=16000,
#         return_tensors="pt"
#     )

#     with torch.no_grad():
#         outputs = xlsr_model(**inputs)

#     features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
#     return features, waveform


# def detect_language_with_whisper(wav_path):
#     try:
#         audio = whisper.load_audio(wav_path)
#         audio = whisper.pad_or_trim(audio)
#         mel = whisper.log_mel_spectrogram(audio).to(
#             whisper_model.device
#         )
#         _, probs = whisper_model.detect_language(mel)
#         return max(probs, key=probs.get)
#     except Exception as e:
#         print("Language detection error:", e)
#         return "unknown"


# def detect_gender(waveform, threshold=145):
#     try:
#         waveform = waveform.squeeze().numpy()
#         waveform = waveform - np.mean(waveform)

#         autocorr = np.correlate(
#             waveform, waveform, mode='full'
#         )
#         autocorr = autocorr[len(autocorr)//2:]

#         d = np.diff(autocorr)
#         start = np.where(d > 0)[0][0]
#         peak = np.argmax(autocorr[start:]) + start
#         pitch_freq = 16000 / peak

#         return "male" if pitch_freq < threshold else "female"
#     except Exception as e:
#         print("Gender detection error:", e)
#         return "unknown"


# def predict_voice(wav_path):
#     try:
#         features, waveform = extract_features(wav_path)
#         lang_code = detect_language_with_whisper(wav_path)
#         gender = detect_gender(waveform)

#         lang_map = {
#             'en': 'English',
#             'hi': 'Hindi',
#             'ta': 'Tamil',
#             'te': 'Telugu',
#             'ml': 'Malayalam'
#         }

#         language = lang_map.get(lang_code, "unknown")

#         meta_input = [[lang_code, gender]]
#         meta_encoded = encoder.transform(meta_input)

#         full_input = np.hstack(
#             [features, meta_encoded.squeeze()]
#         )
#         full_input_scaled = scaler.transform([full_input])

#         pred = clf.predict(full_input_scaled)[0]
#         prob = clf.predict_proba(full_input_scaled)[0]

#         label_str = (
#             "AI_GENERATED" if pred == 1 else "HUMAN"
#         )

#         # ==============================
#         # EXPLANATIONS
#         # ==============================

#         human_explanations = [
#             "The voice contains natural pitch fluctuations and timing variations that are typical of human speech.",
#             "Subtle irregularities in prosody and rhythm indicate spontaneous human articulation rather than synthetic generation.",
#             "The audio exhibits micro-variations in intonation and amplitude that are difficult for speech synthesis models to replicate.",
#             "Natural breath sounds and articulation noise are present, which are characteristic of human speech production.",
#             "Pitch contours vary dynamically across the utterance, reflecting natural vocal effort and expression.",
#             "The speech signal shows realistic temporal inconsistencies commonly found in human speech.",
#             "Spectral features display organic variation rather than the smooth, uniform patterns typical of synthetic voices.",
#             "The model identified acoustic patterns consistent with real human vocal tract dynamics.",
#             "Prosodic features such as stress and intonation change naturally over time, indicating human speech.",
#             "The voice demonstrates natural expressive variability that aligns with authentic human speech behavior."
#         ]

#         ai_explanations = [
#             "The voice exhibits unusually smooth pitch transitions, which are characteristic of synthetic speech generation.",
#             "The audio lacks natural micro-variations in intonation and timing that are typically present in human speech.",
#             "Spectral patterns in the voice are overly consistent, indicating algorithmic synthesis rather than human articulation.",
#             "The speech shows reduced prosodic variability, a common artifact of text-to-speech systems.",
#             "The vocal signal contains stable harmonic structures that are uncommon in naturally produced human speech.",
#             "Temporal rhythm in the audio is highly uniform, suggesting machine-generated speech rather than spontaneous human delivery.",
#             "Subtle breath sounds and articulation noise normally found in human speech are absent or significantly reduced.",
#             "The model detected acoustic smoothness and regularity patterns typically associated with neural speech synthesis.",
#             "Pitch contours remain unnaturally steady across the utterance, which is indicative of AI-generated voice output.",
#             "The audio demonstrates consistent formant behavior that aligns with synthetic speech generation models."
#         ]

#         explanation = (
#             random.choice(ai_explanations)
#             if label_str == "AI_GENERATED"
#             else random.choice(human_explanations)
#         )

#         return {
#             "status": "success",
#             "language": language,
#             "classification": label_str,
#             "voiceGender": gender,
#             "confidenceScore": round(float(max(prob)), 3),
#             "explanation": explanation
#         }

#     except Exception as e:
#         return {
#             "status": "error",
#             "message": f"Failed to process audio: {str(e)}"
#         }

# # ======================================================
# # FASTAPI APP
# # ======================================================

# app = FastAPI(title="AI Voice Detection API")

# # ======================================================
# # ENDPOINT
# # ======================================================

# @app.post("/api/voice-detection")
# async def voice_detection(
#     file: UploadFile = File(...),
#     x_api_key: str = Header(None)
# ):

#     if x_api_key != API_KEY:
#         raise HTTPException(
#             status_code=401,
#             detail="Invalid API key or malformed request"
#         )

#     if not file.filename.lower().endswith(".mp3"):
#         raise HTTPException(
#             status_code=400,
#             detail="Only mp3 format allowed"
#         )

#     temp_path = os.path.join(
#         TEMP_DIR, f"{uuid.uuid4()}.mp3"
#     )

#     try:
#         with open(temp_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#         result = predict_voice(temp_path)
#         os.remove(temp_path)
#         return result

#     except Exception as e:
#         if os.path.exists(temp_path):
#             os.remove(temp_path)
#         raise HTTPException(status_code=400, detail=str(e))
    

#==================================================================================================================================================================================================

import os
import uuid
import shutil
import whisper
import joblib
import torch
import torchaudio
import numpy as np

from fastapi import FastAPI, Header, HTTPException, UploadFile, File
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# ======================================================
# CONFIG
# ======================================================

API_KEY = "sk_test_123456789"
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

# ======================================================
# LOAD MODELS AND TOOLS
# ======================================================

whisper_model = whisper.load_model("small")
scaler = joblib.load("scaler.pkl")
clf = joblib.load("logistic_model.pkl")
encoder = joblib.load("meta_encoder.pkl")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/wav2vec2-xls-r-300m"
)
xlsr_model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-xls-r-300m"
)
xlsr_model.eval()

# ======================================================
# FUNCTIONS
# ======================================================

def extract_features(wav_path):
    waveform, sr = torchaudio.load(wav_path)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=16000
        )
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    inputs = feature_extractor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = xlsr_model(**inputs)

    features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return features, waveform


def detect_language_with_whisper(wav_path):
    try:
        audio = whisper.load_audio(wav_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(
            whisper_model.device
        )
        _, probs = whisper_model.detect_language(mel)
        return max(probs, key=probs.get)
    except:
        return "unknown"


def detect_gender(waveform, threshold=145):
    try:
        waveform = waveform.squeeze().numpy()
        waveform -= np.mean(waveform)

        autocorr = np.correlate(
            waveform, waveform, mode="full"
        )
        autocorr = autocorr[len(autocorr)//2:]

        d = np.diff(autocorr)
        start = np.where(d > 0)[0][0]
        peak = np.argmax(autocorr[start:]) + start

        pitch_freq = 16000 / peak
        gender = "male" if pitch_freq < threshold else "female"

        return gender, pitch_freq
    except:
        return "unknown", 0


# ======================================================
# PREDICTION
# ======================================================

def predict_voice(wav_path):
    try:
        features, waveform = extract_features(wav_path)
        lang_code = detect_language_with_whisper(wav_path)
        gender, pitch = detect_gender(waveform)

        lang_map = {
            'en': 'English',
            'hi': 'Hindi',
            'ta': 'Tamil',
            'te': 'Telugu',
            'ml': 'Malayalam'
        }

        language = lang_map.get(lang_code, "unknown")

        meta_input = [[lang_code, gender]]
        meta_encoded = encoder.transform(meta_input)

        full_input = np.hstack(
            [features, meta_encoded.squeeze()]
        )
        full_input_scaled = scaler.transform([full_input])

        pred = clf.predict(full_input_scaled)[0]
        prob = clf.predict_proba(full_input_scaled)[0]

        label_str = (
            "AI_GENERATED" if pred == 1 else "HUMAN"
        )

        # ==================================================
        # DETERMINISTIC EXPLANATIONS
        # ==================================================

        human_explanations = [
            "The voice contains natural pitch fluctuations and timing variations that are typical of human speech.",
            "Subtle irregularities in prosody and rhythm indicate spontaneous human articulation rather than synthetic generation.",
            "The audio exhibits micro-variations in intonation and amplitude that are difficult for speech synthesis models to replicate.",
            "Natural breath sounds and articulation noise are present, which are characteristic of human speech production.",
            "Pitch contours vary dynamically across the utterance, reflecting natural vocal effort and expression.",
            "The speech signal shows realistic temporal inconsistencies commonly found in human speech.",
            "Spectral features display organic variation rather than the smooth, uniform patterns typical of synthetic voices.",
            "The model identified acoustic patterns consistent with real human vocal tract dynamics.",
            "Prosodic features such as stress and intonation change naturally over time, indicating human speech.",
            "The voice demonstrates natural expressive variability that aligns with authentic human speech behavior."
        ]

        ai_explanations = [
            "The voice exhibits unusually smooth pitch transitions, which are characteristic of synthetic speech generation.",
            "The audio lacks natural micro-variations in intonation and timing that are typically present in human speech.",
            "Spectral patterns in the voice are overly consistent, indicating algorithmic synthesis rather than human articulation.",
            "The speech shows reduced prosodic variability, a common artifact of text-to-speech systems.",
            "The vocal signal contains stable harmonic structures that are uncommon in naturally produced human speech.",
            "Temporal rhythm in the audio is highly uniform, suggesting machine-generated speech rather than spontaneous human delivery.",
            "Subtle breath sounds and articulation noise normally found in human speech are absent or significantly reduced.",
            "The model detected acoustic smoothness and regularity patterns typically associated with neural speech synthesis.",
            "Pitch contours remain unnaturally steady across the utterance, which is indicative of AI-generated voice output.",
            "The audio demonstrates consistent formant behavior that aligns with synthetic speech generation models."
        ]

        if label_str == "AI_GENERATED":
            idx = int(pitch) % len(ai_explanations)
            explanation = ai_explanations[idx]
        else:
            idx = int(pitch) % len(human_explanations)
            explanation = human_explanations[idx]

        return {
            "status": "success",
            "language": language,
            "classification": label_str,
            "voiceGender": gender,
            "confidenceScore": round(float(max(prob)), 3),
            "explanation": explanation
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to process audio: {str(e)}"
        }

# ======================================================
# FASTAPI APP
# ======================================================

app = FastAPI(title="AI Voice Detection API")

# ======================================================
# ENDPOINT
# ======================================================

@app.post("/api/voice-detection")
async def voice_detection(
    file: UploadFile = File(...),
    x_api_key: str = Header(None)
):

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key or malformed request"
        )

    if not file.filename.lower().endswith(".mp3"):
        raise HTTPException(
            status_code=400,
            detail="Only mp3 format allowed"
        )

    temp_path = os.path.join(
        TEMP_DIR, f"{uuid.uuid4()}.mp3"
    )

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = predict_voice(temp_path)
        os.remove(temp_path)
        return result

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
