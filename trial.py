from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import time
import shutil
import asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
import librosa
import soundfile as sf
import psycopg2
from psycopg2 import OperationalError as PostgresError
from psycopg2.pool import SimpleConnectionPool
from faster_whisper import WhisperModel
from transformers import pipeline
import re
from typing import Tuple, List, Dict, Set

# ========= INIT & CONFIG =========
app = FastAPI()
logging.basicConfig(level=logging.INFO,
                   format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
executor = ThreadPoolExecutor(max_workers=4)
MODEL_CACHE_DIR = "./model_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
MAX_CHUNK_DURATION = 60  # seconds
MAX_SUMMARY_TIME = 5.0  # seconds

# ========= MEDICAL CONFIG =========
MEDICAL_KEYWORDS = {
    'patient_reports': [
        r'i feel', r'my \w+', r'have \w+', r'symptom', 
        r'pain', r'problem', r'complaint', r'suffer',
        r'not well', r'unwell', r'issue', r'experience',
        r'ache', r'discomfort', r'dizzy', r'nauseous'
    ],
    'doctor_advice': [
        r'recommend', r'advice', r'should', r'take',
        r'eat', r'medicine', r'tablet', r'diet',
        r'exercise', r'follow up', r'check', r'avoid',
        r'prescribe', r'suggest', r'consult', r'increase',
        r'decrease', r'maintain', r'reduce', r'change'
    ],
    'medical_terms': [
        r'diabet', r'sugar', r'blood', r'test',
        r'pressure', r'level', r'result', r'report',
        r'fasting', r'hb', r'cholest', r'treatment',
        r'fever', r'cough', r'headache', r'infection',
        r'allerg', r'asthma', r'arthritis', r'migraine',
        r'infection', r'inflammat', r'injury', r'x\-ray',
        r'scan', r'clinic', r'hospital', r'pharmacy'
    ]
}

# ========= ADVANCED MEDICAL SUMMARIZER =========
class MedicalSummarizer:
    POLITE_PHRASES = [
        "good morning", "good afternoon", "good evening", "hello", "hi",
        "please", "thank you", "come in", "tell me", "welcome"
    ]

    MEDICAL_KEYWORDS = [
        "pain", "ache", "fever", "cough", "cold", "tired", "hungry",
        "sleepy", "dizzy", "diabet", "sugar", "blood", "pressure",
        "tablet", "medicine", "drug", "dose", "diet", "exercise",
        "treat", "therapy", "test", "scan", "report"
    ]

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def _remove_repetitions(text: str) -> str:
        lines = text.split('\n')
        cleaned = []
        prev_norm = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            norm_line = MedicalSummarizer._normalize_text(line)
            if norm_line != prev_norm:
                cleaned.append(line)
                prev_norm = norm_line

        return '\n'.join(cleaned)

    @staticmethod
    def _merge_consecutive_speakers(text: str) -> str:
        lines = text.split('\n')
        merged = []
        current_speaker = None
        current_content = []

        speaker_pattern = re.compile(
            r'^(patient|doctor|physician)\s*(reports|recommends|advised|said)?\s*:?\s*',
            re.IGNORECASE
        )

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = speaker_pattern.match(line)
            if match:
                speaker_type = match.group(1).lower()
                content = line[match.end():].strip()
                new_speaker = (
                    "Patient reports:" if speaker_type == "patient"
                    else "Doctor recommends:"
                )

                if new_speaker == current_speaker:
                    current_content.append(content)
                else:
                    if current_speaker:
                        merged.append(f"{current_speaker} {'; '.join(current_content)}")
                    current_speaker = new_speaker
                    current_content = [content]
            else:
                if current_speaker:
                    merged.append(f"{current_speaker} {'; '.join(current_content)}")
                    current_speaker = None
                    current_content = []
                merged.append(line)

        if current_speaker:
            merged.append(f"{current_speaker} {'; '.join(current_content)}")

        return '\n'.join(merged)

    @staticmethod
    def _restore_punctuation(text: str) -> str:
        markers = [
            r"\bdoctor\b", r"\bpatient\b", r"\bnurse\b",
            r"\bbut\b", r"\band\b", r"\bso\b", r"\btherefore\b"
        ]
        for pat in markers:
            text = re.sub(f"(\\s+{pat}\\s+)", r". \1", text, flags=re.IGNORECASE)
        text = re.sub(r"\s{2,}", " ", text).strip()
        if not text.endswith((".", "!", "?")):
            text += "."
        return text

    @classmethod
    def _strip_polite_phrases(cls, text: str) -> str:
        pattern = re.compile(r"|".join(map(re.escape, cls.POLITE_PHRASES)), re.I)
        return pattern.sub("", text)

    @staticmethod
    def _is_medical_sentence(sentence: str) -> bool:
        s = sentence.lower()
        return any(k in s for k in MedicalSummarizer.MEDICAL_KEYWORDS)

    @staticmethod
    def extract_dialog_components(text: str) -> Tuple[List[str], List[str]]:
        text = MedicalSummarizer._strip_polite_phrases(text)
        text = MedicalSummarizer._restore_punctuation(text)

        patient_symptoms = []
        doctor_advice = []
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sent in sentences:
            sent = sent.strip()
            if not sent or len(sent.split()) < 3:
                continue
            if not MedicalSummarizer._is_medical_sentence(sent):
                continue

            clean_sent = re.sub(r'\b(I|my|me|you|your|mine)\b', '', sent, flags=re.I).strip()
            clean_sent = clean_sent[0].upper() + clean_sent[1:] if clean_sent else sent

            if re.search(r"\b(feel|felt|feeling|symptom|pain|tired|hungry|dizzy|sleepy)\b", sent, re.I):
                if len(patient_symptoms) < 4:
                    patient_symptoms.append(clean_sent)
            elif re.search(r"\b(recommend|advise|should|prescribe|take|need to|try|avoid)\b", sent, re.I):
                if len(doctor_advice) < 4:
                    doctor_advice.append(clean_sent)

        return patient_symptoms, doctor_advice

    @classmethod
    def create_structured_summary(cls, symptoms: List[str], advice: List[str]) -> str:
        summary_parts = []

        if symptoms:
            seen = set()
            unique_symptoms = []
            for symptom in symptoms:
                norm = cls._normalize_text(symptom)
                if norm not in seen:
                    seen.add(norm)
                    unique_symptoms.append(symptom)
            summary_parts.append(f"Patient reports: {'; '.join(unique_symptoms[:3])}")

        if advice:
            seen = set()
            unique_advice = []
            for adv in advice:
                norm = cls._normalize_text(adv)
                if norm not in seen:
                    seen.add(norm)
                    unique_advice.append(adv)
            summary_parts.append(f"Doctor recommends: {'; '.join(unique_advice[:3])}")

        return '\n'.join(summary_parts)

    @classmethod
    def summarize(cls, text: str) -> str:
        try:
            text = cls._remove_repetitions(text)
            text = cls._strip_polite_phrases(text)
            text = cls._restore_punctuation(text)

            symptoms, advice = cls.extract_dialog_components(text)

            if symptoms or advice:
                summary = cls.create_structured_summary(symptoms, advice)
                summary = cls._merge_consecutive_speakers(summary)
                return cls._remove_repetitions(summary)

            # fallback abstractive summary
            bart = app.state.models["summarizer"]
            abstr = bart(text, max_length=120, min_length=40, do_sample=False)[0]["summary_text"]
            return abstr.strip()

        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            clean_text = re.sub(r'\s+', ' ', text).strip()
            return clean_text[:300] + ("..." if len(clean_text) > 300 else "")

# ========= MODEL LOADING =========
async def load_models() -> dict:
    """Asynchronous model loading with timeout"""
    models = {}
    try:
        logger.info("Loading English model...")
        models["english"] = await asyncio.wait_for(
            asyncio.to_thread(
                WhisperModel,
                "medium",
                device=DEVICE,
                compute_type=COMPUTE_TYPE
            ),
            timeout=60.0
        )

        logger.info("Loading summarizer...")
        models["summarizer"] = await asyncio.wait_for(
            asyncio.to_thread(
                pipeline,
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if DEVICE == "cuda" else -1,
                tokenizer_kwargs={"truncation": True}
            ),
            timeout=45.0
        )
        
        return models
    except asyncio.TimeoutError:
        logger.error("Model loading timed out")
        raise HTTPException(503, "Model loading timeout") from None
    except Exception as exc:
        logger.critical("Model loading failed: %s", exc)
        raise HTTPException(500, "Model initialization failed") from exc

@app.on_event("startup")
async def startup_event():
    try:
        app.state.models = await load_models()
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.critical("Startup failed: %s", e)
        sys.exit(1)

# ========= AUDIO PROCESSING =========
class AudioProcessor:
    @staticmethod
    def split_audio(path: str) -> list[str]:
        try:
            y, sr = librosa.load(path, sr=16000)
            duration = len(y) / sr
            if duration <= MAX_CHUNK_DURATION:
                return [path]

            chunk_size = MAX_CHUNK_DURATION * sr
            chunks = [y[i : i + chunk_size] for i in range(0, len(y), chunk_size)]

            temp_dir = "temp_chunks"
            os.makedirs(temp_dir, exist_ok=True)
            paths: list[str] = []

            for i, chunk in enumerate(chunks):
                chunk_path = os.path.join(
                    temp_dir, f"{os.path.basename(path)}_chunk{i}.wav"
                )
                sf.write(chunk_path, chunk, sr)
                paths.append(chunk_path)

            return paths
        except Exception as exc:
            logger.error("Audio splitting failed: %s", exc)
            return [path]

    @staticmethod
    def preprocess(path: str) -> str:
        try:
            y, sr = librosa.load(path, sr=16000, mono=True)
            y, _ = librosa.effects.trim(y, top_db=30)
            if np.max(np.abs(y)) > 0:
                y = y * (0.9 / np.max(np.abs(y)))
            out_path = f"{path}_processed.wav"
            sf.write(out_path, y, sr)
            return out_path
        except Exception as exc:
            logger.error("Audio preprocessing failed: %s", exc)
            raise HTTPException(400, "Invalid audio file") from exc

# ========= DATABASE LAYER =========
class Database:
    _pool: SimpleConnectionPool | None = None

    @classmethod
    def get_pool(cls) -> SimpleConnectionPool:
        if cls._pool is None:
            try:
                cls._pool = SimpleConnectionPool(
                    1,
                    5,
                    host="localhost",
                    port="5432",
                    dbname="postgres",
                    user="postgres",
                    password="tiger",
                )
            except PostgresError as exc:
                logger.error("DB connection failed: %s", exc)
                raise HTTPException(503, "Database unavailable") from exc
        return cls._pool

    @classmethod
    def save_results(cls, pid: int, en: str, summary: str) -> None:
        conn = None
        try:
            conn = cls.get_pool().getconn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO patient_summaries
                    (patient_id, english_translation, summary)
                    VALUES (%s, %s, %s)
                    """,
                    (pid, en, summary),
                )
                conn.commit()
        except PostgresError as exc:
            logger.error("DB operation failed: %s", exc)
            raise HTTPException(500, "Database error") from exc
        finally:
            if conn:
                cls.get_pool().putconn(conn)

# ========= CORE LOGIC =========
async def process_chunk(chunk_path: str) -> dict:
    processed_path = None
    try:
        processed_path = AudioProcessor.preprocess(chunk_path)

        loop = asyncio.get_event_loop()
        en_future = loop.run_in_executor(None, translate_english, processed_path)
        en_text = await en_future

        summary = MedicalSummarizer.summarize(en_text)

        return {
            "english": en_text,
            "summary": summary,
            "chunk_path": chunk_path,
            "processed_path": processed_path,
        }
    except Exception as exc:
        logger.error("Chunk processing failed: %s", exc)
        raise
    finally:
        if processed_path and os.path.exists(processed_path):
            try:
                os.remove(processed_path)
            except OSError:
                pass

def translate_english(path: str) -> str:
    segments, _ = app.state.models["english"].transcribe(
        path,
        task="translate",
        beam_size=5,
        temperature=[0.0, 0.2],  # Slight variation
        repetition_penalty=1.3,  # Add this
        no_repeat_ngram_size=3,  # Add this
        condition_on_previous_text=False
    )
    
    text = " ".join(s.text.strip() for s in segments if s.text.strip())
    # Improved repetition removal
    text = re.sub(r'\b(\w+\s+)(?=\1)', '', text, flags=re.I)
    return MedicalSummarizer._restore_punctuation(text)

# ========= API ENDPOINTS =========
@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...), patient_id: int = Form(...)) -> dict:
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a")):
        raise HTTPException(400, "Unsupported file format")

    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    upload_path = os.path.join(temp_dir, f"{patient_id}_{int(time.time())}_{file.filename}")

    try:
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        start = time.time()
        chunks = AudioProcessor.split_audio(upload_path)
        results = await asyncio.gather(*(process_chunk(c) for c in chunks))

        combined_en = " ".join(r["english"] for r in results)
        final_summary = MedicalSummarizer.summarize("\n".join(r["summary"] for r in results))

        Database.save_results(patient_id, combined_en, final_summary)  # Save with only English

        return {
            "patient_id": patient_id,
            "english_translation": combined_en,
            "summary": final_summary,
            "processing_time_sec": round(time.time() - start, 2),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unexpected error: %s", exc)
        raise HTTPException(500, "Processing failed") from exc
    finally:
        try:
            if os.path.exists(upload_path):
                os.remove(upload_path)
            for r in results if "results" in locals() else []:
                if "chunk_path" in r and r["chunk_path"] and os.path.exists(r["chunk_path"]):
                    os.remove(r["chunk_path"])
        except OSError as exc:
            logger.warning("Cleanup failed: %s", exc)

@app.get("/health")
async def health_check() -> dict:
    try:
        conn = Database.get_pool().getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        finally:
            Database.get_pool().putconn(conn)

        test_audio = np.zeros(16000)
        sf.write("test.wav", test_audio, 16000)
        app.state.models["english"].transcribe("test.wav", task="translate")
        os.remove("test.wav")

        return {"status": "healthy", "device": DEVICE}
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        raise HTTPException(503, "Service unavailable") from exc

# ========= ENTRY POINT =========
if __name__ == "__main__":
    for d in ("temp_uploads", "temp_chunks", "temp_processed"):
        os.makedirs(d, exist_ok=True)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,
        timeout_keep_alive=120,
        server_header=False,
        lifespan="on"
    )
