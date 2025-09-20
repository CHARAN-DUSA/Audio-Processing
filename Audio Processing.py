import whisper
import os
import soundfile as sf
import sounddevice as sd
import numpy as np
import datetime
import queue
import threading
import tkinter as tk
import random
from tkinter.scrolledtext import ScrolledText
from nltk.tokenize import sent_tokenize
from pymongo import MongoClient
from fpdf import FPDF



# ------------------------ CONFIG ------------------------
MONGO_URL = "mongodb://localhost:27017/"
DATABASES = ["admin","config","local","myFirstDB","studentDB"]
DB_NAME = "myFirstDB"  # Choosing the Database
TRANSCRIPT_COLLECTION = "transcripts"
SAMPLERATE = 16000
CHUNK_DURATION = 10  # seconds per chunk
audio_queue = queue.Queue()
recording = True  # Control flag for stopping recording



# ------------------------ MONGODB ------------------------
client = MongoClient(MONGO_URL)
db = client[DB_NAME]
transcripts = db[TRANSCRIPT_COLLECTION]



# ------------------------ MODELS ------------------------
model = whisper.load_model("small")



# HuggingFace token for Pyannote
HF_TOKEN = os.environ.get("HF_TOKEN")
# Try loading Pyannote speaker-diarization
try:
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)
except Exception as e:
    print("Warning: Pyannote pipeline not loaded. Speaker diarization will be skipped.")
    pipeline = None



# ------------------------ GUI ------------------------
root = tk.Tk()
root.title(f"Audio Processing - DB: {DB_NAME}")
root.geometry("900x600")

text_area = ScrolledText(root, wrap=tk.WORD, font=("Helvetica", 12))
text_area.pack(expand=True, fill=tk.BOTH)
text_area.insert(tk.END, "Starting real-time transcription...\n")



# Stop button
def stop_recording():
    global recording
    recording = False
    text_area.insert(tk.END, "\n🔴 Recording stopped. Exporting PDF...\n")
stop_button = tk.Button(root, text="Stop Recording", command=stop_recording, bg="red", fg="white")
stop_button.pack(pady=5)




# ------------------------ SPEAKER COLORS ------------------------
speaker_colors = {}
def get_color_for_speaker(speaker):
    if speaker not in speaker_colors:
        speaker_colors[speaker] = f"#{random.randint(0,0xFFFFFF):06x}"
    return speaker_colors[speaker]



# ------------------------ AUDIO CALLBACK ------------------------
def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

def start_recording():
    with sd.InputStream(channels=1, samplerate=SAMPLERATE, callback=audio_callback):
        while recording:
            sd.sleep(1000)  # keeps thread alive



# ------------------------ PDF EXPORT ------------------------
def export_pdf(transcript_text, conv_id, action_items, topics):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Transcript: {conv_id}", ln=True, align="C")
    pdf.ln(10)

    # Highlight Action Items
    if action_items:
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(0, 10, "Action Items:", ln=True)
        pdf.set_font("Arial", size=12)
        for item in action_items:
            pdf.multi_cell(0, 8, f"• {item}")
        pdf.ln(5)

    # Highlight Topics
    if topics:
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(0, 10, "Topics:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 8, ", ".join(topics))
        pdf.ln(10)

    # Add full transcript
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, "Full Transcript:", ln=True)
    pdf.set_font("Arial", size=12)
    for line in transcript_text.split("\n"):
        pdf.multi_cell(0, 8, line)

    filename = f"transcript_{conv_id}.pdf"
    pdf.output(filename)
    print(f"✅ PDF saved: {filename}")



# ------------------------ NLP FUNCTIONS ------------------------
def extract_action_items(text):
    sentences = sent_tokenize(text)
    action_items = [s for s in sentences if any(w in s.lower() for w in ['todo','action','follow up','must','should'])]
    return action_items

def extract_topics(text, top_n=5):
    words = [w.lower() for w in text.split()]
    freq = {}
    for w in words:
        if len(w) > 3:
            freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    topics = [w for w, c in sorted_words[:top_n]]
    return topics



# ------------------------ PROCESS AUDIO CHUNKS ------------------------
def process_audio_chunks():
    conv_id = f"meeting_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S')}"
    full_transcript_text = ""
    chunk_counter = 0

    while recording or not audio_queue.empty():
        frames = []
        collected = 0
        frames_needed = CHUNK_DURATION * SAMPLERATE
        while collected < frames_needed and recording:
            if audio_queue.empty():
                sd.sleep(200)
                continue
            data = audio_queue.get()
            frames.append(data)
            collected += len(data)

        if frames:
            audio_chunk = np.concatenate(frames, axis=0).flatten()
            chunk_filename = f"chunk_{chunk_counter}.wav"
            sf.write(chunk_filename, audio_chunk, SAMPLERATE)

            # WHISPER TRANSCRIPTION
            result = model.transcribe(chunk_filename)
            segments = result.get("segments", [])
            full_text = result.get("text", "").strip()
            full_transcript_text += full_text + "\n"

            # DIARIZATION
            diarization_result = None
            if pipeline:
                try:
                    diarization_result = pipeline(chunk_filename)
                except Exception as e:
                    print(f"Warning: Pyannote failed on chunk {chunk_counter}, skipping diarization.")

            for seg in segments:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                text = seg.get("text", "")
                speaker_label = "Unknown"

                if diarization_result:
                    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                        if start >= turn.start and end <= turn.end:
                            speaker_label = speaker
                            break

                # STORE IN MONGODB
                doc = {
                    "conversation_id": conv_id,
                    "speaker": speaker_label,
                    "start_time": start,
                    "end_time": end,
                    "text": text,
                    "created_at": datetime.datetime.now(datetime.timezone.utc)
                }
                transcripts.insert_one(doc)

                # GUI UPDATE
                color = get_color_for_speaker(speaker_label)
                text_area.insert(tk.END, f"[{speaker_label}] {text}\n", speaker_label)
                text_area.tag_config(speaker_label, foreground=color)
                text_area.see(tk.END)

            os.remove(chunk_filename)
            chunk_counter += 1

    # Extract action items and topics
    action_items = extract_action_items(full_transcript_text)
    topics = extract_topics(full_transcript_text)

    # Export PDF
    export_pdf(full_transcript_text, conv_id, action_items, topics)
    text_area.insert(tk.END, f"✅ PDF exported successfully.\n")



# ------------------------ THREADS ------------------------
t1 = threading.Thread(target=start_recording, daemon=True)
t2 = threading.Thread(target=process_audio_chunks, daemon=True)
t1.start()
t2.start()

# ------------------------ START GUI ------------------------
root.mainloop()
