from flask import Flask, request, render_template
import os
import yt_dlp
import ffmpeg
import assemblyai as aai
import requests
import re
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

aai.settings.api_key = "6d4bce6db0d44a6ba2e1ba8a4a4c34c1"

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

WHITESPACE_HANDLER = lambda k: re.sub(r'\s+', ' ', re.sub(r'\n+', ' ', k.strip()))

def download_video(video_url):
    ydl_opts = {
        'format': 'best',
        'outtmpl': 'video.mp4',
        'nopostoverwrites': True,
    }

    if os.path.exists('video.mp4'):
        os.remove('video.mp4')

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

def scrape_youtube_captions(video_url):
    try:
        video_id = video_url.split('v=')[-1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        return " ".join([segment['text'] for segment in transcript])
    except Exception as e:
        print(f"Error fetching captions: {e}")
        return None

def convert_video_to_audio(video_file, audio_file):
    if os.path.exists(audio_file):
        os.remove(audio_file)

    ffmpeg.input(video_file).output(audio_file).overwrite_output().run()

def transcribe_audio(audio_file):
    transcript = aai.Transcriber().transcribe(audio_file)
    return transcript.text if transcript.status != aai.TranscriptStatus.error else transcript.error

def chunk_text(text, chunk_size=1024):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def summarize_text(text):
    text_chunks = chunk_text(WHITESPACE_HANDLER(text), 1024)

    summaries = []
    for chunk in text_chunks:
        input_ids = tokenizer(
            [chunk],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(device)["input_ids"]

        output_ids = model.generate(
            input_ids=input_ids,
            max_length=150,
            no_repeat_ngram_size=2,
            num_beams=4,
            length_penalty=1.0,
        )[0]

        summary = tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        summaries.append(summary)

    return " ".join(summaries)

@app.route('/', methods=['GET', 'POST'])
def index():
    transcription = None
    summary = None
    error = None

    if request.method == 'POST':
        if 'video_url' in request.form:
            video_url = request.form['video_url']
            transcription = scrape_youtube_captions(video_url)

            if not transcription:
                download_video(video_url)
                audio_file_path = 'audio.wav'
                convert_video_to_audio('video.mp4', audio_file_path)
                transcription = transcribe_audio(audio_file_path)

            if transcription:
                summary = summarize_text(transcription)
            else:
                error = "Transcription failed. Please try again."

        elif 'video_file' in request.files:
            video_file = request.files['video_file']
            video_file.save('uploaded_video.mp4')
            audio_file_path = 'audio.wav'
            convert_video_to_audio('uploaded_video.mp4', audio_file_path)
            transcription = transcribe_audio(audio_file_path)
            if transcription is not None:
                summary = summarize_text(transcription)
            else:
                error = "Transcription failed. Please try again."

    return render_template('index.html', transcription=transcription, summary=summary, error=error)

if __name__ == "__main__":
    app.run(debug=True)
