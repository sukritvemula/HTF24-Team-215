from flask import Flask, request, render_template
from youtube_transcript_api import YouTubeTranscriptApi
import nltk
from nltk.corpus import stopwords
import re
import logging
from flask_cors import CORS
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Initialize ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)  # You can adjust the number of workers

def scrape_youtube_captions(video_url):
    try:
        video_id = video_url.split('v=')[-1].split('&')[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        return " ".join([segment['text'] for segment in transcript])
    except Exception as e:
        app.logger.error(f"Error fetching captions: {e}")
        return None

def clean_transcription(transcription):
    """Clean the transcription text."""
    transcription = re.sub(r'\[.*?\]', '', transcription)
    transcription = re.sub(r'\s+', ' ', transcription)
    return transcription.strip()

def chunk_text(text, max_chunk_size=1024):
    """Split text into chunks that fit within model's max token limit."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        if current_size + len(word) + 1 > max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
            current_size += len(word) + 1
            
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def summarize_text(text):
    """Summarize text using Hugging Face transformers."""
    try:
        # Clean the text first
        cleaned_text = clean_transcription(text)
        
        # Split into chunks if text is too long
        chunks = chunk_text(cleaned_text)
        summaries = []
        
        for chunk in chunks:
            # Generate summary for each chunk
            summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        
        # Combine all summaries
        final_summary = " ".join(summaries)
        return final_summary
    except Exception as e:
        app.logger.error(f"Error in summarization: {e}")
        return "Error generating summary. Please try again."

def fetch_transcription(video_url):
    """Fetch the transcription in a separate thread."""
    return scrape_youtube_captions(video_url)

@app.route('/', methods=['GET', 'POST'])
def index():
    transcription = None
    summary = None
    error = None

    if request.method == 'POST':
        video_url = request.form.get('video_url')
        if video_url:
            app.logger.debug("Video URL received: %s", video_url)
            # Use multi-threading to fetch transcription
            future = executor.submit(fetch_transcription, video_url)
            transcription = future.result()  # Wait for the result

            if not transcription:
                error = "No transcript available. The video may not have captions."
            else:
                summary = summarize_text(transcription)

    return render_template('index.html', transcription=transcription, summary=summary, error=error)

if __name__ == "__main__":
    app.run(debug=True)
