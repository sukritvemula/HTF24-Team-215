# YouTube Transcript Summarizer HTF24-Team-215

## Description
YouTube Transcript Summarizer is a Flask web application that allows users to input a YouTube video URL and receive a summarized version of its transcript. The application utilizes the YouTube Transcript API to fetch video captions and employs the Hugging Face Transformers library for summarization. This tool is especially useful for quickly understanding video content without watching the entire video.

## Features
- Fetches captions from YouTube videos.
- Cleans and processes transcription text.
- Summarizes lengthy transcripts using advanced natural language processing techniques.
- User-friendly web interface.

## Requirements
- Python 3.7 or higher

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sukritvemula/HTF24-Team-215.git
   cd HTF24-Team-215
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   Create a `requirements.txt` file in the root directory with the following content:
   ```plaintext
   Flask==2.3.2
   Flask-CORS==3.0.10
   youtube-transcript-api==0.6.1
   nltk==3.7
   transformers==4.32.1
   torch==2.0.1
   ```
   Then run:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data:**
   The application requires certain NLTK data packages to function correctly. You can download them by running the following commands in a Python shell:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## Usage

1. **Run the application:**
   ```bash
   python app.py
   ```

2. **Access the web interface:**
   Open your web browser and go to `http://127.0.0.1:5000`.

3. **Input the YouTube video URL:**
   - Paste the URL of the YouTube video you want to summarize in the input field.
   - Click on the submit button.

4. **View the transcription and summary:**
   The application will display the video transcription along with a summarized version.

## Contributing
Contributions are welcome! If you would like to contribute to the project, please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [NLTK](https://www.nltk.org/)

### Instructions for Use:
1. Copy the entire content above.
2. Create a new file in the root directory of your project named `README.md`.
3. Paste the copied content into the `README.md` file.
4. Replace `yourusername` in the clone command with your actual GitHub username or the appropriate URL for your repository.
