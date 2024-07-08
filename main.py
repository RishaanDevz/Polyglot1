import os
import uuid
import wave
import logging
from flask import Flask, render_template, request, send_file, jsonify
from elevenlabs import Voice, VoiceSettings, play
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import shutil
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Get the API keys from environment variables
gemini_api_key = os.getenv("GEMINI_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")

# Configure Google AI
genai.configure(api_key=gemini_api_key)

# Create instances of the clients
elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key)
deepgram = DeepgramClient(deepgram_api_key)

# Create the Gemini model
generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                              generation_config=generation_config,
                              safety_settings={
                                  HarmCategory.HARM_CATEGORY_HATE_SPEECH:
                                  HarmBlockThreshold.BLOCK_NONE,
                                  HarmCategory.HARM_CATEGORY_HARASSMENT:
                                  HarmBlockThreshold.BLOCK_NONE,
                              })

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# List of supported languages
SUPPORTED_LANGUAGES = [
    "Detect Language", "English (USA)", "English (UK)", "English (Australia)",
    "English (Canada)", "Japanese", "Chinese", "German", "Hindi",
    "French (France)", "French (Canada)", "Korean", "Portuguese (Brazil)",
    "Portuguese (Portugal)", "Italian", "Spanish (Spain)", "Spanish (Mexico)",
    "Indonesian", "Dutch", "Turkish", "Filipino", "Polish", "Swedish",
    "Bulgarian", "Romanian", "Arabic (Saudi Arabia)", "Arabic (UAE)", "Czech",
    "Greek", "Finnish", "Croatian", "Malay", "Slovak", "Danish", "Tamil",
    "Ukrainian"
]


def text_to_speech(text):
    try:
        audio_id = str(uuid.uuid4())
        output_filename = f"static/audio/{audio_id}.wav"

        audio_stream = elevenlabs_client.generate(
            text=text,
            model="eleven_multilingual_v2",
            voice="Alice",
            stream=True)

        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with wave.open(output_filename, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(44100)
            for chunk in audio_stream:
                wav_file.writeframes(chunk)

        logging.info(f"Audio saved to {output_filename}")
        return audio_id
    except Exception as e:
        logging.error(f"Error generating speech: {e}")
        raise


def translate_text(language1, language2, text):
    try:
        prompt = f"""You are Polyglot, an advanced AI translator capable of translating between {language1} and {language2} and {language2} and {language1}. Your task is to provide direct translations without any additional commentary or explanations. 

Examples:
Input: Hello how are you
Output: Bonjour comment ça va?

Input: Ça va bien, et toi?
Output: I'm doing well, and you?

Translate the following text. Provide only the translation, without any additional text:

{text}"""

        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)

        return response.text.strip()
    except Exception as e:
        logging.error(f"Error translating text: {e}")
        raise


def transcribe_audio(audio_data):
    try:
        payload = {"buffer": audio_data}
        options = PrerecordedOptions(
            model="whisper-medium",
            smart_format=True,
            detect_language=True,
        )

        response = deepgram.listen.prerecorded.v("1").transcribe_file(
            payload, options)
        return response['results']['channels'][0]['alternatives'][0][
            'transcript']
    except Exception as e:
        logging.error(f"Error transcribing audio: {e}")
        raise


@app.route('/transcribe', methods=['POST'])
def transcribe_audio_route():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    try:
        audio_data = audio_file.read()
        transcription = transcribe_audio(audio_data)
        return jsonify({'transcription': transcription})
    except Exception as e:
        logging.error(f"Error in /transcribe route: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        language1 = request.form['language1']
        language2 = request.form['language2']
        text = request.form['text']
        try:
            app.logger.info(f"Translating from {language1} to {language2}: {text}")
            translation = translate_text(language1, language2, text)
            app.logger.info(f"Translation result: {translation}")
            audio_id = text_to_speech(translation)
            app.logger.info(f"Generated audio ID: {audio_id}")
            return jsonify({'translation': translation, 'audio_id': audio_id})
        except Exception as e:
            app.logger.error(f"Error processing POST request: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    return render_template('index.html', languages=SUPPORTED_LANGUAGES)


@app.route('/audio/<audio_id>')
def serve_audio(audio_id):
    audio_path = f'static/audio/{audio_id}.wav'
    if os.path.exists(audio_path):
        return send_file(audio_path, mimetype='audio/wav')
    else:
        return jsonify({'error': 'Audio file not found'}), 404


def cleanup_old_audio_files():
    audio_dir = 'static/audio'
    current_time = datetime.now()
    for filename in os.listdir(audio_dir):
        file_path = os.path.join(audio_dir, filename)
        file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
        if current_time - file_modified > timedelta(
                minutes=3):  # Delete files older than 3 minutes 
            os.remove(file_path)





@app.before_request
def before_request():
    cleanup_old_audio_files()
