from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
from google.cloud import texttospeech, speech_v1 as speech, language_v1 as language
import os
import io
import base64
from werkzeug.serving import BaseWSGIServer
import uuid
import sys
import threading

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Google Cloud credentials setup
if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
    credentials_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
else:
    credentials_path = "C:\\Users\\vangm\\speech app\\speech-app-436121-f907e34ec229.json"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# Google TTS, STT, and Language Clients Setup
tts_client = texttospeech.TextToSpeechClient()
stt_client = speech.SpeechClient()
language_client = language.LanguageServiceClient()

def handle_error(args):
    if isinstance(args.exc_value, OSError) and getattr(args.exc_value, 'winerror', None) == 10038:
        print("Server closed.")
    else:
        sys.__excepthook__(args.exc_type, args.exc_value, args.exc_traceback)

threading.excepthook = handle_error

def patch_werkzeug():
    def _cleanup_socket(self):
        if self.socket:
            try:
                self.socket.close()
            except OSError:
                pass
        self.socket = None

    BaseWSGIServer._cleanup_socket = _cleanup_socket

patch_werkzeug()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    try:
        data = request.json
        text_input = data['text']
        
        sentiment = analyze_sentiment(text_input)
        
        synthesis_input = texttospeech.SynthesisInput(text=text_input)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", 
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3)

        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config)

        audio_content = io.BytesIO(response.audio_content)
        audio_content.seek(0)
        
        unique_id = str(uuid.uuid4())
        text_filename = os.path.join(app.config['UPLOAD_FOLDER'], f'text_to_speech_{unique_id}.txt')
        audio_filename = os.path.join(app.config['UPLOAD_FOLDER'], f'text_to_speech_{unique_id}.mp3')
        
        with open(text_filename, 'w', encoding='utf-8') as f:
            f.write(f'Text: {text_input}\nSentiment: {sentiment}\n')
        
        with open(audio_filename, 'wb') as f:
            f.write(audio_content.getvalue())
        
        return jsonify({
            'audio': 'data:audio/mp3;base64,' + base64.b64encode(audio_content.getvalue()).decode('utf-8'),
            'sentiment': sentiment,
            'text_file': f'/uploads/{os.path.basename(text_filename)}',
            'audio_file': f'/uploads/{os.path.basename(audio_filename)}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        audio_content = audio_file.read()
        
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            sample_rate_hertz=48000,
            language_code="en-US"
        )

        response = stt_client.recognize(config=config, audio=audio)

        transcription = response.results[0].alternatives[0].transcript if response.results else "No transcription available"

        sentiment = analyze_sentiment(transcription)

        unique_id = str(uuid.uuid4())
        filename = os.path.join(app.config['UPLOAD_FOLDER'], f'speech_to_text_{unique_id}.txt')
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f'Transcription: {transcription}\nSentiment: {sentiment}\n')

        return jsonify({
            'transcript': transcription, 
            'sentiment': sentiment,
            'text_file': f'/uploads/{os.path.basename(filename)}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def analyze_sentiment(text):
    try:
        document = language.Document(content=text, type_=language.Document.Type.PLAIN_TEXT)
        response = language_client.analyze_sentiment(request={'document': document})
        
        sentiment_score = response.document_sentiment.score

        if sentiment_score > 0.25:
            return 'Positive'
        elif sentiment_score < -0.25:
            return 'Negative'
        else:
            return 'Neutral'
    
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 8080))
        app.run(host='0.0.0.0', port=port, debug=False)
    except KeyboardInterrupt:
        print("Server shutting down...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        sys.exit(0)