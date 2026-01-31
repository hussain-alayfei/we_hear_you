from flask import Flask, render_template, request, jsonify, send_file, abort, Response
from inference_classifier import SignLanguageClassifier
from surah_data import SURAHS, get_all_surahs, get_surah, is_surah_unlocked
import cv2
import numpy as np
import base64
import os
import glob
import requests

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Ensure Arabic characters are not escaped in JSON

# Initialize classifier
try:
    classifier = SignLanguageClassifier()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    classifier = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_surahs')
def get_surahs_route():
    """
    Get all surahs with their metadata
    Returns a JSON object with all surahs
    """
    try:
        surahs = get_all_surahs()
        return jsonify(surahs)
    except Exception as e:
        print(f"Error fetching surahs: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_surah/<surah_id>')
def get_surah_route(surah_id):
    """
    Get a specific surah by ID
    Returns 403 if the surah is locked
    Returns 404 if the surah doesn't exist
    """
    try:
        surah = get_surah(surah_id)
        
        if not surah:
            return jsonify({'error': 'Surah not found'}), 404
        
        if not is_surah_unlocked(surah_id):
            return jsonify({'error': 'Surah is locked'}), 403
        
        return jsonify(surah)
    except Exception as e:
        print(f"Error fetching surah {surah_id}: {e}")
        return jsonify({'error': str(e)}), 500

def normalize_char_for_image(char):
    """
    Normalize Arabic character for image lookup
    Converts all forms of a letter to its base form
    """
    if not char:
        return char
    
    # تطبيع الألف: أ، إ، آ، ء → ا
    char = char.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا').replace('ء', 'ا').replace('ٱ', 'ا')
    
    # تطبيع الياء: ى، ئ → ي
    char = char.replace('ى', 'ي').replace('ئ', 'ي')
    
    # تطبيع التاء المربوطة: ة → ه
    char = char.replace('ة', 'ه')
    
    # إزالة التشكيل
    import re
    char = re.sub(r'[\u064B-\u065F\u0670]', '', char)
    
    return char

@app.route('/sign_image/<path:char>')
def get_sign_image(char):
    """
    Dynamic Image Selector:
    Finds all images starting with `char` in static/signs/
    Sorts them and returns the LAST one.
    Useful for versioning: 'Aleff.jpg', 'Aleff_v2.jpg' -> serves 'Aleff_v2.jpg'
    """
    try:
        # Base directory
        static_folder = os.path.join(app.static_folder, 'signs')
        
        # Decode URL-encoded character if needed
        from urllib.parse import unquote
        char = unquote(char)
        
        # Normalize the character (أ، إ، آ → ا)
        normalized_char = normalize_char_for_image(char)
        
        # PRIORITY 1: Try exact match with normalized character
        exact_path = os.path.join(static_folder, f"{normalized_char}.jpg")
        if os.path.exists(exact_path):
            return send_file(exact_path, mimetype='image/jpeg')
        
        # PRIORITY 2: Search pattern for versioned files (char + suffix + extension)
        pattern = os.path.join(static_folder, f"{normalized_char}*.jpg")
        matches = glob.glob(pattern)
        
        if not matches:
            return abort(404)
            
        # Sort matches and select last one (for versioning support)
        matches.sort()
        last_image = matches[-1]
        
        return send_file(last_image, mimetype='image/jpeg')
        
    except Exception as e:
        return abort(404)

@app.route('/predict', methods=['POST'])
def predict():
    if not classifier:
        return jsonify({'error': 'Model not loaded', 'prediction': None, 'landmarks': []}), 200
    
    try:
        # Check if request has JSON
        if not request.is_json:
            return jsonify({'error': 'No JSON data', 'prediction': None, 'landmarks': []}), 200
            
        json_data = request.get_json(silent=True)
        if not json_data or 'image' not in json_data:
            return jsonify({'error': 'No image data', 'prediction': None, 'landmarks': []}), 200
            
        data = json_data['image']
        
        # Decode base64
        # Data URL format: "data:image/jpeg;base64,/9j/4AAQ..."
        if "," in data:
            header, encoded = data.split(",", 1)
        else:
            encoded = data
        
        try:
            binary = base64.b64decode(encoded)
        except Exception:
            return jsonify({'error': 'Invalid base64', 'prediction': None, 'landmarks': []}), 200
            
        image_array = np.frombuffer(binary, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Failed to decode image', 'prediction': None, 'landmarks': []}), 200

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        label, detection_result = classifier.predict(frame_rgb)
        
        # Serialize landmarks
        landmarks_data = []
        if detection_result and detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                hand_points = []
                for landmark in hand_landmarks:
                    hand_points.append({'x': landmark.x, 'y': landmark.y})
                landmarks_data.append(hand_points)

        # Ensure label is properly encoded (None becomes null in JSON)
        response_data = {
            'prediction': label if label is not None else None,
            'landmarks': landmarks_data
        }
        
        return jsonify(response_data)
    except Exception as e:
        # Return valid JSON even on error
        return jsonify({'error': str(e), 'prediction': None, 'landmarks': []}), 200

# ============================================
# TEXT-TO-SPEECH (ElevenLabs API)
# ============================================
ELEVENLABS_API_KEY = "sk_a29dcc8f5bc1c17879dcd07c21dfbb559a382040f2947f5f"
ELEVENLABS_VOICE_ID = "Jkgj7lZ9O8Am0h71d9fq"

@app.route('/tts', methods=['POST'])
def text_to_speech():
    """
    Convert Arabic text to speech using ElevenLabs Turbo v2.5 model
    Expects JSON body: { "text": "مرحبا" }
    Returns: audio/mpeg stream
    """
    try:
        data = request.get_json()
        raw_text = data.get('text', '') if data else ''
        
        if not raw_text:
            return jsonify({'error': 'No text provided'}), 400

        # Use the raw text directly (no padding that might be spoken)
        text = raw_text
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg"
        }
        payload = {
            "text": text,
            "model_id": "eleven_turbo_v2_5",
            "voice_settings": {
                "stability": 0.6,
                "similarity_boost": 0.8,
                "speed": 0.85  # Slightly slower for clearer pronunciation
            }
        }
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            return Response(response.content, mimetype='audio/mpeg')
        else:
            print(f"ElevenLabs API error: {response.status_code} - {response.text}")
            return jsonify({'error': 'TTS failed', 'status': response.status_code}), 500
            
    except Exception as e:
        print(f"TTS Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
