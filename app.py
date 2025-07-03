from flask import Flask, render_template, request, jsonify
import os
import mimetypes
from werkzeug.utils import secure_filename
from detector import CopyMoveForgeryDetector

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model once
model_path = os.path.join("artifacts", "cnn_model.keras")
model = None

try:
    print(f"🔍 Loading Keras model from: {os.path.abspath(model_path)}")
    model = CopyMoveForgeryDetector(model_path=model_path)
    print(f"✅ Model loaded successfully: {type(model)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp'}
    if file.filename.rsplit('.', 1)[-1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file format'}), 400

    mimetype = mimetypes.guess_type(file.filename)[0]
    if not mimetype or not mimetype.startswith('image/'):
        return jsonify({'error': 'File is not an image'}), 400

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        results = model.detect_forgery(file_path)
        os.remove(file_path)

        return jsonify({
            'prediction': "Forged" if bool(results['is_forged']) else "Authentic",
            'is_forged': bool(results['is_forged']),
            'confidence_percentage': round(float(results['confidence_score']), 2),
            'confidence': float(results['confidence_score'])
        })


    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy' if model else 'unhealthy',
        'model_type': str(type(model)) if model else 'None'
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
