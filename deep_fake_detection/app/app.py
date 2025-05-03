from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import torch
from datetime import datetime
from models import (
    ResNeXtLSTM, EfficientNetLSTM, MobileNetGRU,
    EfficientNetGRU, ResNeXtGRU, MobileNetLSTM
)
from video_processing import VideoProcessor
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}
app.config['LOGS_FILE'] = 'detection_logs.json'

# Model configuration
app.config['MODELS_CONFIG'] = {
    'resnext_lstm': {
        'name': 'ResNeXt-LSTM',
        'path': 'models/resnext_lstm.pt',
        'type': 'CNN-RNN Hybrid',
        'architecture': 'ResNeXt50 + LSTM',
        'strengths': 'Excellent at detecting temporal inconsistencies',
        'bestFor': 'Professional-grade deepfakes',
        'accuracy': '~92%'
    },
    'efficientnet_lstm': {
        'name': 'EfficientNet-LSTM',
        'path': 'models/efficientnet_lstm.pt',
        'type': 'CNN-RNN Hybrid',
        'architecture': 'EfficientNet-B4 + LSTM',
        'strengths': 'Good balance between accuracy and computational efficiency',
        'bestFor': 'General purpose detection',
        'accuracy': '~89%'
    },
    'mobilenet_gru': {
        'name': 'MobileNet-GRU',
        'path': 'models/mobilenet_gru.pt',
        'type': 'CNN-RNN Hybrid',
        'architecture': 'MobileNetV3 + GRU',
        'strengths': 'Fastest model, good for mobile devices',
        'bestFor': 'Real-time detection',
        'accuracy': '~85%'
    },
    'efficientnet_gru': {
        'name': 'EfficientNet-GRU',
        'path': 'models/efficientnet_gru.pt',
        'type': 'CNN-RNN Hybrid',
        'architecture': 'EfficientNet-B3 + GRU',
        'strengths': 'Balanced performance',
        'bestFor': 'General purpose detection',
        'accuracy': '~88%'
    },
    'resnext_gru': {
        'name': 'ResNeXt-GRU',
        'path': 'models/resnext_gru.pt',
        'type': 'CNN-RNN Hybrid',
        'architecture': 'ResNeXt50 + GRU',
        'strengths': 'Good temporal analysis',
        'bestFor': 'High-quality videos',
        'accuracy': '~91%'
    },
    'mobilenet_lstm': {
        'name': 'MobileNet-LSTM',
        'path': 'models/mobilenet_lstm.pt',
        'type': 'CNN-RNN Hybrid',
        'architecture': 'MobileNetV3 + LSTM',
        'strengths': 'Fast with decent sequence analysis',
        'bestFor': 'Real-time with better accuracy',
        'accuracy': '~86%'
    }
}

# Initialize processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
video_processor = VideoProcessor()

# Model class mapping
MODEL_CLASSES = {
    'resnext_lstm': ResNeXtLSTM,
    'efficientnet_lstm': EfficientNetLSTM,
    'mobilenet_gru': MobileNetGRU,
    'efficientnet_gru': EfficientNetGRU,
    'resnext_gru': ResNeXtGRU,
    'mobilenet_lstm': MobileNetLSTM
}

# Load all models
models = {}
for name, config in app.config['MODELS_CONFIG'].items():
    try:
        model = MODEL_CLASSES[name](num_classes=2).to(device)
        model.load_state_dict(torch.load(config['path'], map_location=device))
        model.eval()
        models[name] = {
            'instance': model,
            'name': config['name']
        }
        print(f"Successfully loaded model: {config['name']}")
    except Exception as e:
        print(f"Error loading model {config['name']}: {str(e)}")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_logs():
    if os.path.exists(app.config['LOGS_FILE']):
        with open(app.config['LOGS_FILE'], 'r') as f:
            return json.load(f)
    return []

def save_log(log_entry):
    logs = load_logs()
    logs.insert(0, log_entry)
    with open(app.config['LOGS_FILE'], 'w') as f:
        json.dump(logs[:100], f)  # Keep last 100 entries

@app.route('/')
def index():
    return render_template('index.html', models_config=app.config['MODELS_CONFIG'])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    model_name = request.form.get('model', 'resnext_lstm')
    if model_name not in models:
        return jsonify({'error': 'Invalid model selected'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            faces = video_processor.extract_faces(filepath)
            if faces is None:
                return jsonify({'error': 'No faces detected in video'}), 400
                
            faces_tensor = torch.from_numpy(faces).unsqueeze(0).to(device)
            model = models[model_name]['instance']
            
            with torch.no_grad():
                _, outputs = model(faces_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                fake_prob = probabilities[0][1].item() * 100
                real_prob = probabilities[0][0].item() * 100
                
            # Get model prediction and confidence
            prediction = 'FAKE' if fake_prob > 50 else 'REAL'
            confidence = {
                'real': round(real_prob, 2),
                'fake': round(fake_prob, 2)
            }
            
            os.remove(filepath)
            
            result = {
                'model': models[model_name]['name'],
                'filename': filename,
                'prediction': prediction,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            
            save_log(result)
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/logs')
def get_logs():
    logs = load_logs()
    # Format confidence scores for display in logs
    formatted_logs = []
    for log in logs:
        formatted_log = log.copy()
        formatted_log['confidence_display'] = {
            'real': f"{log['confidence']['real']}%",
            'fake': f"{log['confidence']['fake']}%"
        }
        formatted_logs.append(formatted_log)
    return jsonify(formatted_logs)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)