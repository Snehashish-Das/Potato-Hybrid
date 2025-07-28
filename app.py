from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model
from ai_agent import PotatoLeafAIAgent
from preprocess import grabcut_preprocess  # ✅ Import GrabCut function

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['PROCESSED_FOLDER'] = 'static/processed/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load model and preprocessing components
model = load_model('potato_leaf_classifier.h5')
scaler = joblib.load('scaler.pkl')
with open('class_map.json', 'r') as f:
    class_map = json.load(f)

class_names = list(class_map.keys())
agent = PotatoLeafAIAgent(scaler=scaler, class_map=class_map)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        # ✅ Apply GrabCut and save processed image
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        output_img = grabcut_preprocess(upload_path, save_path=processed_path)

        if output_img is None:
            return render_template('index.html', error="Image preprocessing failed.")

        # ✅ Use processed image for prediction
        Ximg, Xfeat = agent.prepare_single_sample(processed_path)
        prediction = model.predict([Ximg, Xfeat])
        predicted_class = class_names[np.argmax(prediction)]
        confidence = round(100 * np.max(prediction), 2)

        return render_template(
            'index.html',
            filename=filename,
            predicted_class=predicted_class,
            confidence=confidence
        )

    return redirect(request.url)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    app.run(debug=True)
