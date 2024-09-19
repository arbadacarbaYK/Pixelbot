from flask import Flask, request, render_template, send_from_directory, jsonify
import os
import uuid
from pixelateTG import detect_heads, overlay, process_image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        return jsonify({'filename': filename}), 200
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/process', methods=['POST'])
def process_file():
    data = request.json
    filename = data.get('filename')
    overlay_type = data.get('overlay_type')
    if not filename or not overlay_type:
        return jsonify({'error': 'Invalid request'}), 400
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    user_id = str(uuid.uuid4())
    processed_path = overlay(file_path, user_id, overlay_type, 1.5, None)
    return jsonify({'processed_filename': os.path.basename(processed_path)}), 200

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
