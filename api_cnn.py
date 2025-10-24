# Nome do arquivo: api_cnn.py
# API Flask dedicada para servir o modelo CNN treinado.

import os
import io
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.applications.mobilenet_v2 import preprocess_input

# --- 1. CONFIGURAÇÕES E CARREGAMENTO ---
MODEL_PATH = 'modelo_cnn.keras'
CLASSES_PATH = 'classes_cnn.npy'

# Parâmetros que DEVEM ser os mesmos do treinamento
SAMPLE_RATE = 22050
DURATION = 7
IMG_SIZE = (224, 224)

# Carrega o modelo e as classes UMA VEZ quando o servidor inicia
print("Carregando o modelo CNN treinado... Isso pode levar um momento.")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    CLASS_NAMES = np.load(CLASSES_PATH, allow_pickle=True)
    print("Modelo e classes carregados com sucesso!")
    print(f"Classes que o modelo conhece: {list(CLASS_NAMES)}")
except Exception as e:
    print(f"ERRO CRÍTICO AO CARREGAR O MODELO: {e}")
    model = None

# --- 2. FUNÇÃO DE PREVISÃO ---
def prever_som_cnn(audio_path):
    if model is None: return {"error": "Modelo CNN não está carregado."}
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        if len(y) < DURATION * sr: y = np.pad(y, (0, DURATION * sr - len(y)))

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=IMG_SIZE[0])
        log_S = librosa.power_to_db(S, ref=np.max)
        
        buf = io.BytesIO()
        fig, ax = plt.subplots(figsize=(6, 6))
        librosa.display.specshow(log_S, sr=sr, ax=ax, x_axis='time', y_axis='mel')
        ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
        plt.tight_layout(pad=0)
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig); buf.seek(0)

        img = Image.open(buf).convert('RGB').resize(IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        processed_img_array = preprocess_input(img_array)

        predictions = model.predict(processed_img_array, verbose=0)
        score = tf.nn.softmax(predictions[0])

        predicted_class = CLASS_NAMES[np.argmax(score)]
        confidence = 100 * np.max(score)

        return {"previsao": predicted_class, "confianca": f"{confidence:.2f}"}
    except Exception as e:
        return {"error": f"Erro durante o processamento do áudio: {e}"}

# --- 3. CONFIGURAÇÃO DA API FLASK ---
app = Flask(__name__)
CORS(app)
if not os.path.exists('temp_media'): os.makedirs('temp_media')

@app.route('/prever', methods=['POST'])
def handle_previsao():
    if 'media' not in request.files: return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    media_file = request.files['media']
    caminho_temporario = os.path.join('temp_media', media_file.filename)
    media_file.save(caminho_temporario)
    resultado = prever_som_cnn(caminho_temporario)
    os.remove(caminho_temporario)
    return jsonify(resultado)

# --- 4. EXECUÇÃO DO SERVIDOR ---
if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    print("Servidor da API CNN dedicado iniciado em http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000)