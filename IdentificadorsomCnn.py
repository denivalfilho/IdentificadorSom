#IdentificadorsomCnn.py

import os
import sys
import glob
import random
import numpy as np
import librosa
import librosa.display ### NOVO ###
import matplotlib.pyplot as plt ### NOVO ###
import tensorflow as tf
from pydub import AudioSegment
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical

# --- 1. CONFIGURAÇÕES ROBUSTAS ---
try:
    # Pega o caminho absoluto da pasta onde o script está localizado
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Se __file__ não estiver definido (ex: em um notebook interativo)
    script_dir = os.getcwd()

# Cria os caminhos completos baseados na localização do script
DATA_PATH = os.path.join(script_dir, 'DadosDeAudio/')
MODEL_PATH = os.path.join(script_dir, 'modelo_cnn.keras')
CLASSES_PATH = os.path.join(script_dir, 'classes_cnn.npy')

### NOVO: Caminho para salvar as imagens dos espectrogramas ###
SPECTROGRAM_PATH = os.path.join(script_dir, 'EspectrogramasSalvos/')

# Parâmetros de Áudio e Modelo
SAMPLE_RATE = 22050
DURATION = 7
IMG_SIZE = (224, 224) # Altura x Largura da imagem do espectrograma

# Parâmetros de Treinamento
BATCH_SIZE = 32
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 35
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

# --- 2. FUNÇÕES AUXILIARES ---

def converter_audios_para_wav(data_path):
    """
    Converte mídias (áudio e vídeo) para .wav 16kHz mono para consistência.
    """
    print("\n--- Iniciando verificação e conversão de mídias para .wav ---")
    formatos = ["*.mp3", "*.m4a", "*.ogg", "*.flac", "*.mp4", "*.mov", "*.avi", "*.mkv"]
    arquivos_convertidos = 0
    for formato in formatos:
        for f_origem in glob.glob(os.path.join(data_path, '**', formato), recursive=True):
            try:
                f_destino = os.path.splitext(f_origem)[0] + ".wav"
                if os.path.exists(f_destino): continue
                
                audio = AudioSegment.from_file(f_origem)
                audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
                audio.export(f_destino, format="wav")
                
                print(f"Convertido: {os.path.basename(f_origem)} -> {os.path.basename(f_destino)}")
                arquivos_convertidos += 1
            except Exception as e:
                print(f"Erro ao converter {os.path.basename(f_origem)}: {e}")
    if arquivos_convertidos == 0:
        print("Nenhum arquivo novo precisou de conversão.")
    print("------------------------------------------------------")

### NOVA FUNÇÃO: Para salvar a imagem do espectrograma em disco ###
def salvar_espectrograma_imagem(spec_data, sr, output_path):
    """
    Salva os dados de um espectrograma como uma imagem PNG limpa.
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        # Remove eixos, bordas e qualquer espaço em branco
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        
        # Plota o espectrograma
        librosa.display.specshow(spec_data, sr=sr, ax=ax, y_axis='mel', x_axis='time')
        
        # Salva a imagem
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig) # Fecha a figura para liberar memória
    except Exception as e:
        print(f"  -> Erro ao salvar imagem do espectrograma {os.path.basename(output_path)}: {e}")
        # Garante que a figura seja fechada mesmo se houver erro
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)

def gerar_espectrograma_de_arquivo(file_path):
    """
    Carrega um arquivo de áudio e gera um espectrograma em formato de array NumPy.
    MODIFICADO para também retornar os dados brutos para salvamento da imagem.
    """
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        if len(y) < DURATION * sr: y = np.pad(y, (0, DURATION * sr - len(y)))
        
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=IMG_SIZE[0])
        log_S = librosa.power_to_db(S, ref=np.max)
        
        # Redimensiona para o tamanho esperado e garante 3 canais (como uma imagem RGB)
        img_array = tf.image.resize(np.stack((log_S,)*3, axis=-1), [IMG_SIZE[0], IMG_SIZE[1]])
        
        # RETORNA O ARRAY PARA O MODELO E OS DADOS BRUTOS PARA A IMAGEM
        return img_array.numpy(), log_S, sr
    except Exception as e:
        print(f"Erro ao processar espectrograma para {os.path.basename(file_path)}: {e}")
        return None, None, None

def carregar_dados_cnn(data_path):
    """
    Carrega todos os áudios, gera seus espectrogramas em memória E SALVA AS IMAGENS,
    e retorna os dados para o treinamento.
    """
    X, y = [], []
    if not os.path.isdir(data_path):
        print(f"\nERRO: A pasta de dados '{data_path}' não foi encontrada.")
        return None, None
        
    class_names = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    print(f"Encontradas {len(class_names)} classes: {class_names}")

    ### NOVO: Informa ao usuário sobre o salvamento das imagens ###
    print(f"\nAs imagens dos espectrogramas serão salvas em '{SPECTROGRAM_PATH}'")
    os.makedirs(SPECTROGRAM_PATH, exist_ok=True) # Garante que a pasta principal exista

    for label in class_names:
        label_path = os.path.join(data_path, label)
        
        ### NOVO: Cria a subpasta para a classe atual dentro da pasta de espectrogramas ###
        output_class_dir = os.path.join(SPECTROGRAM_PATH, label)
        os.makedirs(output_class_dir, exist_ok=True)

        audio_files = glob.glob(os.path.join(label_path, '*.wav'))
        print(f"Processando {len(audio_files)} arquivos .wav para a classe '{label}'...")
        
        for file_path in audio_files:
            ### MODIFICADO para receber os 3 valores retornados ###
            espectrograma, log_S_bruto, sr = gerar_espectrograma_de_arquivo(file_path)
            
            if espectrograma is not None:
                X.append(espectrograma)
                y.append(label)
                
                ### NOVO: Lógica para salvar a imagem do espectrograma ###
                nome_arquivo_base = os.path.splitext(os.path.basename(file_path))[0]
                caminho_saida_img = os.path.join(output_class_dir, f"{nome_arquivo_base}.png")
                
                # Só salva se a imagem ainda não existir, para poupar tempo em re-treinamentos
                if not os.path.exists(caminho_saida_img):
                    salvar_espectrograma_imagem(log_S_bruto, sr, caminho_saida_img)
    
    return np.array(X), np.array(y)

def construir_modelo_cnn(input_shape, num_classes):
    """
    Constrói a arquitetura da CNN usando MobileNetV2 como base.
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Congela a base inicialmente

    inputs = Input(shape=input_shape)
    x = preprocess_input(inputs) # Camada de pré-processamento da MobileNetV2
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, predictions)
    return model, base_model

# --- 3. FUNÇÕES DE MODO ---

def treinar_modelo():
    """
    Executa todo o pipeline de treinamento da CNN e salva os artefatos.
    """
    # 1. Preparar dados (agora também salva os espectrogramas)
    converter_audios_para_wav(DATA_PATH)
    X, y_text = carregar_dados_cnn(DATA_PATH)

    if X is None or X.shape[0] < 10:
        print("\nTreinamento abortado: não foram encontrados dados suficientes.")
        return

    # 2. Codificar rótulos
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_text)
    y_categorical = to_categorical(y_encoded)
    num_classes = len(encoder.classes_)
    
    np.save(CLASSES_PATH, encoder.classes_)
    print(f"\nClasses salvas em '{CLASSES_PATH}'.")

    # 3. Dividir dados
    X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical)

    # 4. Construir e compilar o modelo
    model, base_model = construir_modelo_cnn(input_shape=IMG_SIZE + (3,), num_classes=num_classes)
    
    # FASE 1: Treinar a cabeça
    print("\n--- FASE 1: Treinando a cabeça do classificador ---")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        X_train, y_train,
        epochs=INITIAL_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # FASE 2: Fine-tuning
    print("\n--- FASE 2: Iniciando o Fine-Tuning ---")
    base_model.trainable = True
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max'),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    model.fit(
        X_train, y_train,
        epochs=TOTAL_EPOCHS,
        initial_epoch=history.epoch[-1],
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n--- Treinamento concluído! Melhor modelo salvo em '{MODEL_PATH}' ---")

def prever_pasta():
    """
    Carrega o modelo CNN treinado e classifica todos os áudios na pasta de dados.
    (Esta função não foi modificada, pois não precisa salvar imagens).
    """
    print(f"\n--- Iniciando modo de previsão na pasta '{DATA_PATH}' ---")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        print("\nERRO: Modelo ou arquivo de classes não encontrado. Rode o modo 'treinar' primeiro.")
        print(f"Use o comando: python {os.path.basename(__file__)} treinar")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = np.load(CLASSES_PATH, allow_pickle=True)
    print("Modelo e classes carregados com sucesso.")
    
    converter_audios_para_wav(DATA_PATH)
    
    arquivos_para_prever = glob.glob(os.path.join(DATA_PATH, '**', '*.wav'), recursive=True)
    if not arquivos_para_prever:
        print("Nenhum arquivo .wav encontrado para prever.")
        return

    random.shuffle(arquivos_para_prever)
    print(f"\nAnalisando {len(arquivos_para_prever)} áudio(s) em ordem aleatória...\n")
    
    for arquivo_wav in arquivos_para_prever:
        espectrograma, _, _ = gerar_espectrograma_de_arquivo(arquivo_wav)
        if espectrograma is None:
            continue
            
        img_array = np.expand_dims(espectrograma, axis=0)
        predictions = model.predict(img_array, verbose=0)
        score = predictions[0]
        
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)
        
        nome_relativo = os.path.relpath(arquivo_wav, DATA_PATH)
        print(f"Arquivo: {nome_relativo:<50} -> Previsão: {predicted_class} (Confiança: {confidence:.2f}%)")

# --- BLOCO PRINCIPAL DE EXECUÇÃO ---
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    if len(sys.argv) < 2 or sys.argv[1] not in ['treinar', 'prever']:
        try:
            script_name = os.path.basename(__file__)
        except NameError:
            script_name = "seu_script.py"
        print("\nUso inválido. Escolha um modo de operação:")
        print(f"   -> Para treinar o modelo: python {script_name} treinar")
        print(f"   -> Para classificar áudios: python {script_name} prever")
    elif sys.argv[1] == 'treinar':
        treinar_modelo()
    elif sys.argv[1] == 'prever':
        prever_pasta()