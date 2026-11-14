from flask import Flask, render_template, request, jsonify
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
import numpy as np
import os

app = Flask(__name__)

# Cargar modelo LiteRT
print("Cargando modelo de IA...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo.tflite")

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Modelo cargado exitosamente.")


# Ruta para servir la interfaz web
@app.route('/')
def index():
    return render_template('index.html')


# Ruta para procesar la imagen
@app.route('/predecir', methods=['POST'])
def predecir():

    if 'imagen' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'}), 400

    archivo = request.files['imagen']

    if archivo.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400

    try:
        # Preprocesamiento robusto para evitar sesgos en el modelo
        img = Image.open(archivo.stream).convert('L')
        
        # Redimensionar a 28x28 con alta calidad
        img_redimensionada = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convertir a matriz numérica
        img_array = np.array(img_redimensionada, dtype=np.float32)
        
        # Invertir colores automáticamente si el fondo es claro (para que coincida con MNIST)
        if np.mean(img_array) > 127:
            img_array = 255.0 - img_array

        # Adaptar según el tipo de datos que espera el modelo
        input_dtype = input_details[0]['dtype']
        if input_dtype == np.float32:
            img_array = img_array / 255.0

        # Agregar dimensión del batch
        img_lista = np.expand_dims(img_array, axis=0)

        # Manejar dimensiones de canales si el modelo los exige
        expected_shape = tuple(input_details[0]['shape'])
        if len(expected_shape) == 4 and img_lista.ndim == 3:
            img_lista = np.expand_dims(img_lista, axis=-1)

        if tuple(img_lista.shape) != expected_shape:
            try:
                img_lista = np.reshape(img_array, expected_shape)
            except Exception:
                return jsonify({
                    'error': (
                        f"Dimensiones incompatibles. "
                        f"Esperadas: {expected_shape}, "
                        f"recibidas: {img_lista.shape}"
                    )
                }), 500

        # Ejecutar predicción con LiteRT
        interpreter.set_tensor(
            input_details[0]['index'],
            img_lista.astype(input_dtype)
        )

        interpreter.invoke()

        predicciones = interpreter.get_tensor(
            output_details[0]['index']
        )

        # Obtener resultado real
        numero_predicho = int(np.argmax(predicciones[0]))
        confianza = float(np.max(predicciones[0]) * 100)

        return jsonify({
            'numero_predicho': numero_predicho,
            'confianza': round(confianza, 2)
        })

    except Exception as e:
        print(f"Error en predicción: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
