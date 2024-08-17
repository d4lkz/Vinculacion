from flask import Flask, request, jsonify
import joblib
import numpy as np

# Cargar el modelo entrenado
model = joblib.load('modelo_entrenado.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "API de predicción está funcionando!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = np.array([
            data['TIPO_DE_DISCAPACIDAD'],
            data['EDAD_ANIOS_MESES'],
            data['PORCENTAJE']
        ]).reshape(1, -1)
        
        prediction = model.predict(features)
        return jsonify(prediction=int(prediction[0]))
    except Exception as e:
        return jsonify(error=str(e)), 400

if __name__ == '__main__':
    app.run(debug=True)
