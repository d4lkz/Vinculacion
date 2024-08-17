import requests

url = 'http://127.0.0.1:5000/predict'

# Datos para la predicción
data = {
    'TIPO_DE_DISCAPACIDAD': '10',
    'EDAD_ANIOS_MESES': 13,
    'PORCENTAJE': 0.7
}

# Enviar solicitud POST a la API
response = requests.post(url, json=data)

# Manejar la respuesta
if response.status_code == 200:
    prediction = response.json()['prediction']
    print(f'Predicción: {prediction}')
else:
    print(f'Error: {response.json()["error"]}')