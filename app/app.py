# Importación de las librerías necesarias
from flask import Flask, request, jsonify, send_from_directory  # Flask y utilidades para manejo de peticiones y respuestas
import joblib  # Para cargar el modelo entrenado
import pandas as pd  # Para manejar datos en formato DataFrame
import os  # Para operaciones del sistema de archivos

# Carga del modelo previamente entrenado desde la ruta correspondiente
modelo = joblib.load('src/modelo.joblib')

# Inicialización de la aplicación Flask
app = Flask(__name__)

# Ruta principal que sirve el archivo index.html desde la carpeta 'static'
@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

# Ruta '/predict' que acepta solicitudes POST con datos en formato JSON
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos enviados en formato JSON
    data = request.get_json()

    try:
        # Extraer las variables necesarias del JSON
        units_sold = data['Units_Sold']
        price_per_unit = data['Price_per_Unit']

        # Crear un DataFrame con los datos para evitar advertencias de scikit-learn
        input_df = pd.DataFrame({'Units_Sold': [units_sold], 'Price_per_Unit': [price_per_unit]})

        # Realizar la predicción con el modelo cargado
        prediccion = modelo.predict(input_df)

        # Convertir la predicción a tipo float y asegurar que no sea negativa
        prediccion_revenue = max(0, float(prediccion[0]))

        # Devolver la predicción en formato JSON
        return jsonify({'prediccion_revenue': prediccion_revenue})

    except Exception as e:
        # En caso de error, devolver un mensaje de error y el código 400
        return jsonify({'error': str(e)}), 400

# Ejecutar la aplicación en modo debug si se ejecuta directamente
if __name__ == '__main__':
    app.run(debug=True)
