from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Cargar el modelo
model = joblib.load('xgboost_model.pkl')

# Inicializar Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "API de Predicción activa. Sube tu archivo Excel o CSV en el endpoint /predict."

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    try:
        # Intentar leer el archivo como Excel, si falla, leer como CSV
        try:
            input_data = pd.read_excel(file)
        except ValueError:
            input_data = pd.read_csv(file)

        # Limpiar los nombres de las columnas
        input_data.columns = input_data.columns.str.replace('\xa0', ' ').str.strip()

        # Validar que las columnas necesarias estén presentes
        expected_columns = ['length', 'width', 'area_m2', 'KM Recorrido', 'Mes']
        missing_columns = [col for col in expected_columns if col not in input_data.columns]

        if missing_columns:
            return jsonify({"error": f"Missing columns in input data: {missing_columns}"}), 400

        # Preprocesar la columna 'Mes' (si es necesario)
        input_data['Mes'] = pd.Categorical(input_data['Mes']).codes

        # Hacer predicciones
        predictions = model.predict(input_data[expected_columns])

        # Agregar predicciones al DataFrame
        input_data['Prediction'] = predictions

        # Retornar predicciones en JSON
        return input_data[['ID', 'Prediction']].to_json(orient='records')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
