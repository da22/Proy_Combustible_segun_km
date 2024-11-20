from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Cargar el modelo
model = joblib.load('xgboost_model.pkl')

# Inicializar Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "API de Predicción activa. Sube tu archivo Excel en el endpoint /predict."

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    try:
        # Leer archivo Excel
        input_data = pd.read_excel(file)
        
        # Preprocesar la columna 'Mes' (si es necesario)
        input_data['Mes'] = pd.Categorical(input_data['Mes']).codes
        
        # Hacer predicciones
        predictions = model.predict(input_data.drop(columns=['ID'], errors='ignore'))
        
        # Agregar predicciones al DataFrame
        input_data['Prediction'] = predictions
        
        # Retornar predicciones en JSON
        return input_data[['ID', 'Prediction']].to_json(orient='records')
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
