# Predicciones de Fórmula 1 2025

Este proyecto utiliza machine learning para predecir los resultados de las carreras de Fórmula 1, específicamente el Gran Premio de China 2025.

## Características

- Predicción de tiempos de carrera basada en datos históricos
- Uso de Gradient Boosting para el modelo de predicción
- Análisis de datos de múltiples carreras
- Visualización de resultados

## Requisitos

- Python 3.9+
- fastf1
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/benjamack/DataScience.git
cd DataScience
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

Para ejecutar las predicciones:
```bash
python prediction1.py
```

## Resultados

El modelo predice los tiempos de carrera para el GP de China 2025, utilizando datos históricos de las primeras carreras de 2024.

## Métricas del Modelo

- Error Absoluto Medio (MAE): 3.47 segundos

## Estructura del Proyecto

- `prediction1.py`: Script principal de predicción
- `requirements.txt`: Lista de dependencias
- `f1_cache/`: Directorio de caché para datos de FastF1

# 🏎️ F1 Predictions 2025 - Machine Learning Model

Welcome to the **F1 Predictions 2025** repository! This project uses **machine learning, FastF1 API data, and historical F1 race results** to predict race outcomes for the 2025 Formula 1 season.

## 🚀 Project Overview
This repository contains a **Gradient Boosting Machine Learning model** that predicts race results based on past performance, qualifying times, and other structured F1 data. The model leverages:
- FastF1 API for historical race data
- 2024 race results
- 2025 qualifying session results
- Over the course of the season we will be adding additional data to improve our model as well
- Feature engineering techniques to improve predictions

## 📊 Data Sources
- **FastF1 API**: Fetches lap times, race results, and telemetry data
- **2025 Qualifying Data**: Used for prediction
- **Historical F1 Results**: Processed from FastF1 for training the model

## 🏁 How It Works
1. **Data Collection**: The script pulls relevant F1 data using the FastF1 API.
2. **Preprocessing & Feature Engineering**: Converts lap times, normalizes driver names, and structures race data.
3. **Model Training**: A **Gradient Boosting Regressor** is trained using 2024 race results.
4. **Prediction**: The model predicts race times for 2025 and ranks drivers accordingly.
5. **Evaluation**: Model performance is measured using **Mean Absolute Error (MAE)**.

### Dependencies
- `fastf1`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

## File Structure 
- For every race the end of the file will be numbered in correlation to the race on the calendar, ex. prediction1 - Australia, prediction2 - China, etc.

## 🔧 Usage
Run the prediction script:
```bash
python3 prediction1.py
```
Expected output:
```
🏁 Predicted 2025 Australian GP Winner 🏁
Driver: Charles Leclerc, Predicted Race Time: 82.67s
...
🔍 Model Error (MAE): 3.22 seconds
```

## 📈 Model Performance
The Mean Absolute Error (MAE) is used to evaluate how well the model predicts race times. Lower MAE values indicate more accurate predictions.

## 📌 Future Improvements
- Incorporate **weather conditions** as a feature
- Add **pit stop strategies** into the model
- Explore **deep learning** models for improved accuracy
- @mar_antaya on Instagram and TikTok will update with the latest predictions before every race of the 2025 F1 season

## 📜 License
This project is licensed under the MIT License.


🏎️ **Start predicting F1 races like a data scientist!** 🚀

