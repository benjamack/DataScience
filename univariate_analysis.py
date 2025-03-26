import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import os

# Crear directorio para plots si no existe
os.makedirs('plots', exist_ok=True)

# Configuración de visualización
plt.style.use('seaborn')
sns.set_palette("husl")

# Habilitar caché de FastF1
fastf1.Cache.enable_cache("f1_cache")

# Cargar datos de múltiples carreras para tener una muestra más representativa
races = [
    (2024, "Australia", "R"),
    (2024, "China", "R"),
    (2024, "Japan", "R")
]

# Lista para almacenar todos los datos
all_data = []

for year, race, session in races:
    print(f"Cargando datos de {race} {year}...")
    session_data = fastf1.get_session(year, race, session)
    session_data.load()
    
    # Extraer datos relevantes
    laps = session_data.laps.copy()
    
    # Convertir tiempos a segundos y manejar valores nulos
    laps['LapTime_seconds'] = pd.to_numeric(laps['LapTime'].dt.total_seconds(), errors='coerce')
    
    # Calcular posición final para cada piloto
    final_positions = laps.groupby('Driver')['Position'].last().reset_index()
    final_positions['Race'] = f"{race} {year}"
    
    # Calcular estadísticas por piloto
    driver_stats = laps.groupby('Driver').agg({
        'LapTime_seconds': ['mean', 'std', 'min']
    }).reset_index()
    
    # Aplanar nombres de columnas
    driver_stats.columns = ['Driver'] + [f"{col[0]}_{col[1]}" for col in driver_stats.columns[1:]]
    
    # Combinar con posiciones finales
    race_data = pd.merge(final_positions, driver_stats, on='Driver')
    all_data.append(race_data)

# Combinar todos los datos
df = pd.concat(all_data, ignore_index=True)

# Limpiar datos
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# Análisis univariado
def analyze_variable(race_df, var_name, title, race_name):
    plt.figure(figsize=(15, 5))
    
    # Gráfico de dispersión con línea de tendencia
    plt.subplot(1, 3, 1)
    sns.regplot(data=race_df, x=var_name, y='Position', scatter_kws={'alpha':0.5})
    plt.title(f'Posición vs {title} - {race_name}')
    
    # Gráfico de caja
    plt.subplot(1, 3, 2)
    sns.boxplot(data=race_df, y=var_name)
    plt.title(f'Distribución de {title} - {race_name}')
    
    # Histograma con KDE
    plt.subplot(1, 3, 3)
    sns.histplot(data=race_df, x=var_name, bins=20, kde=True)
    plt.title(f'Histograma de {title} - {race_name}')
    
    plt.tight_layout()
    plt.savefig(os.path.join('plots', f'analysis_{var_name}_{race_name.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calcular correlación y estadísticas
    correlation = stats.pearsonr(race_df[var_name], race_df['Position'])
    print(f"\nAnálisis de {title} - {race_name}:")
    print(f"Correlación con posición: {correlation[0]:.3f}")
    print(f"P-valor: {correlation[1]:.3f}")
    print(f"Media: {race_df[var_name].mean():.3f}")
    print(f"Desviación estándar: {race_df[var_name].std():.3f}")
    print(f"Mínimo: {race_df[var_name].min():.3f}")
    print(f"Máximo: {race_df[var_name].max():.3f}")
    print(f"Mediana: {race_df[var_name].median():.3f}")

# Variables a analizar
variables = {
    'LapTime_seconds_mean': 'Tiempo Medio de Vuelta',
    'LapTime_seconds_std': 'Desviación Estándar de Tiempos',
    'LapTime_seconds_min': 'Mejor Tiempo de Vuelta'
}

# Análisis por carrera
for race_name in df['Race'].unique():
    race_df = df[df['Race'] == race_name]
    print(f"\nEstadísticas descriptivas para {race_name}:")
    print("=" * 50)
    print(race_df[list(variables.keys()) + ['Position']].describe())
    
    # Realizar análisis para cada variable
    for var, title in variables.items():
        analyze_variable(race_df, var, title, race_name)

# Análisis adicional: Correlación entre variables por carrera
for race_name in df['Race'].unique():
    race_df = df[df['Race'] == race_name]
    plt.figure(figsize=(10, 8))
    correlation_matrix = race_df[list(variables.keys()) + ['Position']].corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0, 
                fmt='.2f',
                square=True,
                linewidths=.5)
    plt.title(f'Matriz de Correlación entre Variables - {race_name}')
    plt.tight_layout()
    plt.savefig(os.path.join('plots', f'correlation_matrix_{race_name.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Imprimir resumen de hallazgos por carrera
print("\nResumen de Hallazgos por Carrera:")
print("=" * 50)
for race_name in df['Race'].unique():
    race_df = df[df['Race'] == race_name]
    print(f"\n{race_name}:")
    print("-" * 30)
    for var, title in variables.items():
        correlation = stats.pearsonr(race_df[var], race_df['Position'])[0]
        print(f"{title}:")
        print(f"  Correlación con posición: {correlation:.3f}")
        print(f"  Interpretación: {'Fuerte' if abs(correlation) > 0.7 else 'Moderada' if abs(correlation) > 0.4 else 'Débil'}")
        print("-" * 20) 