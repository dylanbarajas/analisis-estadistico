import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Datos reales
datos_ganancia_reales = np.array([
    30702.00, 52851.20, 26852.00, 26852.00, 32102.00, 32102.00, 32102.00, 51451.20, 32102.00, 
    37350.00, 73262.50, 32100.00, 54790.00, 73262.50, 32100.00, 54790.00, 27595.00, 27595.00, 
    32100.00, 32100.00, 32100.00, 54790.00, 54790.00, 41200.00, 41200.00, 73262.50, 41200.00, 
    73262.50, 45050.00, 54790.00, 45050.00, 77681.87, 45050.00, 58640.00, 54790.00, 46450.00, 
    77112.50, 73262.50, 45050.00, 41200.00, 54790.00, 44416.00, 60040.00, 41200.00, 41200.00, 
    42440.00, 0.00, 56552.50, 51430.00, 51430.00, 51430.00, 34290.00, 28866.00, 45174.00, 
    45174.00, 49024.00, 63302.50, 63302.50, 37015.50, 42407.00, 36450.00, 36450.00, 27160.00, 
    27160.00, 35050.00, 31200.00, 36450.00, 31200.00, 22050.00, 27300.00, 27300.00, 22050.00, 
    22050.00, 22050.00, 27300.00, 25900.00, 20482.00, 22050.00, 22050.00, 27300.00, 22050.00, 
    39941.67, 19970.83, 25900.00, 21490.00, 27300.00, 21700.00, 22050.00, 22050.00, 27300.00, 
    22050.00, 22050.00, 22050.00, 22050.00, 22050.00, 27300.00, 22050.00, 22050.00, 22050.00, 
    22050.00, 20155.00, 20155.00, 22050.00, 22050.00, 27300.00, 22050.00, 0.00, 22050.00, 
    27300.00, 22080.00, 22080.00, 44160.00, 25900.00, 27300.00, 33614.00, 22780.00, 22780.00, 
    21254.17, 42508.33, 22050.00, 22050.00, 45560.00, 22890.00, 27300.00, 22890.00, 22050.00, 
    22050.00, 27300.00, 27300.00, 22050.00, 22050.00, 25900.00, 27300.00, 22050.00, 17640.00, 
    17640.00, 22050.00, 22050.00, 23940.00, 22050.00, 17640.00, 22050.00, 22050.00, 22050.00, 
    53930.00, 52530.00, 22050.00, 21700.00, 21700.00, 21700.00, 25300.00, 21700.00, 21700.00, 
    21700.00, 25300.00, 21700.00, 21700.00, 21700.00, 25300.00, 25300.00, 25300.00, 25300.00, 
    21700.00, 21700.00, 21700.00, 21700.00, 21700.00, 23600.00, 21700.00, 21700.00, 21700.00, 
    21700.00, 25300.00, 25300.00, 25300.00, 53700.00, 21700.00, 25300.00, 21700.00, 25300.00, 
    25300.00, 21700.00, 21700.00, 16690.00, 20290.00, 16690.00, 18590.00, 16690.00, 20290.00, 
    44682.00, 41082.00, 16690.00, 41082.00, 41082.00, 20290.00, 16690.00, 20290.00, 16690.00, 
    18590.00, 16690.00, 21700.00, 25300.00, 20575.00, 24175.00, 20575.00
])

# Crear DataFrame
df = pd.DataFrame(datos_ganancia_reales, columns=['Ganancia'])

# Estadísticas básicas
n = len(datos_ganancia_reales)
mean = np.mean(datos_ganancia_reales)
std_dev = np.std(datos_ganancia_reales, ddof=1)
stderr = std_dev / np.sqrt(n)

# Calcular intervalos usando la distribución normal estándar
z_value = norm.ppf(1 - 0.025)

# Intervalo de confianza al 95%
ci_lower = mean - z_value * stderr
ci_upper = mean + z_value * stderr

# Intervalo de predicción al 95%
pred_lower = mean - z_value * np.sqrt(1 + 1/n) * std_dev
pred_upper = mean + z_value * np.sqrt(1 + 1/n) * std_dev

# Intervalo de tolerancia al 95%
tolerance_interval = norm.interval(0.95, loc=mean, scale=std_dev)

# Mostrar los datos en Streamlit
st.title("Análisis de Ganancias")
st.write("**Datos de Ganancia**")
st.dataframe(df)

# Estadísticas calculadas
st.write("**Estadísticas Básicas**")
st.metric("Promedio", f"${mean:.2f}")
st.metric("Desviación Estándar", f"${std_dev:.2f}")

# Mostrar intervalos
st.write("**Intervalos Calculados (95%)**")
st.write(f"Intervalo de Confianza: [{ci_lower:.2f}, {ci_upper:.2f}]")
st.write(f"Intervalo de Predicción: [{pred_lower:.2f}, {pred_upper:.2f}]")
st.write(f"Intervalo de Tolerancia: [{tolerance_interval[0]:.2f}, {tolerance_interval[1]:.2f}]")

# Histograma de las ganancias
fig1, ax1 = plt.subplots()
ax1.hist(df['Ganancia'], bins=20, color='skyblue', edgecolor='black')
ax1.axvline(mean, color='red', linestyle='dashed', linewidth=1, label="Promedio")
ax1.axvline(ci_lower, color='green', linestyle='dashed', linewidth=1, label="Confianza 95%")
ax1.axvline(ci_upper, color='green', linestyle='dashed', linewidth=1)
ax1.legend()
st.pyplot(fig1)

# Crear figura de la distribución normal
x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
y = norm.pdf(x, mean, std_dev)

# Crear figura de la distribución normal
fig, ax = plt.subplots()
ax.plot(x, y, label="Distribución Normal", color='blue')
ax.axvline(mean, color='red', linestyle='dashed', linewidth=1, label="Promedio")
ax.axvline(mean - 1.96 * stderr, color='green', linestyle='dashed', linewidth=1, label="Confianza 95% (inferior)")
ax.axvline(mean + 1.96 * stderr, color='green', linestyle='dashed', linewidth=1, label="Confianza 95% (superior)")
ax.fill_between(x, 0, y, where=(x >= mean - 1.96 * stderr) & (x <= mean + 1.96 * stderr), color='green', alpha=0.2)
ax.legend()
ax.set_title("Distribución Normal de las Ganancias")
ax.set_xlabel("Ganancia")
ax.set_ylabel("Densidad de probabilidad")

# Mostrar en Streamlit
st.pyplot(fig)