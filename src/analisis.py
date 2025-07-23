#mismo codigo del archivo analisis.ipynb sin comentarios

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

print("Librerías importadas correctamente.")

ruta = "../data/ventas.csv"
df = pd.read_csv(ruta)
print("Datos cargados correctamente.")
df.head()

print("Dimensiones del dataset:", df.shape)
print("------------------------------------------------------------------------------------")
print("Valores nulos por columna:")
print(df.isnull().sum())
print("------------------------------------------------------------------------------------")
print("Tipos de datos:")
print(df.dtypes)
print("------------------------------------------------------------------------------------")
print("Número de duplicados:", df.duplicated().sum())

df = df.drop_duplicates()
df = df.dropna()

plt.figure(figsize=(8,5))
plt.scatter(df['Units_Sold'], df['Revenue'], alpha=0.5)
plt.title('Unidades Vendidas vs Ingresos')
plt.xlabel('Units Sold')
plt.ylabel('Revenue')
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(df['Price_per_Unit'], df['Revenue'], alpha=0.5, color='orange')
plt.title('Precio por Unidad vs Ingresos')
plt.xlabel('Price per Unit')
plt.ylabel('Revenue')
plt.grid(True)
plt.show()

correlaciones = df.corr(numeric_only=True)
print("Correlación con Revenue:")
print(correlaciones['Revenue'].sort_values(ascending=False))

X_simple = df[['Units_Sold']]
y = df['Revenue']
modelo_simple = LinearRegression()
modelo_simple.fit(X_simple, y)
print("Modelo Simple:")
print(f" Coeficiente (pendiente) Units_Sold: {modelo_simple.coef_[0]:.4f}")
print(f" Intercepto: {modelo_simple.intercept_:.4f}")
print()

X_multi = df[['Units_Sold', 'Price_per_Unit']]
modelo_multi = LinearRegression()
modelo_multi.fit(X_multi, y)
print("Modelo Multivariado:")
print(f" Coeficientes (Units_Sold, Price_per_Unit): {modelo_multi.coef_}")
print(f" Intercepto: {modelo_multi.intercept_:.4f}")

y_pred_simple = modelo_simple.predict(X_simple)
mae_simple = mean_absolute_error(y, y_pred_simple)
mse_simple = mean_squared_error(y, y_pred_simple)
r2_simple = r2_score(y, y_pred_simple)
print("Evaluación del Modelo Simple:")
print(f" MAE (Error absoluto medio): {mae_simple:.2f}")
print(f" MSE (Error cuadrático medio): {mse_simple:.2f}")
print(f" R² (Explicación de la varianza): {r2_simple:.4f}\n")

y_pred_multi = modelo_multi.predict(X_multi)
mae_multi = mean_absolute_error(y, y_pred_multi)
mse_multi = mean_squared_error(y, y_pred_multi)
r2_multi = r2_score(y, y_pred_multi)
print("Evaluación del Modelo Multivariado:")
print(f" MAE (Error absoluto medio): {mae_multi:.2f}")
print(f" MSE (Error cuadrático medio): {mse_multi:.2f}")
print(f" R² (Explicación de la varianza): {r2_multi:.4f}")

plt.figure(figsize=(8,5))
plt.scatter(X_simple, y, label='Datos Reales', alpha=0.6)
plt.plot(X_simple, y_pred_simple, color='red', label='Predicción del Modelo')
plt.title('Regresión Lineal Simple: Units_Sold vs Revenue')
plt.xlabel('Units_Sold')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(y, y_pred_multi, alpha=0.5, color='blue', label='Predicciones del Modelo')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2 , label='Predicción Perfecta (y = x)')
plt.xlabel('Valores Reales (Revenue)')
plt.ylabel('Predicciones del Modelo')
plt.title('Regresión Lineal Múltiple: Units_Sold y Price_per_Unit vs Revenue')
plt.legend()
plt.grid(True)
plt.show()

joblib.dump(modelo_multi, '../src/modelo.joblib')
print("Modelo guardado como src/modelo.joblib")
