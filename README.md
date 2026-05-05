# Predicción de Demanda de Platos

Aplicación web desarrollada en **Python + Streamlit + Plotly** para predecir la demanda de platos mediante modelos de Machine Learning previamente entrenados y guardados en archivos `.pkl`.

La app permite trabajar en dos modos:

1. **Archivo cargado:** el usuario sube un archivo Excel o CSV con datos históricos y la app genera predicciones, tablas, gráficos e indicadores de calidad.
2. **Predicción futura 6 meses:** la app genera fechas futuras y estima la demanda esperada para los próximos meses usando los modelos entrenados.

---

## Objetivo del proyecto

El objetivo de esta aplicación es permitir que un usuario pueda:

- Subir un archivo `.csv` o `.xlsx`.
- Validar que el archivo tenga la estructura mínima requerida.
- Ejecutar modelos de predicción ya entrenados.
- Visualizar resultados por día, semana y mes.
- Consultar insights automáticos.
- Evaluar qué producto tiene mejor comportamiento predictivo.
- Descargar un archivo Excel con los resultados generados.

---

## Tecnologías utilizadas

- **Python**
- **Streamlit** para la interfaz web.
- **Pandas** para limpieza, transformación y análisis de datos.
- **NumPy** para operaciones numéricas.
- **Plotly** para gráficos dinámicos.
- **Scikit-learn** para cargar y ejecutar los modelos de Machine Learning.
- **Joblib** para leer los modelos `.pkl`.
- **OpenPyXL / XlsxWriter** para lectura y exportación de archivos Excel.

---

## Estructura recomendada del proyecto

La estructura del repositorio debe ser la siguiente:

```text
app_prediccion_demanda/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── modelos/
│   ├── modelo_almuerzo.pkl
│   ├── modelo_sopa.pkl
│   ├── modelo_fanesca.pkl
│   ├── modelo_colada_morada.pkl
│   └── config_entrenamiento.pkl
│
└── data_ejemplo/
    └── archivo_ejemplo.xlsx
```

La carpeta `venv/` no debe subirse al repositorio porque puede ser muy pesada y depende de la configuración local de cada equipo.

---

## Archivos principales

### `app.py`

Archivo principal de la aplicación. Contiene:

- Interfaz de usuario en Streamlit.
- Carga de archivos CSV o Excel.
- Validaciones de estructura.
- Limpieza y preparación de datos.
- Generación de variables predictoras.
- Carga de modelos `.pkl`.
- Predicción histórica o futura.
- Visualización de tablas y gráficos.
- Generación de insights automáticos.
- Evaluación de calidad por producto.
- Descarga de resultados en Excel.

---

### `requirements.txt`

Archivo que contiene las librerías necesarias para ejecutar el proyecto.

Ejemplo recomendado:

```txt
streamlit
pandas
numpy
plotly
openpyxl
xlsxwriter
joblib
scikit-learn==1.6.1
```

Se recomienda mantener `scikit-learn==1.6.1`, ya que los modelos `.pkl` fueron generados con esa versión.

---

### Carpeta `modelos/`

Contiene los modelos entrenados y el archivo de configuración.

Debe incluir:

```text
modelo_almuerzo.pkl
modelo_sopa.pkl
modelo_fanesca.pkl
modelo_colada_morada.pkl
config_entrenamiento.pkl
```

Cada modelo corresponde a un producto específico:

| Archivo | Producto que predice |
|---|---|
| `modelo_almuerzo.pkl` | Almuerzo |
| `modelo_sopa.pkl` | Sopa |
| `modelo_fanesca.pkl` | Fanesca |
| `modelo_colada_morada.pkl` | Colada morada |

Todos los modelos son modelos de regresión basados en **Random Forest**.

---

### `config_entrenamiento.pkl`

Archivo de configuración utilizado para mantener información del entrenamiento, como:

- Productos del modelo.
- Variables predictoras.
- Rango histórico de entrenamiento.
- Parámetros generales.
- Temporadas especiales, como fanesca y colada morada.

Este archivo permite que la app sea más flexible y no dependa únicamente de valores escritos directamente en el código.

---

## Variables principales utilizadas

La app genera variables temporales, estacionales y de precio para ejecutar los modelos.

Algunas variables utilizadas son:

```text
anio
mes
dia_mes
dia_semana_num
semana_anio
es_fanesca_temporada
es_colada_temporada
es_inicio_mes
es_quincena
es_fin_mes
es_lunes
es_martes
es_miercoles
es_jueves
es_viernes
mes_sin
mes_cos
dia_semana_sin
dia_semana_cos
tendencia
tendencia_log
crecimiento_anual
preciomenu
preciosopa
fanesca_precio
coladamorada_precio
```

La aplicación también valida las variables que realmente usa cada modelo `.pkl`, de modo que puede evitar errores si el archivo de configuración contiene más variables que algún modelo guardado.

---

## Formato del archivo de entrada

En modo **Archivo cargado**, el usuario puede subir archivos en formato:

```text
.csv
.xlsx
```

El archivo debe contener como mínimo las siguientes columnas:

```text
Fecha
Tipo_plato
Clientes
```

Columnas recomendadas:

```text
Facturación
Precio menú
Precio sopa
Precio fanesca
Precio colada morada
```

Ejemplo de estructura:

| Fecha | Tipo_plato | Clientes | Facturación | preciomenu | preciosopa | fanesca_precio | coladamorada_precio |
|---|---|---:|---:|---:|---:|---:|---:|
| 2025-01-02 | almuerzo | 120 | 540 | 4.5 | 1.8 | 10 | 3.5 |
| 2025-01-02 | sopa | 80 | 144 | 4.5 | 1.8 | 10 | 3.5 |
| 2025-01-03 | colada_morada | 25 | 87.5 | 4.5 | 1.8 | 10 | 3.5 |

---

## Validaciones realizadas por la app

La aplicación valida:

- Si el archivo está vacío.
- Si falta la columna `Fecha`.
- Si falta la columna `Tipo_plato`.
- Si falta la columna `Clientes`.
- Si existen fechas inválidas.
- Si existen valores negativos en clientes.
- Si existen precios negativos.
- Si existen tipos de plato no reconocidos.
- Si el archivo contiene fechas fuera del rango histórico del modelo.

Si faltan columnas de precio, la app puede completar precios por defecto según reglas internas.

---

## Modos de ejecución

### 1. Archivo cargado

En este modo, el usuario sube un Excel o CSV.

La app:

1. Lee el archivo.
2. Valida columnas mínimas.
3. Limpia datos.
4. Transforma los datos.
5. Genera variables predictoras.
6. Aplica los modelos `.pkl`.
7. Muestra predicciones.
8. Calcula métricas de evaluación si existen valores reales en `Clientes`.

Este modo permite comparar valores reales contra valores predichos.

---

### 2. Predicción futura 6 meses

En este modo, la app no requiere archivo cargado.

El usuario configura:

- Fecha inicial futura.
- Número de meses a predecir.
- Precio de menú.
- Precio de sopa.
- Precio de fanesca.
- Precio de colada morada.

La app genera fechas futuras de lunes a viernes y aplica los modelos entrenados para estimar la demanda.

Este modo no muestra métricas de evaluación, porque no existen valores reales futuros contra los cuales comparar.

---

## Resultados mostrados en la app

La aplicación muestra:

### KPIs generales

- Total estimado.
- Producto líder.
- Día fuerte.
- Mes fuerte.
- Fecha pico.

### Insights automáticos

Ejemplos:

- Producto con mayor demanda: Almuerzo.
- El día con mayor consumo estimado es Miércoles.
- El mes con mayor consumo estimado es Octubre.
- Total estimado: 12.450 platos.

### Consulta por periodo

El usuario puede consultar resultados por:

- Día.
- Semana.
- Mes.

### Gráficos dinámicos

La app incluye gráficos de:

- Predicción diaria.
- Consumo semanal.
- Consumo mensual.
- Error por producto.
- Total real vs total predicho.

Los gráficos de barras incluyen valores visibles dentro de las barras para facilitar la lectura.

---

## Métricas de evaluación

Cuando se carga un archivo histórico con valores reales en `Clientes`, la app calcula métricas por producto.

Las métricas incluidas son:

| Métrica | Interpretación |
|---|---|
| RMSE | Penaliza más los errores grandes. Menor es mejor. |
| MAPE % | Error porcentual promedio. Menor es mejor. |
| Precisión aproximada % | Referencia calculada como `100 - MAPE`. Mayor es mejor. |
| R² | Indica qué tanto se ajusta el modelo a los datos. Mientras más cercano a 1, mejor. |
| Sesgo promedio | Indica si el modelo tiende a sobreestimar o subestimar. |

La app no usa MAE en la versión actual.

---

## Archivo Excel descargable

La app permite descargar un Excel con varias hojas:

```text
predicciones
detalle_diario_largo
agregado_semanal
agregado_mensual
insights
metricas_productos
diagnostico_modelos
```

Este archivo permite revisar los resultados fuera de la aplicación.

---

## Instalación del proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/USUARIO/NOMBRE_REPOSITORIO.git
cd NOMBRE_REPOSITORIO
```

Reemplaza `USUARIO/NOMBRE_REPOSITORIO` por el enlace real del repositorio.

---

### 2. Crear entorno virtual

Se recomienda usar **Python 3.11**.

```bash
py -3.11 -m venv venv
```

---

### 3. Activar entorno virtual

En CMD:

```cmd
venv\Scripts\activate.bat
```

En PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

Si PowerShell bloquea la activación, ejecutar:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\Activate.ps1
```

---

### 4. Instalar dependencias

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

---

### 5. Ejecutar la aplicación

```bash
python -m streamlit run app.py
```

La aplicación se abrirá normalmente en:

```text
http://localhost:8501
```

---

## Comandos rápidos

Si ya tienes el entorno virtual creado:

```cmd
cd C:\ruta\del\proyecto
venv\Scripts\activate.bat
python -m streamlit run app.py
```

Si necesitas reinstalar dependencias:

```cmd
python -m pip install -r requirements.txt
```

---

## Problemas comunes

### Error: `streamlit` no se reconoce como comando

Usar:

```bash
python -m streamlit run app.py
```

---

### Error: no se encuentra `scikit-learn`

Instalar dependencias:

```bash
python -m pip install -r requirements.txt
```

---

### Error con Python 3.14

Se recomienda usar Python 3.11, porque los modelos `.pkl` fueron generados con `scikit-learn==1.6.1`.

Verificar versión:

```bash
python --version
```

---

### Error: no se encontraron modelos

Verificar que exista la carpeta `modelos/` con estos archivos:

```text
modelo_almuerzo.pkl
modelo_sopa.pkl
modelo_fanesca.pkl
modelo_colada_morada.pkl
config_entrenamiento.pkl
```

---

### Error: el archivo no contiene columna Fecha

Verificar que el archivo subido tenga una columna llamada `Fecha`, `fecha`, `date` o un nombre equivalente reconocido por la app.

---

## Recomendaciones para GitHub

Crear un archivo `.gitignore` con este contenido:

```gitignore
venv/
__pycache__/
*.pyc
.env
.streamlit/secrets.toml
.ipynb_checkpoints/
outputs/
*.log
```

No subir la carpeta `venv/`.

Sí subir:

```text
app.py
requirements.txt
README.md
.gitignore
modelos/
data_ejemplo/
```

---

## Comandos para subir a GitHub

Si el repositorio ya existe:

```bash
git status
git add app.py requirements.txt README.md .gitignore modelos data_ejemplo
git commit -m "Agregar app de prediccion de demanda"
git push
```

Si aún no está conectado a un remoto:

```bash
git init
git add .
git commit -m "Agregar app de prediccion de demanda"
git branch -M main
git remote add origin https://github.com/USUARIO/NOMBRE_REPOSITORIO.git
git push -u origin main
```

---

## Consideraciones importantes

- Los modelos `.pkl` solo deben cargarse si provienen de una fuente confiable.
- Las predicciones futuras deben interpretarse como estimaciones.
- La calidad del resultado depende de la calidad del dataset histórico.
- Si cambian los modelos, se debe actualizar la carpeta `modelos/`.
- Si cambian las variables predictoras, se debe actualizar `config_entrenamiento.pkl`.

---

## Autoría y contexto

Este proyecto fue desarrollado como una aplicación de ciencia de datos para estimar demanda de platos usando modelos de Machine Learning, visualización interactiva y consulta por periodos.

La solución permite pasar de un análisis en notebooks a una aplicación funcional, más amigable y fácil de usar para usuarios no técnicos.
