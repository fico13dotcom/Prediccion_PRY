# Predicción de Demanda de Platos

Aplicación web desarrollada para estimar la demanda de diferentes tipos de platos a partir de datos históricos y modelos de Machine Learning previamente entrenados.

La solución permite cargar información en formato Excel o CSV, ejecutar modelos predictivos y visualizar resultados de forma clara mediante tablas, gráficos e indicadores automáticos.

---

## Objetivo

El objetivo principal de esta aplicación es facilitar la predicción de demanda de platos como:

- Almuerzo
- Sopa
- Fanesca
- Colada morada

La app permite que el usuario pueda consultar estimaciones por día, semana o mes, así como generar predicciones futuras para apoyar la planificación operativa y la toma de decisiones.

---

## Funcionalidades principales

- Carga de archivos `.xlsx` o `.csv`.
- Validación básica de estructura del archivo.
- Limpieza y preparación automática de datos.
- Ejecución de modelos predictivos entrenados.
- Predicción de demanda por tipo de plato.
- Visualización de resultados mediante gráficos interactivos.
- Consulta de resultados por día, semana o mes.
- Generación de insights automáticos.
- Evaluación de calidad de predicción por producto.
- Descarga de resultados en formato Excel.

---

## Tecnologías utilizadas

El proyecto fue desarrollado con las siguientes tecnologías:

- **Python** como lenguaje principal.
- **Streamlit** para construir la interfaz web.
- **Pandas** para el procesamiento y transformación de datos.
- **NumPy** para operaciones numéricas.
- **Plotly** para gráficos interactivos.
- **Scikit-learn** para el uso de modelos de Machine Learning.
- **Joblib** para cargar modelos entrenados en formato `.pkl`.
- **OpenPyXL / XlsxWriter** para lectura y generación de archivos Excel.

---

## Modelos utilizados

La aplicación utiliza modelos de Machine Learning guardados en archivos `.pkl`.

Cada modelo está asociado a un tipo de plato específico:

- `modelo_almuerzo.pkl`
- `modelo_sopa.pkl`
- `modelo_fanesca.pkl`
- `modelo_colada_morada.pkl`

Además, se incluye un archivo de configuración:

- `config_entrenamiento.pkl`

Este archivo permite mantener información relacionada con las variables y parámetros usados durante el entrenamiento de los modelos.

---

## Estructura general del proyecto

```text
Prediccion_PRY/
│
├── app.py
├── requirements.txt
├── README.md
│
└── modelos/
    ├── modelo_almuerzo.pkl
    ├── modelo_sopa.pkl
    ├── modelo_fanesca.pkl
    ├── modelo_colada_morada.pkl
    └── config_entrenamiento.pkl
```

---

## Ejecución del proyecto

Para ejecutar la aplicación localmente:

```bash
python -m streamlit run app.py
```

La aplicación se abrirá en el navegador, normalmente en:

```text
http://localhost:8501
```

---

## Consideraciones

- La carpeta `venv/` no debe subirse al repositorio.
- Los modelos `.pkl` deben estar dentro de la carpeta `modelos/`.
- Las predicciones futuras deben interpretarse como estimaciones.
- La calidad de los resultados depende de la calidad de los datos históricos utilizados.
