# 7.1.3 Plan de pruebas y validación del prototipo

Con el propósito de verificar el correcto funcionamiento del prototipo de predicción de demanda, se definió un plan de pruebas orientado a validar la integridad de los datos de entrada, el tratamiento de errores y la generación de predicciones. Este plan incluye casos de prueba con entradas válidas e inválidas, criterios de aceptación y evidencias de ejecución, con el fin de asegurar la trazabilidad y confiabilidad del sistema desarrollado. La tabla presenta los principales casos de prueba aplicados al prototipo.

| ID | Fase | Caso de prueba | Descripción | Criterio de aceptación | Evidencia |
|---|---|---|---|---|---|
| CP-01 | Limpieza de datos | Carga del dataset original | Verificar que el archivo original pueda cargarse y normalizarse correctamente. | El archivo se lee sin errores y se identifican columnas esenciales. | dataset_limpio.xlsx / logs de limpieza |
| CP-02 | Limpieza de datos | Validación de columnas obligatorias | Confirmar la existencia de Fecha, Tipo_plato y Clientes. | El sistema detecta columnas faltantes y muestra error claro. | Reporte de validación / mensaje de error |
| CP-03 | Limpieza de datos | Tratamiento de fechas inválidas | Evaluar registros con fechas incorrectas o nulas. | El sistema excluye o reporta registros inválidos. | Reporte de limpieza |
| CP-04 | Preparación de datos | Generación de variables predictoras | Validar variables temporales, estacionales, cíclicas, tendencia y precios. | Todas las variables predictoras están presentes y sin nulos críticos. | dataset_preparado.xlsx / auditoría |
| CP-05 | Preparación de datos | Control de productos estacionales | Verificar reglas de fanesca y colada morada fuera de temporada. | Se aplican las reglas estacionales definidas. | dataset_preparado.xlsx |
| CP-06 | División del dataset | Datasets por producto | Crear dataset independiente para almuerzo, sopa, fanesca y colada morada. | Se crean carpetas por producto con train, validación y prueba. | datasets_por_plato/ |
| CP-07 | División del dataset | Split cronológico | Validar que train, validación y prueba respeten el orden temporal. | No existe fuga de información temporal. | resumen_division_dataset.xlsx |
| CP-08 | Modelado | Prueba de modelos candidatos | Entrenar y evaluar varios modelos candidatos por producto. | Cada modelo genera métricas de validación. | comparacion_validacion |
| CP-09 | Modelado | Selección del mejor modelo | Seleccionar modelo por menor RMSE, usando MAE y R² como apoyo. | Existe un modelo seleccionado y justificado por producto. | seleccion_modelos / reporte_texto |
| CP-10 | Modelado | Evaluación final en prueba | Medir desempeño del modelo seleccionado en datos no usados para selección. | Se calculan MAE, RMSE, R², WAPE y MAPE. | metricas_prueba |
| CP-11 | Modelado | Evaluación estacional | Validar desempeño adicional en fanesca y colada morada durante temporada. | Se generan métricas estacionales cuando aplica. | metricas_estacionales |
| CP-12 | Persistencia | Generación de modelos PKL y configuración | Guardar modelos entrenados y configuración. | Existen modelo_*.pkl, config_entrenamiento.pkl y config_entrenamiento.json. | modelos_mejor_modelo/ |
| CP-13 | App Streamlit | Carga de modelos | Validar que la app cargue PKL y configuración. | El diagnóstico muestra todos los productos. | validacion_app_v5_multimodelo.ipynb |
| CP-14 | App Streamlit | Carga de archivo Excel/CSV | Validar carga de archivo histórico. | La app muestra vista previa y permite ejecutar predicción. | Vista previa en app |
| CP-15 | App Streamlit | Backtesting histórico | Comparar predicción contra valores reales de un archivo pasado. | La app calcula MAE, RMSE y R². | Evaluación de calidad por producto |
| CP-16 | App Streamlit | Predicción futura 6 meses | Generar calendario futuro y estimar demanda. | Se generan predicciones diarias, semanales y mensuales. | Tablas y gráficos futuros |
| CP-17 | App Streamlit | Consulta por periodo | Validar filtros por día, semana y mes. | Tablas y gráficos cambian según el nivel seleccionado. | Consulta por periodo |
| CP-18 | App Streamlit | Descarga de resultados | Exportar predicciones, agregados, métricas e insights. | Se descarga Excel final correctamente. | predicciones_demanda.xlsx |

## Selección de modelos

La selección de modelos se realizó mediante validación cronológica. Para cada producto se entrenaron varios modelos candidatos y se evaluaron en un conjunto de validación independiente. El criterio principal de selección fue el menor RMSE, debido a que esta métrica penaliza con mayor fuerza los errores grandes, lo cual es relevante en predicción de demanda porque un pico mal estimado puede afectar producción, compras o inventario. Como métricas complementarias se utilizaron MAE, para interpretar el error promedio en unidades; R², para revisar la capacidad explicativa del modelo; WAPE, para evaluar el error ponderado frente al total real; y MAPE, para obtener una referencia porcentual cuando existen valores reales distintos de cero.

El conjunto de prueba no fue utilizado para escoger el modelo, sino únicamente para evaluar el desempeño final del modelo seleccionado. Esta separación permite evitar fuga de información y proporciona una medición más realista del comportamiento esperado del modelo ante datos no utilizados durante el entrenamiento ni durante la selección.