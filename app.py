# ============================================================
# APP DE PREDICCIÓN DE DEMANDA DE PLATOS
# Streamlit + Plotly + modelos PKL + config_entrenamiento.pkl
# Versión compatible con selección automática de modelos por producto
# ============================================================

import io
import json
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Predicción de demanda",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_DIR = Path(__file__).parent
RUTA_MODELOS = APP_DIR / "modelos"
RUTA_CONFIG = RUTA_MODELOS / "config_entrenamiento.pkl"
RUTA_CONFIG_JSON = RUTA_MODELOS / "config_entrenamiento.json"

CONFIG_DEFAULT = {
    "version": "default",
    "modelo": "Seleccion_automatica_por_producto",
    "enfoque": "un_modelo_por_producto_con_seleccion_automatica",
    "productos": ["almuerzo", "sopa", "fanesca", "colada_morada"],
    "variables_predictoras": [
        "anio", "mes", "dia_mes", "dia_semana_num", "semana_anio",
        "es_fanesca_temporada", "es_colada_temporada",
        "es_inicio_mes", "es_quincena", "es_fin_mes",
        "es_lunes", "es_martes", "es_miercoles", "es_jueves", "es_viernes",
        "mes_sin", "mes_cos", "dia_semana_sin", "dia_semana_cos",
        "tendencia", "tendencia_log", "crecimiento_anual",
        "preciomenu", "preciosopa", "fanesca_precio", "coladamorada_precio"
    ],
    "fecha_min_modelo": "2023-01-02",
    "fecha_max_modelo": "2025-12-31",
    "temporada_colada_morada": {"inicio_mes": 10, "inicio_dia": 1, "fin_mes": 11, "fin_dia": 4},
    "temporada_fanesca": {"meses": [2, 3]},
    "base_ciclo_dia_semana": 5,
    "modelo_por_producto": {},
    "criterio_seleccion": {
        "principal": "Menor RMSE en validación",
        "secundarios": ["Menor MAE", "Mayor R²"]
    },
    "modelos_candidatos": [],
    "razon_por_producto": {},
    "parametros_modelo": {}
}

DIAS_ES = {
    0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves",
    4: "Viernes", 5: "Sábado", 6: "Domingo"
}

MESES_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}


# ============================================================
# ESTILOS
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #F8FAFC 0%, #EEF2FF 45%, #FFF7ED 100%);
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 28px 30px;
        border-radius: 26px;
        background:
            radial-gradient(circle at top left, rgba(249,115,22,0.22), transparent 28%),
            radial-gradient(circle at top right, rgba(37,99,235,0.22), transparent 28%),
            linear-gradient(135deg, #1E1B4B 0%, #312E81 45%, #6D28D9 100%);
        color: white;
        box-shadow: 0 20px 40px rgba(15,23,42,0.18);
        margin-bottom: 24px;
    }
    .hero h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.03em;
    }
    .hero p {
        color: rgba(255,255,255,0.86);
        font-size: 1.02rem;
        margin-top: 8px;
        margin-bottom: 0;
        max-width: 960px;
    }
    .kpi-card {
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(226,232,240,0.95);
        border-radius: 22px;
        padding: 20px 20px 18px 20px;
        box-shadow: 0 12px 30px rgba(15,23,42,0.08);
        min-height: 120px;
    }
    .kpi-title {
        color: #64748B;
        font-size: 0.83rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
    }
    .kpi-value {
        color: #0F172A;
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .kpi-sub {
        color: #64748B;
        font-size: 0.9rem;
        margin-top: 8px;
    }
    .section-title {
        color: #0F172A;
        font-size: 1.28rem;
        font-weight: 800;
        margin: 12px 0 6px 0;
    }
    .insight {
        background: linear-gradient(135deg, rgba(91,33,182,0.10), rgba(249,115,22,0.10));
        border-left: 5px solid #7C3AED;
        border-radius: 18px;
        padding: 14px 16px;
        margin: 8px 0;
        color: #1E1B4B;
        font-weight: 600;
    }
    .info-box {
        background: rgba(255,255,255,0.78);
        border: 1px solid rgba(226,232,240,0.95);
        border-radius: 18px;
        padding: 14px 16px;
        margin: 10px 0 16px 0;
        color: #334155;
        box-shadow: 0 8px 20px rgba(15,23,42,0.05);
        line-height: 1.55;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.84);
        border: 1px solid rgba(226,232,240,0.92);
        border-radius: 18px;
        padding: 14px;
        box-shadow: 0 10px 26px rgba(15,23,42,0.06);
    }
    .stButton > button {
        background: linear-gradient(135deg, #5B21B6 0%, #2563EB 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 0.75rem 1.15rem;
        font-weight: 800;
        box-shadow: 0 12px 26px rgba(37,99,235,0.22);
    }
    .stButton > button:hover {
        border: none;
        filter: brightness(1.06);
        color: white;
    }
    .stDownloadButton > button {
        border-radius: 16px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONFIG Y MODELOS
# ============================================================

@st.cache_resource(show_spinner=False)
def cargar_config():
    """
    Carga la configuración del entrenamiento.

    Compatibilidad V5:
    - config_entrenamiento.pkl puede contener la configuración base.
    - config_entrenamiento.json puede contener metadatos adicionales de selección automática:
      modelo_por_producto, criterio_seleccion, modelos_candidatos y razon_por_producto.

    Si ambos existen, se combinan y el JSON tiene prioridad para mostrar la selección
    real de modelos por producto.
    """
    config = CONFIG_DEFAULT.copy()

    if RUTA_CONFIG.exists():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                config_pkl = joblib.load(RUTA_CONFIG)
                if isinstance(config_pkl, dict):
                    config.update(config_pkl)
            except Exception as e:
                st.warning(f"No se pudo leer config_entrenamiento.pkl: {e}")

    if RUTA_CONFIG_JSON.exists():
        try:
            with open(RUTA_CONFIG_JSON, "r", encoding="utf-8") as f:
                config_json = json.load(f)
            if isinstance(config_json, dict):
                config.update(config_json)
        except Exception as e:
            st.warning(f"No se pudo leer config_entrenamiento.json: {e}")

    return config


CONFIG = cargar_config()
PRODUCTOS = CONFIG.get("productos", CONFIG_DEFAULT["productos"])
VARIABLES_CONFIG = CONFIG.get("variables_predictoras", CONFIG_DEFAULT["variables_predictoras"])
RANGO_HISTORICO_MIN = pd.Timestamp(CONFIG.get("fecha_min_modelo", "2023-01-02"))
RANGO_HISTORICO_MAX = pd.Timestamp(CONFIG.get("fecha_max_modelo", "2025-12-31"))
BASE_CICLO_DIA_SEMANA = int(CONFIG.get("base_ciclo_dia_semana", CONFIG_DEFAULT.get("base_ciclo_dia_semana", 5)))
MODELO_POR_PRODUCTO = CONFIG.get("modelo_por_producto", {}) or {}
RAZON_POR_PRODUCTO = CONFIG.get("razon_por_producto", {}) or {}


@st.cache_resource(show_spinner=False)
def cargar_modelos():
    modelos = {}
    faltantes = []

    for producto in PRODUCTOS:
        modelo_path = RUTA_MODELOS / f"modelo_{producto}.pkl"
        if not modelo_path.exists():
            faltantes.append(str(modelo_path))
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelos[producto] = joblib.load(modelo_path)

    if faltantes:
        raise FileNotFoundError("No se encontraron estos modelos: " + ", ".join(faltantes))

    return modelos


def obtener_variables_modelo(modelo):
    """
    Si el modelo fue entrenado con nombres de columnas, usamos esas columnas.
    Esto evita errores si config_entrenamiento.pkl tiene 26 variables pero los PKL cargados tienen 13.
    """
    if hasattr(modelo, "feature_names_in_"):
        return list(modelo.feature_names_in_)
    return list(VARIABLES_CONFIG)


def obtener_modelo_config_producto(producto):
    """
    Devuelve el nombre del modelo seleccionado para el producto según config_entrenamiento.json.
    Si no existe, intenta inferirlo desde la configuración general.
    """
    return MODELO_POR_PRODUCTO.get(producto, CONFIG.get("modelo", "No definido"))


def obtener_estimador_final(modelo):
    """
    Si el modelo es un Pipeline de scikit-learn, devuelve el último estimador.
    Si no, devuelve el objeto del modelo.
    """
    if hasattr(modelo, "steps") and getattr(modelo, "steps"):
        return modelo.steps[-1][1]
    return modelo


def obtener_tipo_modelo_real(modelo):
    """
    Devuelve un nombre entendible del modelo realmente cargado.
    Para Pipeline muestra: Pipeline → ElasticNet, por ejemplo.
    """
    if hasattr(modelo, "steps") and getattr(modelo, "steps"):
        return "Pipeline → " + type(modelo.steps[-1][1]).__name__
    return type(modelo).__name__


def diagnostico_modelos():
    try:
        modelos = cargar_modelos()
    except Exception:
        return pd.DataFrame()

    filas = []
    for producto, modelo in modelos.items():
        variables = obtener_variables_modelo(modelo)
        estimador_final = obtener_estimador_final(modelo)
        modelo_config = obtener_modelo_config_producto(producto)
        modelo_real = obtener_tipo_modelo_real(modelo)

        filas.append({
            "Producto": nombre_producto(producto),
            "Modelo seleccionado en config": modelo_config,
            "Tipo real cargado": modelo_real,
            "Coincide config vs PKL": "Sí" if modelo_config in modelo_real or modelo_real in modelo_config else "Revisar",
            "Variables usadas por el PKL": len(variables),
            "Variables en config": len(VARIABLES_CONFIG),
            "Árboles": getattr(estimador_final, "n_estimators", np.nan),
            "Max depth": getattr(estimador_final, "max_depth", np.nan),
            "Min samples leaf": getattr(estimador_final, "min_samples_leaf", np.nan),
        })
    return pd.DataFrame(filas)


# ============================================================
# LIMPIEZA Y PREPARACIÓN
# ============================================================

def normalize_col(col):
    col = str(col).strip()
    reemplazos = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
        "ñ": "n", "Ñ": "N"
    }
    for a, b in reemplazos.items():
        col = col.replace(a, b)
    col = re.sub(r"[^A-Za-z0-9]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col.lower()


def to_numeric_clean(series):
    s = series.astype(str).str.strip()
    s = s.replace(["", "nan", "None", "NULL", "NaN"], np.nan)
    s = s.str.replace("$", "", regex=False)
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def normalize_tipo_plato(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    s = (
        s.replace("á", "a").replace("é", "e").replace("í", "i")
        .replace("ó", "o").replace("ú", "u")
    )
    s = re.sub(r"\s+", "_", s)

    mapping = {
        "almuerzo": "almuerzo",
        "menu": "almuerzo",
        "menú": "almuerzo",
        "sopa": "sopa",
        "fanesca": "fanesca",
        "colada_morada": "colada_morada",
        "coladamorada": "colada_morada",
        "colada": "colada_morada",
        "colada_morada_": "colada_morada",
        "": np.nan
    }
    return mapping.get(s, s if s in PRODUCTOS else np.nan)


def normalize_diasemana(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    s = (
        s.replace("á", "a").replace("é", "e").replace("í", "i")
        .replace("ó", "o").replace("ú", "u")
    )
    mapa = {
        "lun": "Lun", "lunes": "Lun",
        "mar": "Mar", "martes": "Mar",
        "mie": "Mie", "miercoles": "Mie", "miércoles": "Mie",
        "jue": "Jue", "jueves": "Jue",
        "vie": "Vie", "viernes": "Vie",
        "sab": "Sab", "sabado": "Sab", "sábado": "Sab",
        "dom": "Dom", "domingo": "Dom"
    }
    return mapa.get(s, np.nan)


def weekday_3letters(fecha):
    dias = {0: "Lun", 1: "Mar", 2: "Mie", 3: "Jue", 4: "Vie", 5: "Sab", 6: "Dom"}
    return dias.get(fecha.weekday(), np.nan)


def get_precio_menu(fecha):
    if fecha.year == 2023:
        return 4.0
    if fecha.year == 2024:
        return 4.0 if fecha.month <= 8 else 4.5
    if fecha.year == 2025:
        return 4.5 if fecha.month <= 7 else 5.0
    if fecha.year >= 2026:
        return 5.0
    return np.nan


def get_precio_sopa(fecha):
    if fecha.year == 2023:
        return 1.5
    if fecha.year == 2024:
        return 1.5 if fecha.month <= 8 else 1.8
    if fecha.year == 2025:
        return 1.8 if fecha.month <= 7 else 2.0
    if fecha.year >= 2026:
        return 2.0
    return np.nan


def get_precio_fanesca(fecha):
    if fecha.year == 2023:
        return 7.0
    if fecha.year == 2024:
        return 8.5
    if fecha.year >= 2025:
        return 10.0
    return np.nan


def get_precio_colada(fecha):
    if fecha.year == 2023:
        return 2.0
    if fecha.year == 2024:
        return 2.8
    if fecha.year >= 2025:
        return 3.5
    return np.nan


def completar_precio_por_fecha(df, columna, funcion_precio):
    df[columna] = pd.to_numeric(df[columna], errors="coerce")
    mask = df[columna].isna()
    if mask.any():
        df.loc[mask, columna] = df.loc[mask, "fecha"].map(funcion_precio).astype(float)
    return df


def resolver_aliases(df):
    aliases = {
        "fecha": ["fecha", "date", "dia", "día"],
        "diasemana": ["diasemana", "dia_semana", "dia_de_la_semana", "día_semana"],
        "tipo_plato": ["tipo_plato", "tipoplato", "tipo_de_plato", "producto", "plato"],
        "clientes": ["clientes", "cantidad_clientes", "cantidad", "platos_vendidos", "unidades"],
        "facturacion": ["facturacion", "facturacion_diaria", "venta", "ventas", "ingreso_diario", "ingresos"],
        "preciomenu": ["preciomenu", "precio_menu", "menu_precio", "precio_almuerzo"],
        "preciosopa": ["preciosopa", "precio_sopa", "sopa_precio"],
        "fanesca_precio": ["fanesca_precio", "precio_fanesca"],
        "coladamorada_precio": [
            "coladamorada_precio", "precio_coladamorada", "colada_morada_precio",
            "precio_colada_morada", "precio_colada"
        ]
    }

    resolved = {}
    for target, options in aliases.items():
        for opt in options:
            if opt in df.columns:
                resolved[target] = opt
                break
    return resolved


def validar_estructura(df):
    errores = []
    advertencias = []

    if df.empty:
        errores.append("El archivo cargado está vacío.")

    df_tmp = df.copy()
    df_tmp.columns = [normalize_col(c) for c in df_tmp.columns]
    resolved = resolver_aliases(df_tmp)

    obligatorias = {
        "fecha": "Fecha",
        "tipo_plato": "Tipo_plato",
        "clientes": "Clientes"
    }

    for campo, nombre_visible in obligatorias.items():
        if campo not in resolved:
            errores.append(f"El archivo no contiene la columna obligatoria: {nombre_visible}.")

    tiene_precio_o_facturacion = any(
        campo in resolved
        for campo in ["facturacion", "preciomenu", "preciosopa", "fanesca_precio", "coladamorada_precio"]
    )

    if not tiene_precio_o_facturacion:
        advertencias.append(
            "El archivo no contiene columnas de precio o facturación. "
            "La app completará precios por defecto según las reglas del modelo."
        )

    return errores, advertencias


def leer_archivo(uploaded_file):
    nombre = uploaded_file.name.lower()
    if nombre.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin-1")
    if nombre.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Formato no permitido. Sube un archivo .csv o .xlsx.")


def limpiar_datos(df):
    df = df.copy()
    df.columns = [normalize_col(c) for c in df.columns]
    resolved = resolver_aliases(df)

    if "fecha" not in resolved:
        raise ValueError("El archivo no contiene la columna Fecha.")
    if "tipo_plato" not in resolved:
        raise ValueError("El archivo no contiene la columna Tipo_plato.")
    if "clientes" not in resolved:
        raise ValueError("El archivo no contiene la columna Clientes.")

    for target, source in resolved.items():
        df[target] = df[source]

    for col in [
        "diasemana", "tipo_plato", "clientes", "facturacion",
        "preciomenu", "preciosopa", "fanesca_precio", "coladamorada_precio"
    ]:
        if col not in df.columns:
            df[col] = np.nan

    df = df[[
        "fecha", "diasemana", "tipo_plato", "clientes", "facturacion",
        "preciomenu", "preciosopa", "fanesca_precio", "coladamorada_precio"
    ]].copy()

    total_filas = len(df)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=False)
    fechas_invalidas = int(df["fecha"].isna().sum())

    if fechas_invalidas == total_filas:
        raise ValueError("La columna Fecha no contiene valores válidos.")
    if fechas_invalidas > 0:
        st.warning(f"Se encontraron {fechas_invalidas} fechas inválidas. Esas filas serán excluidas.")

    df = df.dropna(subset=["fecha"]).copy()

    for col in ["clientes", "facturacion", "preciomenu", "preciosopa", "fanesca_precio", "coladamorada_precio"]:
        df[col] = to_numeric_clean(df[col])

    if (df["clientes"].dropna() < 0).any():
        raise ValueError("La columna Clientes contiene valores negativos.")

    for col in ["facturacion", "preciomenu", "preciosopa", "fanesca_precio", "coladamorada_precio"]:
        if (df[col].dropna() < 0).any():
            st.warning(f"La columna {col} contiene valores negativos. Revisa el archivo de entrada.")

    df["tipo_plato"] = df["tipo_plato"].apply(normalize_tipo_plato)
    tipos_invalidos = int(df["tipo_plato"].isna().sum())

    if tipos_invalidos > 0:
        st.warning(f"Se encontraron {tipos_invalidos} filas con Tipo_plato inválido. Esas filas serán excluidas.")

    df["diasemana"] = df["diasemana"].apply(normalize_diasemana)
    mask_dia_invalido = ~df["diasemana"].isin(["Lun", "Mar", "Mie", "Jue", "Vie"])
    df.loc[mask_dia_invalido, "diasemana"] = df.loc[mask_dia_invalido, "fecha"].apply(weekday_3letters)

    df = completar_precio_por_fecha(df, "preciomenu", get_precio_menu)
    df = completar_precio_por_fecha(df, "preciosopa", get_precio_sopa)
    df = completar_precio_por_fecha(df, "fanesca_precio", get_precio_fanesca)
    df = completar_precio_por_fecha(df, "coladamorada_precio", get_precio_colada)

    df = df[df["tipo_plato"].notna()].copy()

    if df.empty:
        raise ValueError("No quedaron registros válidos después de la limpieza.")

    df["clientes"] = df["clientes"].fillna(0).round(0)
    df = df.drop_duplicates().sort_values(["fecha", "tipo_plato"]).reset_index(drop=True)

    return df


def dentro_temporada_colada(fecha):
    temp = CONFIG.get("temporada_colada_morada", CONFIG_DEFAULT["temporada_colada_morada"])
    inicio = pd.Timestamp(year=fecha.year, month=int(temp["inicio_mes"]), day=int(temp["inicio_dia"]))
    fin = pd.Timestamp(year=fecha.year, month=int(temp["fin_mes"]), day=int(temp["fin_dia"]))
    return int(inicio <= fecha <= fin)


def es_fanesca_temporada(fecha):
    temp = CONFIG.get("temporada_fanesca", CONFIG_DEFAULT["temporada_fanesca"])
    meses = temp.get("meses", [3, 4])
    return int(fecha.month in meses)


def agregar_variables_predictoras(df_model):
    df_model = df_model.copy()
    df_model["fecha"] = pd.to_datetime(df_model["fecha"])

    df_model["anio"] = df_model["fecha"].dt.year
    df_model["mes"] = df_model["fecha"].dt.month
    df_model["dia_mes"] = df_model["fecha"].dt.day
    df_model["dia_semana_num"] = df_model["fecha"].dt.weekday
    df_model["semana_anio"] = df_model["fecha"].dt.isocalendar().week.astype(int)

    df_model["es_fanesca_temporada"] = df_model["fecha"].apply(es_fanesca_temporada).astype(int)
    df_model["es_colada_temporada"] = df_model["fecha"].apply(dentro_temporada_colada).astype(int)

    df_model["es_inicio_mes"] = (df_model["dia_mes"] <= 5).astype(int)
    df_model["es_quincena"] = df_model["dia_mes"].between(13, 17).astype(int)
    df_model["es_fin_mes"] = (df_model["dia_mes"] >= 25).astype(int)

    df_model["es_lunes"] = (df_model["dia_semana_num"] == 0).astype(int)
    df_model["es_martes"] = (df_model["dia_semana_num"] == 1).astype(int)
    df_model["es_miercoles"] = (df_model["dia_semana_num"] == 2).astype(int)
    df_model["es_jueves"] = (df_model["dia_semana_num"] == 3).astype(int)
    df_model["es_viernes"] = (df_model["dia_semana_num"] == 4).astype(int)

    df_model["mes_sin"] = np.sin(2 * np.pi * df_model["mes"] / 12)
    df_model["mes_cos"] = np.cos(2 * np.pi * df_model["mes"] / 12)
    df_model["dia_semana_sin"] = np.sin(2 * np.pi * df_model["dia_semana_num"] / BASE_CICLO_DIA_SEMANA)
    df_model["dia_semana_cos"] = np.cos(2 * np.pi * df_model["dia_semana_num"] / BASE_CICLO_DIA_SEMANA)

    fecha_base = RANGO_HISTORICO_MIN
    df_model["tendencia"] = (df_model["fecha"] - fecha_base).dt.days
    df_model["tendencia"] = df_model["tendencia"].clip(lower=0)
    df_model["tendencia_log"] = np.log1p(df_model["tendencia"])
    df_model["crecimiento_anual"] = df_model["anio"] - fecha_base.year

    return df_model


def preparar_datos(df):
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"]).copy()

    ventas_wide = df.pivot_table(
        index="fecha",
        columns="tipo_plato",
        values="clientes",
        aggfunc="sum",
        fill_value=0
    ).reset_index()

    for col in PRODUCTOS:
        if col not in ventas_wide.columns:
            ventas_wide[col] = 0

    precios = df.groupby("fecha", as_index=False)[[
        "preciomenu", "preciosopa", "fanesca_precio", "coladamorada_precio"
    ]].max()

    df_model = ventas_wide.merge(precios, on="fecha", how="left")
    df_model = df_model.sort_values("fecha").reset_index(drop=True)
    df_model = agregar_variables_predictoras(df_model)

    if "fanesca" in df_model.columns:
        df_model.loc[df_model["es_fanesca_temporada"] == 0, "fanesca"] = 0
    if "colada_morada" in df_model.columns:
        df_model.loc[df_model["es_colada_temporada"] == 0, "colada_morada"] = 0

    # Aseguramos todas las variables de config aunque los PKL actuales usen menos.
    for var in VARIABLES_CONFIG:
        if var not in df_model.columns:
            df_model[var] = 0

    return df_model[["fecha"] + VARIABLES_CONFIG + PRODUCTOS].copy()


def generar_datos_futuros(fecha_inicio, meses, preciomenu, preciosopa, fanesca_precio, coladamorada_precio):
    fecha_inicio = pd.to_datetime(fecha_inicio)
    fecha_fin = fecha_inicio + pd.DateOffset(months=int(meses))

    fechas = pd.date_range(fecha_inicio, fecha_fin, freq="B")  # lunes a viernes

    df_model = pd.DataFrame({
        "fecha": fechas,
        "preciomenu": float(preciomenu),
        "preciosopa": float(preciosopa),
        "fanesca_precio": float(fanesca_precio),
        "coladamorada_precio": float(coladamorada_precio),
    })

    for producto in PRODUCTOS:
        df_model[producto] = 0

    df_model = agregar_variables_predictoras(df_model)

    for var in VARIABLES_CONFIG:
        if var not in df_model.columns:
            df_model[var] = 0

    return df_model[["fecha"] + VARIABLES_CONFIG + PRODUCTOS].copy()


def generar_predicciones(df_preparado):
    modelos = cargar_modelos()
    df_pred_wide = df_preparado[["fecha"]].copy()

    diagnosticos = []

    for producto in PRODUCTOS:
        modelo = modelos[producto]
        variables_modelo = obtener_variables_modelo(modelo)

        faltantes = [v for v in variables_modelo if v not in df_preparado.columns]
        if faltantes:
            raise ValueError(
                f"Faltan variables para el modelo {producto}: {', '.join(faltantes)}"
            )

        X = df_preparado[variables_modelo].copy().fillna(0)
        pred = modelo.predict(X)
        pred = np.maximum(pred, 0)

        # Reglas de negocio para productos estacionales.
        # Aunque el modelo pueda estimar valores bajos fuera de temporada,
        # la salida operativa se fuerza a 0 cuando la fecha no corresponde.
        if producto == "fanesca" and "es_fanesca_temporada" in df_preparado.columns:
            pred = np.where(df_preparado["es_fanesca_temporada"].to_numpy() == 1, pred, 0)
        if producto == "colada_morada" and "es_colada_temporada" in df_preparado.columns:
            pred = np.where(df_preparado["es_colada_temporada"].to_numpy() == 1, pred, 0)

        df_pred_wide[producto] = np.round(pred).astype(int)

        diagnosticos.append({
            "Producto": nombre_producto(producto),
            "Modelo seleccionado": obtener_modelo_config_producto(producto),
            "Tipo real cargado": obtener_tipo_modelo_real(modelo),
            "Variables usadas": len(variables_modelo),
            "Variables": ", ".join(variables_modelo),
            "Explicación de selección": RAZON_POR_PRODUCTO.get(producto, "No disponible en la configuración.")
        })

    df_largo = df_pred_wide.melt(
        id_vars=["fecha"],
        value_vars=PRODUCTOS,
        var_name="tipo_plato",
        value_name="cantidad_predicha"
    )

    df_largo["fecha"] = pd.to_datetime(df_largo["fecha"])
    df_largo["anio"] = df_largo["fecha"].dt.year
    df_largo["mes"] = df_largo["fecha"].dt.month
    df_largo["mes_nombre"] = df_largo["mes"].map(MESES_ES)
    df_largo["semana_anio"] = df_largo["fecha"].dt.isocalendar().week.astype(int)
    df_largo["dia_semana_num"] = df_largo["fecha"].dt.weekday
    df_largo["dia_semana"] = df_largo["dia_semana_num"].map(DIAS_ES)

    prediccion_diaria = df_largo.copy()

    prediccion_semanal = (
        df_largo
        .groupby(["anio", "semana_anio", "tipo_plato"], as_index=False)["cantidad_predicha"]
        .sum()
    )

    prediccion_mensual = (
        df_largo
        .groupby(["anio", "mes", "mes_nombre", "tipo_plato"], as_index=False)["cantidad_predicha"]
        .sum()
    )

    return df_pred_wide, prediccion_diaria, prediccion_semanal, prediccion_mensual, pd.DataFrame(diagnosticos)


# ============================================================
# MÉTRICAS, INSIGHTS, GRÁFICOS Y EXPORTACIÓN
# ============================================================

def nombre_producto(texto):
    return {
        "almuerzo": "Almuerzo",
        "sopa": "Sopa",
        "fanesca": "Fanesca",
        "colada_morada": "Colada morada"
    }.get(texto, str(texto).replace("_", " ").title())


def formato_entero(valor):
    try:
        return f"{int(round(float(valor))):,}".replace(",", ".")
    except Exception:
        return "0"


def formato_decimal(valor, decimales=2):
    try:
        if pd.isna(valor):
            return "N/A"
        return f"{float(valor):,.{decimales}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "N/A"


def aplicar_estilo_barras(fig, texttemplate="%{text}"):
    """
    Centra las etiquetas dentro de las barras para que los gráficos sean más legibles.
    """
    fig.update_traces(
        texttemplate=texttemplate,
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=12, color="white"),
        cliponaxis=False
    )
    fig.update_layout(
        uniformtext_minsize=10,
        uniformtext_mode="show"
    )
    return fig


def pivot_predicciones(df_largo):
    tabla = (
        df_largo.pivot_table(
            index="fecha",
            columns="tipo_plato",
            values="cantidad_predicha",
            aggfunc="sum",
            fill_value=0
        )
        .reset_index()
        .sort_values("fecha")
    )

    for producto in PRODUCTOS:
        if producto not in tabla.columns:
            tabla[producto] = 0

    tabla = tabla[["fecha"] + PRODUCTOS]
    tabla = tabla.rename(columns={
        "fecha": "Fecha",
        "almuerzo": "Almuerzo",
        "sopa": "Sopa",
        "fanesca": "Fanesca",
        "colada_morada": "Colada morada"
    })
    return tabla


def generar_insights(df_largo):
    if df_largo.empty:
        return {"producto_mayor": "Sin datos", "dia_mayor": "Sin datos", "mes_mayor": "Sin datos", "total": 0, "fecha_pico": "Sin datos"}

    total_por_producto = df_largo.groupby("tipo_plato")["cantidad_predicha"].sum()
    producto_mayor = nombre_producto(total_por_producto.idxmax())

    total_por_dia_semana = df_largo.groupby("dia_semana")["cantidad_predicha"].sum()
    dia_mayor = total_por_dia_semana.idxmax()

    total_por_mes = df_largo.groupby("mes_nombre")["cantidad_predicha"].sum()
    mes_mayor = total_por_mes.idxmax()

    total_por_fecha = df_largo.groupby("fecha")["cantidad_predicha"].sum()
    fecha_pico = total_por_fecha.idxmax().strftime("%Y-%m-%d")

    total = int(df_largo["cantidad_predicha"].sum())

    return {"producto_mayor": producto_mayor, "dia_mayor": dia_mayor, "mes_mayor": mes_mayor, "total": total, "fecha_pico": fecha_pico}


def calcular_metricas_modelos(df_preparado, pred_wide):
    """
    Calcula métricas de evaluación por producto usando los valores reales del archivo cargado
    frente a las predicciones generadas por cada modelo PKL.

    Métricas principales:
    - MAE: error absoluto promedio en unidades.
    - RMSE: error cuadrático medio, penaliza más los errores grandes.
    - R²: capacidad explicativa del modelo. Mayor es mejor.
    """
    filas_metricas = []

    base = df_preparado[["fecha"] + PRODUCTOS].copy()
    pred = pred_wide[["fecha"] + PRODUCTOS].copy()

    for producto in PRODUCTOS:
        df_eval = base[["fecha", producto]].merge(
            pred[["fecha", producto]],
            on="fecha",
            how="inner",
            suffixes=("_real", "_predicho")
        )

        df_eval = df_eval.rename(columns={
            f"{producto}_real": "real",
            f"{producto}_predicho": "prediccion"
        })
        df_eval["real"] = pd.to_numeric(df_eval["real"], errors="coerce").fillna(0)
        df_eval["prediccion"] = pd.to_numeric(df_eval["prediccion"], errors="coerce").fillna(0)

        y_true = df_eval["real"].to_numpy(dtype=float)
        y_pred = df_eval["prediccion"].to_numpy(dtype=float)
        n = len(df_eval)

        mae = float(np.mean(np.abs(y_true - y_pred))) if n else np.nan
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if n else np.nan

        ss_res = float(np.sum((y_true - y_pred) ** 2)) if n else np.nan
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) if n else np.nan
        r2 = 1 - (ss_res / ss_tot) if n and ss_tot > 0 else np.nan

        filas_metricas.append({
            "modelo": producto,
            "Modelo": nombre_producto(producto),
            "Registros evaluados": n,
            "Total real": int(round(float(np.sum(y_true)))) if n else 0,
            "Total predicho": int(round(float(np.sum(y_pred)))) if n else 0,
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2,
        })

    metricas = pd.DataFrame(filas_metricas)

    # El ranking principal usa RMSE porque penaliza con más fuerza los errores grandes.
    # Para demanda diaria, esto ayuda a identificar productos con picos mal estimados.
    metricas["ranking_valor"] = metricas["RMSE"]
    metricas = metricas.sort_values("ranking_valor", ascending=True).reset_index(drop=True)
    metricas["Ranking"] = np.arange(1, len(metricas) + 1)

    return metricas


def formatear_metricas(metricas):
    columnas = [
        "Ranking", "Modelo", "Registros evaluados",
        "Total real", "Total predicho",
        "MAE", "RMSE", "R²"
    ]

    df = metricas[columnas].copy()

    for col in ["MAE", "RMSE", "R²"]:
        df[col] = df[col].apply(lambda x: "N/A" if pd.isna(x) else round(float(x), 3))

    return df


def grafico_diario(df_largo):
    df_plot = df_largo.copy().sort_values("fecha")
    df_plot["Producto"] = df_plot["tipo_plato"].map(nombre_producto)
    fig = px.line(
        df_plot,
        x="fecha",
        y="cantidad_predicha",
        color="Producto",
        markers=True,
        labels={"fecha": "Fecha", "cantidad_predicha": "Cantidad predicha", "Producto": "Producto"},
        title="Predicción diaria por producto"
    )
    fig.update_layout(height=430, hovermode="x unified", title_font_size=20, legend_title_text="Producto", margin=dict(l=20, r=20, t=60, b=20))
    return fig


def grafico_semanal(df_largo):
    df_plot = df_largo.groupby(["anio", "semana_anio", "tipo_plato"], as_index=False)["cantidad_predicha"].sum()
    df_plot["periodo"] = df_plot["anio"].astype(str) + " - S" + df_plot["semana_anio"].astype(str).str.zfill(2)
    df_plot["Producto"] = df_plot["tipo_plato"].map(nombre_producto)
    df_plot["Etiqueta"] = df_plot["cantidad_predicha"].apply(formato_entero)

    fig = px.bar(
        df_plot,
        x="periodo",
        y="cantidad_predicha",
        color="Producto",
        barmode="group",
        text="Etiqueta",
        labels={"periodo": "Semana", "cantidad_predicha": "Cantidad predicha", "Producto": "Producto"},
        title="Consumo semanal estimado"
    )
    fig.update_layout(height=430, title_font_size=20, legend_title_text="Producto", margin=dict(l=20, r=20, t=60, b=20))
    aplicar_estilo_barras(fig)
    return fig


def grafico_mensual(df_largo):
    df_plot = df_largo.groupby(["anio", "mes", "mes_nombre", "tipo_plato"], as_index=False)["cantidad_predicha"].sum()
    df_plot["periodo"] = df_plot["anio"].astype(str) + " - " + df_plot["mes"].astype(str).str.zfill(2)
    df_plot["Producto"] = df_plot["tipo_plato"].map(nombre_producto)
    df_plot["Etiqueta"] = df_plot["cantidad_predicha"].apply(formato_entero)

    fig = px.bar(
        df_plot,
        x="periodo",
        y="cantidad_predicha",
        color="Producto",
        barmode="group",
        text="Etiqueta",
        labels={"periodo": "Mes", "cantidad_predicha": "Cantidad predicha", "Producto": "Producto"},
        title="Consumo mensual estimado"
    )
    fig.update_layout(height=430, title_font_size=20, legend_title_text="Producto", margin=dict(l=20, r=20, t=60, b=20))
    aplicar_estilo_barras(fig)
    return fig


def grafico_mae_rmse_por_producto(metricas):
    df_plot = metricas[["Modelo", "MAE", "RMSE"]].copy()
    df_plot = df_plot.melt(
        id_vars="Modelo",
        value_vars=["MAE", "RMSE"],
        var_name="Métrica",
        value_name="Valor"
    )
    df_plot["Etiqueta"] = df_plot["Valor"].apply(lambda x: formato_decimal(x, 2))

    fig = px.bar(
        df_plot,
        x="Modelo",
        y="Valor",
        color="Métrica",
        barmode="group",
        text="Etiqueta",
        title="Errores por producto: MAE y RMSE",
        labels={
            "Modelo": "Producto",
            "Valor": "Error en unidades",
            "Métrica": "Métrica"
        }
    )
    fig.update_layout(
        height=420,
        title_font_size=20,
        legend_title_text="Métrica",
        margin=dict(l=20, r=20, t=60, b=20)
    )
    aplicar_estilo_barras(fig)
    return fig


def grafico_r2_por_producto(metricas):
    df_plot = metricas[["Modelo", "R²"]].copy()
    df_plot["R²_plot"] = df_plot["R²"].fillna(0)
    df_plot["Etiqueta"] = df_plot["R²"].apply(lambda x: "N/A" if pd.isna(x) else formato_decimal(x, 3))

    fig = px.bar(
        df_plot,
        x="Modelo",
        y="R²_plot",
        text="Etiqueta",
        title="R² por producto: mayor es mejor",
        labels={
            "Modelo": "Producto",
            "R²_plot": "R²"
        }
    )
    fig.update_layout(
        height=420,
        title_font_size=20,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    aplicar_estilo_barras(fig)
    return fig


def grafico_totales_real_predicho(metricas):
    df_plot = metricas[["Modelo", "Total real", "Total predicho"]].copy()
    df_plot = df_plot.melt(id_vars="Modelo", value_vars=["Total real", "Total predicho"], var_name="Serie", value_name="Cantidad")
    df_plot["Etiqueta"] = df_plot["Cantidad"].apply(formato_entero)

    fig = px.bar(
        df_plot,
        x="Modelo",
        y="Cantidad",
        color="Serie",
        barmode="group",
        text="Etiqueta",
        title="Total real vs total predicho por producto",
        labels={"Modelo": "Producto", "Cantidad": "Cantidad", "Serie": "Serie"}
    )
    fig.update_layout(height=420, title_font_size=20, legend_title_text="Serie", margin=dict(l=20, r=20, t=60, b=20))
    aplicar_estilo_barras(fig)
    return fig


def crear_excel_descargable(pred_wide, diaria, semanal, mensual, insights, metricas=None, diagnostico=None):
    buffer = io.BytesIO()

    df_insights = pd.DataFrame([
        {"Insight": "Producto con mayor demanda", "Valor": insights["producto_mayor"]},
        {"Insight": "Día con mayor consumo", "Valor": insights["dia_mayor"]},
        {"Insight": "Mes con mayor consumo", "Valor": insights["mes_mayor"]},
        {"Insight": "Fecha pico", "Valor": insights["fecha_pico"]},
        {"Insight": "Total estimado", "Valor": insights["total"]},
    ])

    pred_wide_export = pred_wide.rename(columns={
        "fecha": "Fecha", "almuerzo": "Almuerzo", "sopa": "Sopa",
        "fanesca": "Fanesca", "colada_morada": "Colada morada"
    })

    metricas_export = formatear_metricas(metricas) if metricas is not None and not metricas.empty else None

    with pd.ExcelWriter(buffer, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        pred_wide_export.to_excel(writer, sheet_name="predicciones", index=False)
        diaria.to_excel(writer, sheet_name="detalle_diario_largo", index=False)
        semanal.to_excel(writer, sheet_name="agregado_semanal", index=False)
        mensual.to_excel(writer, sheet_name="agregado_mensual", index=False)
        df_insights.to_excel(writer, sheet_name="insights", index=False)

        if metricas_export is not None:
            metricas_export.to_excel(writer, sheet_name="metricas_productos", index=False)
        if diagnostico is not None and not diagnostico.empty:
            diagnostico.to_excel(writer, sheet_name="diagnostico_modelos", index=False)

        workbook = writer.book
        header_format = workbook.add_format({"bold": True, "font_color": "white", "bg_color": "#5B21B6", "border": 1})

        for sheet_name, df_sheet in writer.sheets.items():
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes(1, 0)

        # Ajustar encabezados
        hojas = {
            "predicciones": pred_wide_export,
            "detalle_diario_largo": diaria,
            "agregado_semanal": semanal,
            "agregado_mensual": mensual,
            "insights": df_insights,
        }
        if metricas_export is not None:
            hojas["metricas_productos"] = metricas_export
        if diagnostico is not None and not diagnostico.empty:
            hojas["diagnostico_modelos"] = diagnostico

        for sheet_name, df_sheet in hojas.items():
            worksheet = writer.sheets[sheet_name]
            for col_num, value in enumerate(df_sheet.columns.values):
                worksheet.write(0, col_num, value, header_format)
                worksheet.set_column(col_num, col_num, min(max(len(str(value)) + 4, 14), 42))

    buffer.seek(0)
    return buffer


def mostrar_kpi_card(titulo, valor, subtitulo=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{titulo}</div>
            <div class="kpi-value">{valor}</div>
            <div class="kpi-sub">{subtitulo}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def mostrar_insight(texto):
    st.markdown(f'<div class="insight">{texto}</div>', unsafe_allow_html=True)


def mostrar_info(texto):
    st.markdown(f'<div class="info-box">{texto}</div>', unsafe_allow_html=True)


# ============================================================
# INTERFAZ
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>Predicción de demanda de platos</h1>
        <p>
            Carga un archivo Excel o CSV para evaluar datos históricos o genera una predicción futura.
            La app usa modelos PKL y el archivo config_entrenamiento.pkl para estimar demanda por producto,
            revisar resultados por periodo y evaluar la calidad del modelo con MAE, RMSE y R². Cada producto puede usar un algoritmo distinto según la selección automática del entrenamiento.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("### Panel de control")
    st.caption(
        "Desde este panel defines el flujo de trabajo de la app. "
        "Cada opción cambia la forma en que se preparan los datos y el tipo de resultados que se muestran."
    )

    modo = st.radio(
        "Modo de ejecución",
        ["Archivo cargado", "Predicción futura 6 meses"],
        help=(
            "Archivo cargado: procesa un Excel/CSV histórico y permite calcular métricas si existen valores reales. "
            "Predicción futura: genera fechas futuras y estima demanda sin comparar contra valores reales."
        )
    )

    if modo == "Archivo cargado":
        st.info(
            "Este modo lee el archivo cargado, valida columnas, limpia datos, genera variables predictoras, "
            "aplica los modelos y calcula MAE, RMSE y R² cuando hay valores reales."
        )
    else:
        st.info(
            "Este modo crea un calendario futuro de lunes a viernes, usa los precios configurados y genera "
            "una estimación de demanda. No calcula métricas porque no existen valores reales futuros."
        )

    st.markdown("### Configuración detectada")
    st.caption("Información tomada de config_entrenamiento.pkl o de la configuración por defecto.")
    st.caption(f"Versión config: {CONFIG.get('version', 'Sin config')}")
    st.caption(f"Modelo: {CONFIG.get('modelo', 'No definido')}")
    st.caption(f"Rango modelo: {RANGO_HISTORICO_MIN.date()} a {RANGO_HISTORICO_MAX.date()}")
    st.caption(f"Base ciclo día semana: {BASE_CICLO_DIA_SEMANA}")

    if MODELO_POR_PRODUCTO:
        with st.expander("Modelos seleccionados por producto"):
            for prod, modelo_sel in MODELO_POR_PRODUCTO.items():
                st.write(f"**{nombre_producto(prod)}:** {modelo_sel}")

    uploaded_file = None
    ejecutar = False

    if modo == "Archivo cargado":
        uploaded_file = st.file_uploader(
            "Sube tu archivo CSV o Excel",
            type=["csv", "xlsx"],
            help="El archivo debe incluir como mínimo Fecha, Tipo_plato y Clientes."
        )

        st.markdown("### Estructura mínima")
        st.caption("Columnas requeridas:")
        st.markdown("- Fecha\n- Tipo_plato\n- Clientes")
        st.caption("Recomendadas:")
        st.markdown("- Facturación\n- Precio menú / sopa / fanesca / colada")

        ejecutar = st.button("Generar predicción", use_container_width=True)

    else:
        fecha_inicio_futuro = st.date_input(
            "Fecha inicial futura",
            value=(RANGO_HISTORICO_MAX + pd.Timedelta(days=1)).date()
        )

        meses_futuro = st.slider("Meses a predecir", min_value=1, max_value=12, value=6)

        st.markdown("### Precios futuros")
        precio_menu_futuro = st.number_input("Precio menú / almuerzo", min_value=0.0, value=float(get_precio_menu(pd.Timestamp(fecha_inicio_futuro))), step=0.1)
        precio_sopa_futuro = st.number_input("Precio sopa", min_value=0.0, value=float(get_precio_sopa(pd.Timestamp(fecha_inicio_futuro))), step=0.1)
        precio_fanesca_futuro = st.number_input("Precio fanesca", min_value=0.0, value=float(get_precio_fanesca(pd.Timestamp(fecha_inicio_futuro))), step=0.1)
        precio_colada_futuro = st.number_input("Precio colada morada", min_value=0.0, value=float(get_precio_colada(pd.Timestamp(fecha_inicio_futuro))), step=0.1)

        ejecutar = st.button("Generar predicción futura", use_container_width=True)


st.markdown('<div class="section-title">Diagnóstico de modelos cargados</div>', unsafe_allow_html=True)
mostrar_info(
    "Esta sección valida los modelos disponibles en la carpeta <b>modelos</b>. "
    "Muestra el tipo de modelo, la cantidad de variables usadas por cada PKL y si existe diferencia "
    "frente al archivo de configuración. Sirve para detectar incompatibilidades antes de ejecutar predicciones."
)
diag_general = diagnostico_modelos()
if not diag_general.empty:
    st.dataframe(diag_general, use_container_width=True, hide_index=True)

    if (diag_general["Variables usadas por el PKL"] != diag_general["Variables en config"]).any():
        st.warning(
            "La configuración tiene un número de variables diferente al que usan algunos modelos PKL. "
            "La app prioriza las variables guardadas dentro de cada modelo para evitar errores de predicción."
        )

    if "Coincide config vs PKL" in diag_general.columns and (diag_general["Coincide config vs PKL"] == "Revisar").any():
        st.warning(
            "Al menos un modelo cargado no parece coincidir con el nombre registrado en config_entrenamiento.json. "
            "Esto puede ocurrir si se reemplazó el JSON pero no se copiaron los PKL nuevos a la carpeta modelos."
        )
else:
    st.info("Aún no se pudo leer el diagnóstico de modelos. Verifica la carpeta modelos.")


if modo == "Archivo cargado":
    if uploaded_file is None:
        col1, col2 = st.columns([1.2, 0.8])
        with col1:
            st.markdown('<div class="section-title">¿Cómo usar la app?</div>', unsafe_allow_html=True)
            st.markdown(
                """
                1. Sube un archivo `.csv` o `.xlsx`.
                2. Revisa la vista previa.
                3. Haz clic en **Generar predicción**.
                4. Consulta resultados por día, semana o mes.
                5. Descarga el Excel final.
                """
            )
        with col2:
            st.markdown('<div class="section-title">Productos del modelo</div>', unsafe_allow_html=True)
            st.markdown("- Almuerzo  \n- Sopa  \n- Fanesca  \n- Colada morada")
        st.info("Carga un archivo desde el panel lateral para iniciar.")
        st.stop()

    try:
        df_original = leer_archivo(uploaded_file)
    except Exception as e:
        st.error(str(e))
        st.stop()

    errores, advertencias = validar_estructura(df_original)

    st.markdown('<div class="section-title">Vista previa del archivo cargado</div>', unsafe_allow_html=True)
    mostrar_info(
        "La vista previa permite verificar que el archivo fue leído correctamente. "
        "En este punto todavía no se ejecutan los modelos; primero se revisa estructura, fechas, productos y columnas mínimas."
    )
    st.dataframe(df_original.head(20), use_container_width=True)

    if errores:
        for error in errores:
            st.error(error)
        st.stop()

    for advertencia in advertencias:
        st.warning(advertencia)

    archivo_id = f"{uploaded_file.name}-{getattr(uploaded_file, 'size', 0)}-{modo}"

else:
    df_original = None
    archivo_id = f"futuro-{fecha_inicio_futuro}-{meses_futuro}-{precio_menu_futuro}-{precio_sopa_futuro}-{precio_fanesca_futuro}-{precio_colada_futuro}"


if st.session_state.get("archivo_id") != archivo_id:
    st.session_state["archivo_id"] = archivo_id
    st.session_state["resultados_prediccion"] = None

if ejecutar:
    try:
        with st.spinner("Procesando datos, aplicando modelos y generando resultados..."):
            if modo == "Archivo cargado":
                df_limpio = limpiar_datos(df_original)
                df_preparado = preparar_datos(df_limpio)

                fecha_min = df_preparado["fecha"].min()
                fecha_max = df_preparado["fecha"].max()

                advertencia_rango = None
                if fecha_min < RANGO_HISTORICO_MIN or fecha_max > RANGO_HISTORICO_MAX:
                    advertencia_rango = (
                        "El archivo contiene fechas fuera del rango histórico usado para entrenar los modelos "
                        f"({RANGO_HISTORICO_MIN.date()} a {RANGO_HISTORICO_MAX.date()}). "
                        "La predicción puede ser menos precisa."
                    )

                tipo_resultado = "histórico"
            else:
                df_limpio = None
                df_preparado = generar_datos_futuros(
                    fecha_inicio=fecha_inicio_futuro,
                    meses=meses_futuro,
                    preciomenu=precio_menu_futuro,
                    preciosopa=precio_sopa_futuro,
                    fanesca_precio=precio_fanesca_futuro,
                    coladamorada_precio=precio_colada_futuro
                )
                advertencia_rango = (
                    "Predicción futura generada fuera del rango histórico de entrenamiento. "
                    "Interprétala como una estimación, no como dato observado."
                )
                tipo_resultado = "futuro"

            pred_wide, pred_diaria, pred_semanal, pred_mensual, diagnostico_pred = generar_predicciones(df_preparado)
            insights_globales = generar_insights(pred_diaria)

            metricas_modelos = None
            if modo == "Archivo cargado" and int(df_preparado[PRODUCTOS].sum().sum()) > 0:
                metricas_modelos = calcular_metricas_modelos(df_preparado, pred_wide)

            st.session_state["resultados_prediccion"] = {
                "df_limpio": df_limpio,
                "df_preparado": df_preparado,
                "pred_wide": pred_wide,
                "pred_diaria": pred_diaria,
                "pred_semanal": pred_semanal,
                "pred_mensual": pred_mensual,
                "insights_globales": insights_globales,
                "metricas_modelos": metricas_modelos,
                "diagnostico_pred": diagnostico_pred,
                "advertencia_rango": advertencia_rango,
                "tipo_resultado": tipo_resultado,
            }

    except Exception as e:
        st.error(f"No se pudo ejecutar la predicción: {e}")
        st.stop()

if st.session_state.get("resultados_prediccion") is None:
    if modo == "Archivo cargado":
        st.info("El archivo fue cargado correctamente. Haz clic en **Generar predicción** para continuar.")
    else:
        st.info("Configura los precios futuros y haz clic en **Generar predicción futura**.")
    st.stop()

resultados = st.session_state["resultados_prediccion"]
df_preparado = resultados["df_preparado"]
pred_wide = resultados["pred_wide"]
pred_diaria = resultados["pred_diaria"]
pred_semanal = resultados["pred_semanal"]
pred_mensual = resultados["pred_mensual"]
insights_globales = resultados["insights_globales"]
metricas_modelos = resultados["metricas_modelos"]
diagnostico_pred = resultados["diagnostico_pred"]

if resultados.get("advertencia_rango"):
    st.warning(resultados["advertencia_rango"])

st.success("Predicción generada correctamente.")

# KPIs globales
k1, k2, k3, k4 = st.columns(4)
with k1:
    mostrar_kpi_card("Total estimado", formato_entero(insights_globales["total"]), "Platos proyectados")
with k2:
    mostrar_kpi_card("Producto líder", insights_globales["producto_mayor"], "Mayor demanda total")
with k3:
    mostrar_kpi_card("Día fuerte", insights_globales["dia_mayor"], "Mayor consumo estimado")
with k4:
    mostrar_kpi_card("Mes fuerte", insights_globales["mes_mayor"], f"Fecha pico: {insights_globales['fecha_pico']}")

st.markdown('<div class="section-title">Insights automáticos</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    mostrar_insight(f"Producto con mayor demanda: {insights_globales['producto_mayor']}.")
    mostrar_insight(f"El día con mayor consumo estimado es {insights_globales['dia_mayor']}.")
with c2:
    mostrar_insight(f"El mes con mayor consumo estimado es {insights_globales['mes_mayor']}.")
    mostrar_insight(f"Total estimado: {formato_entero(insights_globales['total'])} platos.")

with st.expander("Ver variables usadas en la predicción"):
    st.dataframe(diagnostico_pred, use_container_width=True, hide_index=True)


# Evaluación solo con datos reales
if metricas_modelos is not None and not metricas_modelos.empty:
    st.markdown('<div class="section-title">Evaluación de calidad por producto</div>', unsafe_allow_html=True)
    mostrar_info(
        "Esta sección compara los valores reales del archivo cargado contra las predicciones generadas por cada modelo. "
        "Cada PKL corresponde a un producto específico; por eso la comparación indica qué producto está siendo mejor estimado, "
        "no que un modelo pueda aplicarse a otro plato. Las métricas principales son <b>MAE</b>, <b>RMSE</b> y <b>R²</b>."
    )

    mejor_fila = metricas_modelos.iloc[0]
    peor_fila = metricas_modelos.sort_values("ranking_valor", ascending=False).iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Mejor predicción", mejor_fila["Modelo"])
    with m2:
        st.metric("Criterio", f"Menor RMSE: {mejor_fila['RMSE']:.2f}")
    with m3:
        st.metric("Requiere revisión", peor_fila["Modelo"])
    with m4:
        st.metric("Total real evaluado", formato_entero(df_preparado[PRODUCTOS].sum().sum()))

    cmet1, cmet2, cmet3 = st.columns(3)
    with cmet1:
        st.metric("MAE mejor modelo", formato_decimal(mejor_fila["MAE"], 2))
    with cmet2:
        st.metric("RMSE mejor modelo", formato_decimal(mejor_fila["RMSE"], 2))
    with cmet3:
        st.metric("R² mejor modelo", formato_decimal(mejor_fila["R²"], 3))

    mostrar_insight(
        f"El producto con mejor comportamiento predictivo es **{mejor_fila['Modelo']}**, "
        f"porque tiene el menor RMSE dentro de la evaluación."
    )
    mostrar_insight(
        f"El producto que requiere más revisión es **{peor_fila['Modelo']}**, "
        f"porque presenta el error más alto según RMSE."
    )

    with st.expander("Ver ranking y explicación de métricas", expanded=True):
        st.dataframe(formatear_metricas(metricas_modelos), use_container_width=True, hide_index=True)
        st.markdown(
            """
            **Cómo leer esta sección:**  

            - **MAE (Mean Absolute Error):** mide el error promedio en unidades reales.  
              Si el MAE es 10, el modelo se equivoca en promedio por 10 platos. Menor es mejor.

            - **RMSE (Root Mean Squared Error):** mide error en unidades, pero penaliza más los errores grandes.  
              Es útil para detectar productos donde el modelo puede fallar mucho en ciertos días. Menor es mejor.

            - **R² (coeficiente de determinación):** indica qué tanto el modelo explica la variabilidad de los datos reales.  
              Un valor cercano a 1 es mejor. Si es negativo, el modelo puede funcionar peor que una predicción basada en el promedio.
            """
        )

    st.markdown("#### Gráficos de métricas")
    mostrar_info(
        "Los gráficos comparan el desempeño de cada producto. MAE y RMSE se leen en unidades de platos, "
        "mientras que R² mide capacidad explicativa. En MAE/RMSE menor es mejor; en R² mayor es mejor."
    )

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(grafico_mae_rmse_por_producto(metricas_modelos), use_container_width=True)
    with g2:
        st.plotly_chart(grafico_r2_por_producto(metricas_modelos), use_container_width=True)

    st.plotly_chart(grafico_totales_real_predicho(metricas_modelos), use_container_width=True)

elif resultados.get("tipo_resultado") == "futuro":
    st.info(
        "La evaluación de métricas no se muestra en modo futuro porque no existen valores reales contra los cuales comparar. "
        "En este modo solo se generan estimaciones de demanda."
    )


# Consulta por periodo
st.markdown('<div class="section-title">Consulta por periodo</div>', unsafe_allow_html=True)
mostrar_info(
    "Esta sección permite explorar la predicción a distintos niveles. "
    "Selecciona <b>Día</b> para revisar una fecha específica, <b>Semana</b> para acumular la demanda semanal "
    "o <b>Mes</b> para analizar el total mensual estimado."
)

nivel = st.radio(
    "Selecciona el nivel de consulta",
    ["Día", "Semana", "Mes"],
    horizontal=True,
    help=(
        "Día muestra una fecha puntual. Semana acumula todos los días de la semana seleccionada. "
        "Mes agrupa todas las predicciones del mes elegido."
    )
)

fecha_min_date = pred_diaria["fecha"].min().date()
fecha_max_date = pred_diaria["fecha"].max().date()

df_filtrado = pred_diaria.copy()
titulo_periodo = "Periodo completo"

if nivel == "Día":
    fecha_sel = st.date_input("Selecciona una fecha", value=fecha_min_date, min_value=fecha_min_date, max_value=fecha_max_date)
    fecha_sel = pd.to_datetime(fecha_sel)
    df_filtrado = pred_diaria[pred_diaria["fecha"] == fecha_sel].copy()
    titulo_periodo = f"Día seleccionado: {fecha_sel.date()}"

elif nivel == "Semana":
    fecha_ref = st.date_input("Selecciona una fecha dentro de la semana", value=fecha_min_date, min_value=fecha_min_date, max_value=fecha_max_date)
    fecha_ref = pd.to_datetime(fecha_ref)
    anio_sel = int(fecha_ref.isocalendar().year)
    semana_sel = int(fecha_ref.isocalendar().week)

    df_filtrado = pred_diaria[
        (pred_diaria["fecha"].dt.isocalendar().year.astype(int) == anio_sel) &
        (pred_diaria["semana_anio"] == semana_sel)
    ].copy()
    titulo_periodo = f"Semana seleccionada: {anio_sel} - Semana {semana_sel}"

else:
    meses_disponibles = pred_diaria[["anio", "mes", "mes_nombre"]].drop_duplicates().sort_values(["anio", "mes"])
    opciones_mes = [f"{int(row.anio)}-{int(row.mes):02d} | {row.mes_nombre}" for _, row in meses_disponibles.iterrows()]
    opcion = st.selectbox("Selecciona un mes", opciones_mes)
    anio_sel = int(opcion.split("-")[0])
    mes_sel = int(opcion.split("-")[1].split(" ")[0])
    df_filtrado = pred_diaria[(pred_diaria["anio"] == anio_sel) & (pred_diaria["mes"] == mes_sel)].copy()
    titulo_periodo = f"Mes seleccionado: {MESES_ES[mes_sel]} {anio_sel}"

if df_filtrado.empty:
    st.warning("No hay predicciones para el periodo seleccionado.")
    st.stop()

insights_periodo = generar_insights(df_filtrado)
st.markdown(f"#### {titulo_periodo}")

p1, p2, p3 = st.columns(3)
with p1:
    st.metric("Total del periodo", formato_entero(insights_periodo["total"]))
with p2:
    st.metric("Producto con mayor demanda", insights_periodo["producto_mayor"])
with p3:
    st.metric("Día con mayor consumo", insights_periodo["dia_mayor"])

st.markdown("### Tabla de predicciones")

if nivel == "Día":
    tabla_periodo = pivot_predicciones(df_filtrado)
else:
    tabla_periodo = df_filtrado.groupby("tipo_plato", as_index=False)["cantidad_predicha"].sum()
    tabla_periodo["Producto"] = tabla_periodo["tipo_plato"].map(nombre_producto)
    tabla_periodo = tabla_periodo[["Producto", "cantidad_predicha"]].rename(columns={"cantidad_predicha": "Cantidad predicha"})

st.dataframe(tabla_periodo, use_container_width=True, hide_index=True)

st.markdown("### Gráficos dinámicos")

if nivel == "Día":
    df_periodo_grafico = df_filtrado.assign(
        Producto=df_filtrado["tipo_plato"].map(nombre_producto),
        Etiqueta=df_filtrado["cantidad_predicha"].apply(formato_entero)
    )

    fig_periodo = px.bar(
        df_periodo_grafico,
        x="Producto",
        y="cantidad_predicha",
        text="Etiqueta",
        labels={"cantidad_predicha": "Cantidad predicha"},
        title="Predicción del día seleccionado"
    )
    fig_periodo.update_layout(height=420, title_font_size=20)
    aplicar_estilo_barras(fig_periodo)
    st.plotly_chart(fig_periodo, use_container_width=True)
elif nivel == "Semana":
    st.plotly_chart(grafico_diario(df_filtrado), use_container_width=True)
else:
    st.plotly_chart(grafico_semanal(df_filtrado), use_container_width=True)

tab1, tab2, tab3 = st.tabs(["Predicción diaria", "Consumo semanal", "Consumo mensual"])

with tab1:
    st.plotly_chart(grafico_diario(pred_diaria), use_container_width=True)
    st.dataframe(pivot_predicciones(pred_diaria), use_container_width=True, hide_index=True)

with tab2:
    st.plotly_chart(grafico_semanal(pred_diaria), use_container_width=True)
    st.dataframe(pred_semanal, use_container_width=True, hide_index=True)

with tab3:
    st.plotly_chart(grafico_mensual(pred_diaria), use_container_width=True)
    st.dataframe(pred_mensual, use_container_width=True, hide_index=True)

excel_buffer = crear_excel_descargable(
    pred_wide=pred_wide,
    diaria=pred_diaria,
    semanal=pred_semanal,
    mensual=pred_mensual,
    insights=insights_globales,
    metricas=metricas_modelos,
    diagnostico=diagnostico_pred
)

st.download_button(
    label="Descargar Excel con predicciones",
    data=excel_buffer,
    file_name="predicciones_demanda.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True
)
