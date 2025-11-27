"""
Feature Engineering for SBA Mexico Expected Loss Model
Adapted for Mexican SME (PyME) loan market with NAFIN guarantees

Variables adaptadas:
- NAICS → SCIAN (Sistema de Clasificación Industrial de América del Norte)
- SBA_Appv → NAFIN_Appv (Nacional Financiera)
- State → Estados mexicanos (CDMX, JAL, NL, etc.)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def prepare_data(filepath):
    """
    Load and prepare Mexican PyME loan data with all features
    
    Parámetros:
    -----------
    filepath : str
        Ruta al archivo CSV (sba_mexico_sintetico.csv)
    
    Retorna:
    --------
    X : DataFrame con features
    y_pd : Series con variable de default (0/1)
    y_loss : Series con pérdida en MXN (ChgOffPrinGr)
    df : DataFrame original con todas las columnas
    """
    df = pd.read_csv(filepath)
    
    # =========================================
    # TARGET VARIABLES (Variables Objetivo)
    # =========================================
    # ChgOffPrinGr: Monto del principal cargado a pérdida (en MXN)
    df['ChgOffPrinGr'] = df['ChgOffPrinGr'].fillna(0).astype(float)
    y_loss = df['ChgOffPrinGr']
    
    # Default: 1 si hubo incumplimiento, 0 si se pagó
    # Usamos la columna Default directamente si existe, sino la derivamos
    if 'Default' in df.columns:
        y_pd = df['Default'].astype(int)
    else:
        y_pd = (y_loss > 0).astype(int)
    
    # =========================================
    # CLEAN BASE FEATURES (Limpieza de datos)
    # =========================================
    # GrAppv: Monto bruto aprobado por el banco (MXN)
    df['GrAppv'] = df['GrAppv'].fillna(0.0)
    
    # NAFIN_Appv: Monto garantizado por Nacional Financiera (MXN)
    # Equivalente a SBA_Appv en el modelo original
    if 'NAFIN_Appv' in df.columns:
        df['NAFIN_Appv'] = df['NAFIN_Appv'].fillna(0.0)
    elif 'SBA_Appv' in df.columns:
        # Compatibilidad con formato SBA original
        df['NAFIN_Appv'] = df['SBA_Appv'].fillna(0.0)
    else:
        df['NAFIN_Appv'] = 0.0
    
    # Term: Plazo del préstamo en meses
    df['Term'] = df['Term'].fillna(0).astype(int)
    
    # NoEmp: Número de empleados
    df['NoEmp'] = df['NoEmp'].fillna(0).astype(int)
    
    # NewExist: 1 = Nuevo, 2 = Existente
    df['NewExist'] = df['NewExist'].fillna(2.0)
    
    # IsNewBusiness: Variable binaria derivada
    if 'New' in df.columns:
        df['IsNewBusiness'] = df['New'].astype(int)
    else:
        df['IsNewBusiness'] = (df['NewExist'] == 1.0).astype(int)
    
    # SCIAN: Sistema de Clasificación Industrial de América del Norte (México)
    # Equivalente al NAICS, compatible a 2 dígitos
    if 'SCIAN' in df.columns:
        df['SCIAN'] = df['SCIAN'].astype(str).str[:2].replace({'0': '00', 'na': '00', 'nan': '00'})
    elif 'NAICS' in df.columns:
        df['SCIAN'] = df['NAICS'].astype(str).str[:2].replace({'0': '00', 'na': '00', 'nan': '00'})
    else:
        df['SCIAN'] = '00'
    
    # State: Estado de la República Mexicana
    if 'State' in df.columns:
        df['State'] = df['State'].fillna('CDMX').astype(str)
    else:
        df['State'] = 'CDMX'
    
    # =========================================
    # FEATURE ENGINEERING (Ingeniería de características)
    # =========================================
    
    # NAFIN_Portion: Proporción del préstamo garantizada por NAFIN
    # Equivalente a SBA_Portion
    df['NAFIN_Portion'] = np.where(
        df['GrAppv'] > 0, 
        df['NAFIN_Appv'] / df['GrAppv'], 
        0.0
    )
    
    # Loan_per_Employee: Monto del préstamo por empleado
    # Indicador de tamaño relativo del préstamo
    df['Loan_per_Employee'] = np.where(
        (df['NoEmp'] + 1) > 0, 
        df['GrAppv'] / (df['NoEmp'] + 1), 
        0.0
    )
    
    # Term_Years: Plazo en años
    df['Term_Years'] = df['Term'] / 12.0
    
    # Debt_to_NAFIN: Porción no garantizada (riesgo del banco)
    df['Debt_to_NAFIN'] = df['GrAppv'] - df['NAFIN_Appv']
    
    # Log_GrAppv: Logaritmo del monto (para capturar efectos no lineales)
    df['Log_GrAppv'] = np.log1p(df['GrAppv'])
    
    # =========================================
    # FEATURES ADICIONALES PARA MÉXICO
    # =========================================
    
    # RealEstate: Indicador de garantía inmobiliaria
    if 'RealEstate' in df.columns:
        df['HasRealEstate'] = df['RealEstate'].fillna(0).astype(int)
    else:
        df['HasRealEstate'] = 0
    
    # Recession: Indicador de período de crisis (COVID 2020-2021)
    if 'Recession' in df.columns:
        df['InRecession'] = df['Recession'].fillna(0).astype(int)
    else:
        df['InRecession'] = 0
    
    # UrbanRural: Ubicación (1=Urbano, 2=Rural, 0=No especificado)
    if 'UrbanRural' in df.columns:
        df['IsUrban'] = (df['UrbanRural'] == 1).astype(int)
    else:
        df['IsUrban'] = 1
    
    # =========================================
    # SELECT FEATURES (Selección de variables)
    # =========================================
    num_feats = [
        'GrAppv',           # Monto aprobado
        'NAFIN_Appv',       # Monto garantizado por NAFIN
        'Debt_to_NAFIN',    # Porción no garantizada
        'Log_GrAppv',       # Log del monto
        'Term',             # Plazo en meses
        'Term_Years',       # Plazo en años
        'NoEmp',            # Número de empleados
        'IsNewBusiness',    # Negocio nuevo (1) o existente (0)
        'NAFIN_Portion',    # Proporción garantizada
        'Loan_per_Employee', # Monto por empleado
        'HasRealEstate',    # Tiene garantía inmobiliaria
        'InRecession',      # En período de crisis
        'IsUrban',          # Ubicación urbana
    ]
    
    cat_feats = ['SCIAN', 'State']
    
    # Verificar que todas las columnas existen
    available_num = [f for f in num_feats if f in df.columns]
    available_cat = [f for f in cat_feats if f in df.columns]
    
    X = df[available_num + available_cat].copy()
    
    return X, y_pd, y_loss, df


def create_preprocessor():
    """
    Create preprocessing pipeline for Mexican PyME loan data
    
    Retorna:
    --------
    ColumnTransformer con StandardScaler para numéricas y OneHotEncoder para categóricas
    """
    num_feats = [
        'GrAppv', 'NAFIN_Appv', 'Debt_to_NAFIN', 'Log_GrAppv', 
        'Term', 'Term_Years', 'NoEmp', 'IsNewBusiness', 
        'NAFIN_Portion', 'Loan_per_Employee', 'HasRealEstate',
        'InRecession', 'IsUrban'
    ]
    
    cat_feats = ['SCIAN', 'State']
    
    # OneHotEncoder para variables categóricas
    try:
        cat_tf = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Compatibilidad con versiones anteriores de sklearn
        cat_tf = OneHotEncoder(handle_unknown="ignore", sparse=False)
    
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_feats),
        ("cat", cat_tf, cat_feats),
    ], remainder="drop")
    
    return preprocessor


def transform_data(preprocessor, X):
    """
    Transform data preserving DataFrame structure
    
    Parámetros:
    -----------
    preprocessor : ColumnTransformer fitteado
    X : DataFrame con features originales
    
    Retorna:
    --------
    DataFrame con features transformadas
    """
    X_mat = preprocessor.transform(X)
    
    # Convertir sparse matrix a dense si es necesario
    if hasattr(X_mat, "toarray"):
        X_mat = X_mat.toarray()
    
    # Obtener nombres de columnas
    try:
        cols = preprocessor.get_feature_names_out()
    except:
        cols = [f"f{i}" for i in range(X_mat.shape[1])]
    
    return pd.DataFrame(X_mat, index=X.index, columns=cols)


# =========================================
# DICCIONARIO DE SECTORES SCIAN
# =========================================
SECTORES_SCIAN = {
    '11': 'Agricultura, ganadería, pesca',
    '21': 'Minería',
    '22': 'Generación de energía',
    '23': 'Construcción',
    '31': 'Manufactura - Alimentos',
    '32': 'Manufactura - Textil/Química',
    '33': 'Manufactura - Metálica/Maquinaria',
    '43': 'Comercio al por mayor',
    '46': 'Comercio al por menor',
    '48': 'Transporte',
    '51': 'Información en medios',
    '52': 'Servicios financieros',
    '53': 'Servicios inmobiliarios',
    '54': 'Servicios profesionales',
    '56': 'Servicios de apoyo',
    '61': 'Servicios educativos',
    '62': 'Servicios de salud',
    '71': 'Esparcimiento/Cultura',
    '72': 'Alojamiento/Alimentos',
    '81': 'Otros servicios',
    '00': 'No especificado',
}

# =========================================
# DICCIONARIO DE ESTADOS MEXICANOS
# =========================================
ESTADOS_MEXICO = {
    'AGS': 'Aguascalientes',
    'BC': 'Baja California',
    'BCS': 'Baja California Sur',
    'CAM': 'Campeche',
    'CHIS': 'Chiapas',
    'CHIH': 'Chihuahua',
    'CDMX': 'Ciudad de México',
    'COAH': 'Coahuila',
    'COL': 'Colima',
    'DGO': 'Durango',
    'GTO': 'Guanajuato',
    'GRO': 'Guerrero',
    'HGO': 'Hidalgo',
    'JAL': 'Jalisco',
    'MEX': 'Estado de México',
    'MIC': 'Michoacán',
    'MOR': 'Morelos',
    'NAY': 'Nayarit',
    'NL': 'Nuevo León',
    'OAX': 'Oaxaca',
    'PUE': 'Puebla',
    'QRO': 'Querétaro',
    'QROO': 'Quintana Roo',
    'SLP': 'San Luis Potosí',
    'SIN': 'Sinaloa',
    'SON': 'Sonora',
    'TAB': 'Tabasco',
    'TAM': 'Tamaulipas',
    'TLAX': 'Tlaxcala',
    'VER': 'Veracruz',
    'YUC': 'Yucatán',
    'ZAC': 'Zacatecas',
}
