"""
================================================================================
GENERADOR DE DATASET SINTÉTICO V2 - CRÉDITOS PYME MÉXICO
================================================================================
Versión mejorada con:
- Relaciones causales más fuertes entre variables y default
- Efectos multiplicativos (no solo aditivos)
- Menor ruido aleatorio
- Calibración directa con tasas IMOR de CNBV por sector

Objetivo: Generar dataset donde el modelo pueda alcanzar AUC > 0.75
================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ==============================================================================
# PARÁMETROS CALIBRADOS CON DATOS REALES DE MÉXICO
# ==============================================================================

# Distribución de empresas por Estado (Censos Económicos 2024)
DISTRIBUCION_ESTADOS = {
    'MEX': 0.127, 'VER': 0.069, 'PUE': 0.067, 'JAL': 0.065, 'CDMX': 0.062,
    'GTO': 0.055, 'MIC': 0.048, 'OAX': 0.045, 'CHIS': 0.042, 'NL': 0.040,
    'GRO': 0.035, 'TAM': 0.032, 'SLP': 0.030, 'CHIH': 0.028, 'SIN': 0.027,
    'HGO': 0.026, 'SON': 0.025, 'TAB': 0.022, 'COAH': 0.021, 'QRO': 0.020,
    'MOR': 0.019, 'DGO': 0.016, 'YUC': 0.016, 'QROO': 0.015, 'ZAC': 0.014,
    'AGS': 0.013, 'TLAX': 0.012, 'NAY': 0.010, 'COL': 0.008, 'CAM': 0.007,
    'BCS': 0.006, 'BC': 0.025,
}

# Estados con mayor desarrollo económico (menor riesgo base)
ESTADOS_DESARROLLO_ALTO = {'NL', 'CDMX', 'JAL', 'QRO', 'AGS', 'COAH', 'CHIH', 'BC', 'SON'}
ESTADOS_DESARROLLO_MEDIO = {'MEX', 'GTO', 'PUE', 'TAM', 'SIN', 'YUC', 'QROO', 'MOR', 'SLP', 'DGO', 'BCS', 'COL'}
# El resto son desarrollo bajo

# Sectores SCIAN con tasas IMOR calibradas de CNBV
# Las tasas son las tasas BASE de morosidad del sector
SECTORES_SCIAN = {
    '11': {'nombre': 'Agricultura, ganadería, pesca', 'pct': 0.02, 'imor_base': 0.09, 'volatilidad': 'alta'},
    '21': {'nombre': 'Minería', 'pct': 0.01, 'imor_base': 0.07, 'volatilidad': 'alta'},
    '22': {'nombre': 'Generación de energía', 'pct': 0.005, 'imor_base': 0.03, 'volatilidad': 'baja'},
    '23': {'nombre': 'Construcción', 'pct': 0.08, 'imor_base': 0.11, 'volatilidad': 'alta'},
    '31': {'nombre': 'Manufactura - Alimentos', 'pct': 0.06, 'imor_base': 0.05, 'volatilidad': 'baja'},
    '32': {'nombre': 'Manufactura - Textil/Química', 'pct': 0.04, 'imor_base': 0.07, 'volatilidad': 'media'},
    '33': {'nombre': 'Manufactura - Metálica/Maquinaria', 'pct': 0.05, 'imor_base': 0.06, 'volatilidad': 'media'},
    '43': {'nombre': 'Comercio al por mayor', 'pct': 0.08, 'imor_base': 0.06, 'volatilidad': 'media'},
    '46': {'nombre': 'Comercio al por menor', 'pct': 0.35, 'imor_base': 0.08, 'volatilidad': 'media'},
    '48': {'nombre': 'Transporte', 'pct': 0.04, 'imor_base': 0.08, 'volatilidad': 'alta'},
    '51': {'nombre': 'Información en medios', 'pct': 0.02, 'imor_base': 0.05, 'volatilidad': 'media'},
    '52': {'nombre': 'Servicios financieros', 'pct': 0.02, 'imor_base': 0.03, 'volatilidad': 'baja'},
    '53': {'nombre': 'Servicios inmobiliarios', 'pct': 0.03, 'imor_base': 0.07, 'volatilidad': 'alta'},
    '54': {'nombre': 'Servicios profesionales', 'pct': 0.05, 'imor_base': 0.04, 'volatilidad': 'baja'},
    '56': {'nombre': 'Servicios de apoyo', 'pct': 0.04, 'imor_base': 0.06, 'volatilidad': 'media'},
    '61': {'nombre': 'Servicios educativos', 'pct': 0.02, 'imor_base': 0.03, 'volatilidad': 'baja'},
    '62': {'nombre': 'Servicios de salud', 'pct': 0.03, 'imor_base': 0.03, 'volatilidad': 'baja'},
    '71': {'nombre': 'Esparcimiento/Cultura', 'pct': 0.02, 'imor_base': 0.10, 'volatilidad': 'alta'},
    '72': {'nombre': 'Alojamiento/Alimentos', 'pct': 0.12, 'imor_base': 0.12, 'volatilidad': 'alta'},
    '81': {'nombre': 'Otros servicios', 'pct': 0.06, 'imor_base': 0.08, 'volatilidad': 'media'},
}

# Bancos
BANCOS_MEXICO = {
    'BBVA México': 0.22, 'Banorte': 0.18, 'Santander México': 0.15,
    'Citibanamex': 0.12, 'HSBC México': 0.08, 'Scotiabank México': 0.06,
    'Banco Azteca': 0.05, 'Inbursa': 0.04, 'BanBajío': 0.03,
    'Banregio': 0.025, 'Afirme': 0.02, 'Banco del Bajío': 0.015,
    'Multiva': 0.01, 'Banca Mifel': 0.01,
}

PARAMETROS_CREDITO = {
    'monto_min': 50_000,
    'monto_max': 50_000_000,
    'monto_mediana_micro': 150_000,
    'monto_mediana_pequena': 800_000,
    'monto_mediana_mediana': 3_500_000,
    'monto_mediana_grande': 15_000_000,
    'tasa_min': 8.0,
    'tasa_max': 28.0,
    'lgd_con_garantia_nafin': 0.25,
    'lgd_con_garantia_inmob': 0.35,
    'lgd_sin_garantia': 0.75,
}

# ==============================================================================
# FUNCIONES GENERADORAS MEJORADAS
# ==============================================================================

def generar_estados(n):
    estados = list(DISTRIBUCION_ESTADOS.keys())
    probs = np.array(list(DISTRIBUCION_ESTADOS.values()))
    probs = probs / probs.sum()  # Normalizar para que sumen 1
    return np.random.choice(estados, size=n, p=probs)

def generar_sectores(n):
    sectores = list(SECTORES_SCIAN.keys())
    probs = np.array([SECTORES_SCIAN[s]['pct'] for s in sectores])
    probs = probs / probs.sum()  # Normalizar
    return np.random.choice(sectores, size=n, p=probs)

def generar_bancos(n):
    bancos = list(BANCOS_MEXICO.keys())
    probs = np.array(list(BANCOS_MEXICO.values()))
    probs = probs / probs.sum()  # Normalizar
    return np.random.choice(bancos, size=n, p=probs)

def generar_tamano_empresa(n):
    """Distribución de tamaño de empresas en México"""
    tamanos = ['micro', 'pequena', 'mediana', 'grande']
    probs = [0.95, 0.035, 0.012, 0.003]
    return np.random.choice(tamanos, size=n, p=probs)

def generar_empleados(tamanos):
    """Genera número de empleados según tamaño"""
    empleados = []
    for t in tamanos:
        if t == 'micro':
            emp = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                                   p=[0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05, 0.03, 0.02])
        elif t == 'pequena':
            emp = np.random.randint(11, 51)
        elif t == 'mediana':
            emp = np.random.randint(51, 251)
        else:
            emp = np.random.randint(251, 501)
        empleados.append(emp)
    return np.array(empleados)

def generar_new_exist(n, sectores):
    """
    Genera si es negocio nuevo o existente.
    La probabilidad de ser nuevo varía por sector (algunos tienen más rotación).
    """
    new_exist = []
    for scian in sectores:
        vol = SECTORES_SCIAN.get(scian, {}).get('volatilidad', 'media')
        if vol == 'alta':
            prob_nuevo = 0.35  # Sectores volátiles tienen más negocios nuevos
        elif vol == 'media':
            prob_nuevo = 0.25
        else:
            prob_nuevo = 0.15  # Sectores estables tienen negocios más establecidos
        
        new_exist.append(1 if np.random.random() < prob_nuevo else 2)
    return np.array(new_exist)

def generar_montos(tamanos, sectores):
    """Genera montos de préstamo según tamaño y sector"""
    montos = []
    for t, scian in zip(tamanos, sectores):
        if t == 'micro':
            mediana = PARAMETROS_CREDITO['monto_mediana_micro']
        elif t == 'pequena':
            mediana = PARAMETROS_CREDITO['monto_mediana_pequena']
        elif t == 'mediana':
            mediana = PARAMETROS_CREDITO['monto_mediana_mediana']
        else:
            mediana = PARAMETROS_CREDITO['monto_mediana_grande']
        
        # Sectores de construcción e inmobiliario requieren más capital
        if scian in ['23', '53']:
            mediana *= 1.3
        
        monto = np.random.lognormal(np.log(mediana), 0.6)
        monto = np.clip(monto, PARAMETROS_CREDITO['monto_min'], 
                       PARAMETROS_CREDITO['monto_max'])
        montos.append(round(monto, -3))
    
    return np.array(montos)

def generar_plazo(n, montos):
    """Plazos correlacionados con monto (préstamos grandes = plazos más largos)"""
    plazos = []
    for monto in montos:
        if monto < 200_000:
            plazo = np.random.choice([6, 12, 18, 24], p=[0.15, 0.35, 0.25, 0.25])
        elif monto < 500_000:
            plazo = np.random.choice([12, 18, 24, 36], p=[0.20, 0.25, 0.30, 0.25])
        elif monto < 2_000_000:
            plazo = np.random.choice([24, 36, 48, 60], p=[0.20, 0.35, 0.30, 0.15])
        else:
            plazo = np.random.choice([36, 48, 60, 84, 120], p=[0.15, 0.25, 0.30, 0.20, 0.10])
        plazos.append(plazo)
    return np.array(plazos)

def generar_programa_nafin(n, tamanos, sectores):
    """
    NAFIN prioriza ciertos sectores y tamaños.
    Esto será importante para el modelo - tener NAFIN reduce riesgo.
    """
    con_nafin = []
    for t, scian in zip(tamanos, sectores):
        # NAFIN tiene programas prioritarios
        if scian in ['31', '32', '33']:  # Manufactura - prioritario
            prob_nafin = 0.50
        elif scian in ['11']:  # Agricultura
            prob_nafin = 0.45
        elif scian in ['54', '51']:  # Servicios profesionales, tecnología
            prob_nafin = 0.40
        elif scian in ['72', '71']:  # Restaurantes, entretenimiento - más difícil
            prob_nafin = 0.20
        else:
            prob_nafin = 0.35
        
        # Empresas más grandes tienen mejor acceso
        if t == 'grande':
            prob_nafin *= 1.3
        elif t == 'mediana':
            prob_nafin *= 1.15
        elif t == 'micro':
            prob_nafin *= 0.85
        
        prob_nafin = min(prob_nafin, 0.70)
        con_nafin.append(np.random.random() < prob_nafin)
    
    return np.array(con_nafin)

def generar_garantia_nafin(con_programa_nafin, montos):
    """Porcentaje de garantía NAFIN"""
    garantias = []
    for tiene_nafin, monto in zip(con_programa_nafin, montos):
        if tiene_nafin:
            # Préstamos más pequeños obtienen mayor porcentaje de garantía
            if monto <= 500_000:
                garantia = np.random.uniform(0.70, 0.80)
            elif monto <= 2_000_000:
                garantia = np.random.uniform(0.60, 0.75)
            else:
                garantia = np.random.uniform(0.50, 0.70)
        else:
            garantia = 0.0
        garantias.append(round(garantia, 2))
    return np.array(garantias)

def generar_garantia_inmobiliaria(n, tamanos, montos):
    """Garantía inmobiliaria - más común en préstamos grandes y empresas establecidas"""
    garantias = []
    for t, monto in zip(tamanos, montos):
        if t == 'grande':
            prob = 0.60
        elif t == 'mediana':
            prob = 0.45
        elif t == 'pequena':
            prob = 0.30
        else:
            prob = 0.10
        
        # Préstamos más grandes requieren más garantía
        if monto > 2_000_000:
            prob += 0.15
        elif monto > 1_000_000:
            prob += 0.10
        
        prob = min(prob, 0.75)
        garantias.append(1 if np.random.random() < prob else 0)
    
    return np.array(garantias)

def generar_fecha_aprobacion(n, fecha_inicio='2015-01-01', fecha_fin='2024-12-31'):
    start = datetime.strptime(fecha_inicio, '%Y-%m-%d')
    end = datetime.strptime(fecha_fin, '%Y-%m-%d')
    delta = (end - start).days
    
    fechas = []
    for _ in range(n):
        random_days = np.random.randint(0, delta)
        fecha = start + timedelta(days=random_days)
        fechas.append(fecha)
    return fechas

def es_periodo_crisis(fecha):
    """COVID y otras crisis"""
    year = fecha.year
    month = fecha.month
    if year == 2020 and month >= 3:
        return True
    if year == 2021 and month <= 6:
        return True
    return False

def generar_urban_rural(estados):
    """Urbano/Rural basado en estado"""
    urban_rural = []
    for estado in estados:
        if estado in ['CDMX', 'NL', 'JAL', 'QRO', 'AGS']:
            prob_urbano = 0.90
        elif estado in ['MEX', 'GTO', 'PUE', 'COAH', 'CHIH', 'BC', 'SON', 'TAM']:
            prob_urbano = 0.75
        elif estado in ['OAX', 'CHIS', 'GRO', 'HGO', 'ZAC', 'DGO']:
            prob_urbano = 0.50
        else:
            prob_urbano = 0.65
        
        urban_rural.append(1 if np.random.random() < prob_urbano else 2)
    return np.array(urban_rural)


# ==============================================================================
# FUNCIÓN CENTRAL: CÁLCULO DE PROBABILIDAD DE DEFAULT (MEJORADA)
# ==============================================================================

def calcular_score_riesgo(row):
    """
    Calcula un SCORE de riesgo (no probabilidad directa) que será más predictivo.
    Usa efectos MULTIPLICATIVOS para que las interacciones sean más fuertes.
    
    Retorna un score donde:
    - Score alto = Mayor probabilidad de default
    - Score bajo = Menor probabilidad de default
    """
    
    # =========================================================================
    # 1. COMPONENTE BASE: IMOR del sector (calibrado con CNBV)
    # =========================================================================
    sector_info = SECTORES_SCIAN.get(row['SCIAN'], {'imor_base': 0.07, 'volatilidad': 'media'})
    score_base = sector_info['imor_base']
    
    # =========================================================================
    # 2. MULTIPLICADORES (efectos que amplifican o reducen el riesgo)
    # =========================================================================
    multiplicador = 1.0
    
    # --- Negocio Nuevo vs Existente (EFECTO FUERTE) ---
    if row['NewExist'] == 1:  # Nuevo
        if sector_info['volatilidad'] == 'alta':
            multiplicador *= 1.8  # Nuevo en sector volátil = muy riesgoso
        elif sector_info['volatilidad'] == 'media':
            multiplicador *= 1.5
        else:
            multiplicador *= 1.3
    else:  # Existente
        multiplicador *= 0.85  # Bonus por ser establecido
    
    # --- Tamaño de Empresa (EFECTO FUERTE) ---
    if row['NoEmp'] <= 3:
        multiplicador *= 1.4  # Micro muy pequeña = alto riesgo
    elif row['NoEmp'] <= 10:
        multiplicador *= 1.15
    elif row['NoEmp'] <= 50:
        multiplicador *= 1.0  # Baseline
    elif row['NoEmp'] <= 250:
        multiplicador *= 0.80  # Mediana = menor riesgo
    else:
        multiplicador *= 0.60  # Grande = bajo riesgo
    
    # --- Garantía NAFIN (EFECTO PROTECTOR FUERTE) ---
    if row['Portion'] >= 0.70:
        multiplicador *= 0.65  # Alta garantía NAFIN = mucha protección
    elif row['Portion'] >= 0.50:
        multiplicador *= 0.75
    elif row['Portion'] > 0:
        multiplicador *= 0.85
    # Sin NAFIN = multiplicador no cambia
    
    # --- Garantía Inmobiliaria (EFECTO PROTECTOR) ---
    if row['RealEstate'] == 1:
        multiplicador *= 0.75  # Tener garantía real reduce riesgo
    
    # --- Ratio Monto/Empleado (EFECTO MODERADO) ---
    ratio = row['GrAppv'] / max(row['NoEmp'] * 80_000, 1)
    if ratio > 8:
        multiplicador *= 1.35  # Préstamo muy grande para el tamaño
    elif ratio > 5:
        multiplicador *= 1.20
    elif ratio > 3:
        multiplicador *= 1.10
    elif ratio < 1:
        multiplicador *= 0.90  # Préstamo conservador
    
    # --- Plazo (EFECTO MODERADO) ---
    if row['Term'] > 84:
        multiplicador *= 1.25  # Plazos muy largos = más tiempo para fallar
    elif row['Term'] > 60:
        multiplicador *= 1.15
    elif row['Term'] < 18:
        multiplicador *= 0.95  # Plazos cortos = menos riesgo
    
    # --- Ubicación Geográfica ---
    if row['State'] in ESTADOS_DESARROLLO_ALTO:
        multiplicador *= 0.85
    elif row['State'] in ESTADOS_DESARROLLO_MEDIO:
        multiplicador *= 1.0
    else:
        multiplicador *= 1.15  # Estados menos desarrollados
    
    if row['UrbanRural'] == 2:  # Rural
        multiplicador *= 1.20
    
    # --- Crisis/Recesión (EFECTO MUY FUERTE) ---
    if row['Recession'] == 1:
        if sector_info['volatilidad'] == 'alta':
            multiplicador *= 1.7  # Sector volátil en crisis = desastre
        else:
            multiplicador *= 1.4
    
    # =========================================================================
    # 3. CÁLCULO FINAL DEL SCORE
    # =========================================================================
    score_final = score_base * multiplicador
    
    # Clip para mantener en rango razonable (2% a 40%)
    score_final = np.clip(score_final, 0.02, 0.40)
    
    return score_final


def generar_default_deterministico(df):
    """
    Genera defaults de manera más determinística basada en el score.
    Reduce la aleatoriedad para que el modelo pueda aprender mejor.
    """
    # Calcular score de riesgo para cada registro
    df['score_riesgo'] = df.apply(calcular_score_riesgo, axis=1)
    
    # Convertir score a probabilidad con menos ruido
    # Usamos una distribución más "puntiaguda" alrededor del score
    n = len(df)
    
    # Generar default: usamos el score directamente con poco ruido
    ruido = np.random.normal(0, 0.02, n)  # Ruido pequeño (std=2%)
    prob_ajustada = df['score_riesgo'] + ruido
    prob_ajustada = np.clip(prob_ajustada, 0.01, 0.50)
    
    # Para hacer más determinístico: usamos umbrales
    # Los que tienen score muy alto SIEMPRE fallan, los muy bajos NUNCA fallan
    df['Default'] = 0
    
    # Score > 0.25: 80% de probabilidad de default (casi seguro)
    mask_muy_alto = prob_ajustada > 0.25
    df.loc[mask_muy_alto, 'Default'] = (np.random.random(mask_muy_alto.sum()) < 0.80).astype(int)
    
    # Score 0.15-0.25: probabilidad = score * 2.5 (amplificado)
    mask_alto = (prob_ajustada > 0.15) & (prob_ajustada <= 0.25)
    probs_alto = prob_ajustada[mask_alto] * 2.5
    df.loc[mask_alto, 'Default'] = (np.random.random(mask_alto.sum()) < probs_alto.values).astype(int)
    
    # Score 0.08-0.15: probabilidad = score * 1.5
    mask_medio = (prob_ajustada > 0.08) & (prob_ajustada <= 0.15)
    probs_medio = prob_ajustada[mask_medio] * 1.5
    df.loc[mask_medio, 'Default'] = (np.random.random(mask_medio.sum()) < probs_medio.values).astype(int)
    
    # Score 0.04-0.08: probabilidad = score
    mask_bajo = (prob_ajustada > 0.04) & (prob_ajustada <= 0.08)
    probs_bajo = prob_ajustada[mask_bajo]
    df.loc[mask_bajo, 'Default'] = (np.random.random(mask_bajo.sum()) < probs_bajo.values).astype(int)
    
    # Score < 0.04: 3% de probabilidad (muy bajo pero no cero)
    mask_muy_bajo = prob_ajustada <= 0.04
    df.loc[mask_muy_bajo, 'Default'] = (np.random.random(mask_muy_bajo.sum()) < 0.03).astype(int)
    
    # Guardar score para análisis (se puede eliminar después)
    df['prob_default'] = prob_ajustada
    
    return df


def generar_perdida(df):
    """Genera pérdida para los defaults"""
    df['ChgOffPrinGr'] = 0.0
    
    mask_default = df['Default'] == 1
    
    for idx in df[mask_default].index:
        tiene_nafin = df.loc[idx, 'Portion'] > 0
        tiene_inmobiliario = df.loc[idx, 'RealEstate'] == 1
        
        # LGD base según garantías
        if tiene_nafin and tiene_inmobiliario:
            lgd = 0.20  # Doble protección
        elif tiene_nafin:
            lgd = PARAMETROS_CREDITO['lgd_con_garantia_nafin']
        elif tiene_inmobiliario:
            lgd = PARAMETROS_CREDITO['lgd_con_garantia_inmob']
        else:
            lgd = PARAMETROS_CREDITO['lgd_sin_garantia']
        
        # Agregar variabilidad al LGD (menos que antes)
        lgd_ajustado = np.random.uniform(lgd * 0.85, lgd * 1.15)
        lgd_ajustado = np.clip(lgd_ajustado, 0.10, 0.95)
        
        # Saldo pendiente correlacionado con plazo transcurrido
        plazo = df.loc[idx, 'Term']
        # Asumimos que en promedio fallan a mitad del préstamo
        saldo_pendiente_pct = np.random.uniform(0.40, 0.80)
        
        perdida = df.loc[idx, 'GrAppv'] * lgd_ajustado * saldo_pendiente_pct
        df.loc[idx, 'ChgOffPrinGr'] = round(perdida, 2)
    
    return df


# ==============================================================================
# FUNCIÓN PRINCIPAL DE GENERACIÓN
# ==============================================================================

def generar_dataset_pyme_mexico(n_registros=50000):
    """Genera el dataset completo de préstamos PyME México"""
    
    print(f"Generando {n_registros:,} registros de préstamos PyME...")
    print("=" * 60)
    
    # 1. Variables base
    print("[1/10] Generando ubicaciones...")
    estados = generar_estados(n_registros)
    urban_rural = generar_urban_rural(estados)
    
    print("[2/10] Generando sectores económicos...")
    sectores = generar_sectores(n_registros)
    
    print("[3/10] Generando características de empresas...")
    tamanos = generar_tamano_empresa(n_registros)
    empleados = generar_empleados(tamanos)
    new_exist = generar_new_exist(n_registros, sectores)
    
    print("[4/10] Generando montos y plazos...")
    montos = generar_montos(tamanos, sectores)
    plazos = generar_plazo(n_registros, montos)
    
    print("[5/10] Generando garantías NAFIN...")
    con_nafin = generar_programa_nafin(n_registros, tamanos, sectores)
    garantias_nafin = generar_garantia_nafin(con_nafin, montos)
    nafin_appv = montos * garantias_nafin
    
    print("[6/10] Generando garantías inmobiliarias...")
    real_estate = generar_garantia_inmobiliaria(n_registros, tamanos, montos)
    
    print("[7/10] Generando fechas y períodos...")
    fechas = generar_fecha_aprobacion(n_registros)
    recession = [1 if es_periodo_crisis(f) else 0 for f in fechas]
    
    print("[8/10] Generando bancos y tasas...")
    bancos = generar_bancos(n_registros)
    # Tasa correlacionada con riesgo percibido
    tasas = []
    for t, scian, ne in zip(tamanos, sectores, new_exist):
        tasa_base = 12.0
        if t == 'micro':
            tasa_base += 4
        elif t == 'pequena':
            tasa_base += 2
        if ne == 1:  # Nuevo
            tasa_base += 3
        sector_info = SECTORES_SCIAN.get(scian, {})
        if sector_info.get('volatilidad') == 'alta':
            tasa_base += 2
        tasa = np.random.normal(tasa_base, 1.5)
        tasa = np.clip(tasa, PARAMETROS_CREDITO['tasa_min'], PARAMETROS_CREDITO['tasa_max'])
        tasas.append(round(tasa, 2))
    
    # Crear DataFrame
    df = pd.DataFrame({
        'LoanNr': range(1, n_registros + 1),
        'Name': [f'EMPRESA_{i:06d}' for i in range(1, n_registros + 1)],
        'State': estados,
        'SCIAN': sectores,
        'Bank': bancos,
        'ApprovalDate': fechas,
        'Term': plazos,
        'NoEmp': empleados,
        'NewExist': new_exist,
        'UrbanRural': urban_rural,
        'GrAppv': montos,
        'NAFIN_Appv': nafin_appv.round(2),
        'Portion': garantias_nafin,
        'RealEstate': real_estate,
        'Recession': recession,
        'InterestRate': tasas,
    })
    
    print("[9/10] Calculando defaults (modelo mejorado)...")
    df = generar_default_deterministico(df)
    
    print("[10/10] Calculando pérdidas...")
    df = generar_perdida(df)
    
    # Añadir columnas adicionales para compatibilidad SBA
    df['NAICS'] = df['SCIAN']  # Compatibilidad
    df['SBA_Appv'] = df['NAFIN_Appv']  # Compatibilidad
    df['ApprovalFY'] = df['ApprovalDate'].apply(lambda x: x.year)
    df['RevLineCr'] = np.random.choice(['Y', 'N'], size=n_registros, p=[0.25, 0.75])
    df['LowDoc'] = np.random.choice(['Y', 'N'], size=n_registros, p=[0.30, 0.70])
    
    # Limpiar columna temporal de probabilidad (opcional: mantener para análisis)
    # df.drop('prob_default', axis=1, inplace=True)
    
    print("=" * 60)
    print("✓ Dataset generado exitosamente!")
    
    return df


def analizar_dataset(df):
    """Análisis del dataset generado"""
    print("\n" + "=" * 60)
    print("ANÁLISIS DEL DATASET GENERADO")
    print("=" * 60)
    
    print(f"\n📊 Estadísticas Generales:")
    print(f"   Total préstamos: {len(df):,}")
    print(f"   Tasa de default: {df['Default'].mean()*100:.2f}%")
    print(f"   Monto total aprobado: ${df['GrAppv'].sum():,.0f} MXN")
    print(f"   Monto promedio: ${df['GrAppv'].mean():,.0f} MXN")
    print(f"   Pérdida total: ${df['ChgOffPrinGr'].sum():,.0f} MXN")
    print(f"   Con garantía NAFIN: {(df['Portion'] > 0).sum():,} ({(df['Portion'] > 0).mean()*100:.1f}%)")
    print(f"   Con garantía inmobiliaria: {(df['RealEstate'] == 1).sum():,} ({(df['RealEstate'] == 1).mean()*100:.1f}%)")
    
    print(f"\n📈 Distribución del Score de Riesgo:")
    print(f"   Mínimo: {df['prob_default'].min()*100:.1f}%")
    print(f"   Percentil 25: {df['prob_default'].quantile(0.25)*100:.1f}%")
    print(f"   Mediana: {df['prob_default'].median()*100:.1f}%")
    print(f"   Percentil 75: {df['prob_default'].quantile(0.75)*100:.1f}%")
    print(f"   Máximo: {df['prob_default'].max()*100:.1f}%")
    
    print(f"\n🏢 Default por Tamaño de Empresa:")
    df['TamanoEmpresa'] = pd.cut(df['NoEmp'], 
                                  bins=[0, 10, 50, 250, 1000],
                                  labels=['Micro (≤10)', 'Pequeña (11-50)', 
                                         'Mediana (51-250)', 'Grande (>250)'])
    for tam in df['TamanoEmpresa'].cat.categories:
        subset = df[df['TamanoEmpresa'] == tam]
        print(f"   {tam}: {subset['Default'].mean()*100:.1f}% default ({len(subset):,} préstamos)")
    
    print(f"\n🆕 Default por Tipo de Negocio:")
    print(f"   Nuevo: {df[df['NewExist']==1]['Default'].mean()*100:.1f}% default")
    print(f"   Existente: {df[df['NewExist']==2]['Default'].mean()*100:.1f}% default")
    
    print(f"\n🛡️ Default por Garantía NAFIN:")
    print(f"   Con NAFIN (≥70%): {df[df['Portion']>=0.70]['Default'].mean()*100:.1f}% default")
    print(f"   Con NAFIN (50-70%): {df[(df['Portion']>=0.50) & (df['Portion']<0.70)]['Default'].mean()*100:.1f}% default")
    print(f"   Con NAFIN (<50%): {df[(df['Portion']>0) & (df['Portion']<0.50)]['Default'].mean()*100:.1f}% default")
    print(f"   Sin NAFIN: {df[df['Portion']==0]['Default'].mean()*100:.1f}% default")
    
    print(f"\n🏠 Default por Garantía Inmobiliaria:")
    print(f"   Con garantía: {df[df['RealEstate']==1]['Default'].mean()*100:.1f}% default")
    print(f"   Sin garantía: {df[df['RealEstate']==0]['Default'].mean()*100:.1f}% default")
    
    print(f"\n📅 Default en Período de Crisis (COVID):")
    print(f"   Durante crisis: {df[df['Recession']==1]['Default'].mean()*100:.1f}% default")
    print(f"   Período normal: {df[df['Recession']==0]['Default'].mean()*100:.1f}% default")
    
    print(f"\n🏭 Top 5 Sectores con Mayor Default:")
    sector_default = df.groupby('SCIAN').agg({
        'Default': 'mean',
        'LoanNr': 'count',
        'GrAppv': 'sum'
    }).sort_values('Default', ascending=False)
    
    for scian in sector_default.head(5).index:
        info = SECTORES_SCIAN.get(scian, {'nombre': 'Desconocido'})
        tasa = sector_default.loc[scian, 'Default'] * 100
        n = sector_default.loc[scian, 'LoanNr']
        print(f"   {scian} - {info['nombre'][:30]}: {tasa:.1f}% ({n:,} préstamos)")
    
    print("\n" + "=" * 60)


# ==============================================================================
# EJECUCIÓN
# ==============================================================================

if __name__ == "__main__":
    # Generar dataset
    df = generar_dataset_pyme_mexico(n_registros=50000)
    
    # Analizar
    analizar_dataset(df)
    
    # Guardar
    output_file = 'sba_mexico_sintetico_v2.csv'
    
    # Preparar para guardar (eliminar columnas temporales si no se quieren)
    df_save = df.drop(['TamanoEmpresa', 'score_riesgo'], axis=1, errors='ignore')
    df_save['ApprovalDate'] = df_save['ApprovalDate'].dt.strftime('%Y-%m-%d')
    
    df_save.to_csv(output_file, index=False)
    print(f"\n✓ Dataset guardado en: {output_file}")
    print(f"  Tamaño: {len(df_save):,} registros, {len(df_save.columns)} columnas")
