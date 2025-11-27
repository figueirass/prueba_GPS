# Modelo de Pérdida Esperada para Créditos PyME México

Sistema de Machine Learning para estimar la Pérdida Esperada (EL) en garantías de créditos PyME en México, diseñado desde la perspectiva del garante (NAFIN) en lugar del prestamista.

## 🎯 Descripción del Proyecto

### Perspectiva NAFIN vs. Perspectiva del Banco

Este proyecto modela el riesgo crediticio desde el **punto de vista de NAFIN como garante**, lo cual es fundamentalmente diferente de los modelos de crédito bancarios tradicionales:

| Aspecto | Modelo Bancario | Modelo NAFIN (Este Proyecto) |
|---------|----------------|------------------------------|
| **Pregunta** | ¿Debemos aprobar este préstamo? (Sí/No) | ¿Cuánto dinero perderemos en esta garantía? ($) |
| **Tipo de Modelo** | Clasificación (Modelo PD) | Regresión (Modelo EL) |
| **Salida** | Probabilidad de Default | Pérdida Esperada en Pesos |
| **Decisión** | Aprobar/Rechazar | Cálculo de Comisión de Garantía |

### Concepto Clave: Pérdida Esperada

NAFIN necesita estimar la **Pérdida Esperada (EL)** para cada préstamo garantizado:

```
Pérdida Esperada (EL) = Probabilidad de Default (PD) × Pérdida Dado el Default (LGD)
```

- **PD**: Probabilidad de que el prestatario incumpla (0 a 1)
- **LGD**: Monto en pesos que NAFIN perderá si ocurre el incumplimiento
- **EL**: Monto esperado en pesos que NAFIN perderá en esta garantía

Esta estimación de EL determina la **comisión de garantía** que NAFIN cobra.

## 🏗️ Arquitectura

El modelo utiliza un **enfoque de dos etapas**:

### Etapa 1: Modelo de Probabilidad de Default (PD)
- **Tipo**: Clasificación Binaria
- **Modelos**: Random Forest, Regresión Logística
- **Salida**: Probabilidad de que el préstamo entre en default
- **Calibración**: Regresión isotónica para precisión de probabilidades

### Etapa 2: Modelo de Pérdida Dado el Default (LGD)
- **Tipo**: Regresión (entrenado solo con préstamos en default)
- **Modelos**: XGBoost, Random Forest, Gradient Boosting
- **Salida**: Monto de pérdida en MXN si ocurre el default

### Etapa 3: Cálculo de Pérdida Esperada
- **Fórmula**: EL = PD × LGD × Factor de Calibración
- **Calibración**: Ajustes globales y por segmento
- **Segmentos**: Monto del préstamo, plazo, sector (SCIAN), estado

## 📁 Estructura del Proyecto

```
sba-mexico-model/
├── README.md
├── requirements.txt
├── train.py              # Pipeline de entrenamiento
├── quoter.py             # Calculadora de cotizaciones
├── analyze_results.py    # Análisis y visualizaciones
│
├── features.py           # Ingeniería de características
├── models.py             # Modelos PD y LGD
│
├── data/
│   └── sba_mexico_sintetico.csv    # Datos de préstamos PyME
│
├── graficos/             # Visualizaciones generadas
│   ├── 01_panorama_datos.png
│   ├── 02_modelo_pd.png
│   └── ...
│
└── sba_mexico_model.pkl  # Modelos entrenados
```

## 🚀 Inicio Rápido

### 1. Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tuusuario/sba-mexico-model.git
cd sba-mexico-model

# Crear ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Preparar Datos

Coloca el archivo de datos en el directorio:
```bash
sba_mexico_sintetico.csv
```

### 3. Entrenar Modelos

```bash
python train.py
```

**Salida esperada:**
```
============================================================
ENTRENAMIENTO - MODELO DE PÉRDIDA ESPERADA PYME MÉXICO
============================================================

[1/6] Cargando datos...
   ✓ Cargados 50,000 préstamos
   ✓ Tasa de default: 10.14%
   ✓ Pérdida promedio (defaults): $90,145.23 MXN

...

✓ ENTRENAMIENTO COMPLETADO
Modelos guardados en: sba_mexico_model.pkl
```

### 4. Generar Cotizaciones

```bash
python quoter.py
```

Ejemplo interactivo:
```
============================================================
CALCULADORA DE GARANTÍA NAFIN - CRÉDITO PYME MÉXICO
============================================================

Ingresa los datos del préstamo:

1. Monto del préstamo (MXN): $500000
2. Plazo (meses, ej: 36): 48
3. Número de empleados: 12
4. ¿Es negocio nuevo? (s/n): n
5. Código SCIAN (2 dígitos, ej: 46): 72
6. Estado (ej: JAL, CDMX, NL): JAL
7. Tasa de interés del banco (%, ej: 12.5): 14.5
8. ¿Tiene garantía inmobiliaria? (s/n): n

============================================================
COTIZACIÓN DE GARANTÍA NAFIN - CRÉDITO PYME
============================================================

--- Evaluación de Riesgo ---
Monto del Préstamo:       $500,000.00 MXN
Monto Garantizado NAFIN:  $400,000.00 MXN
Probabilidad de Default:  11.23%
Pérdida Dado Default:     $142,567.00 MXN
Pérdida Esperada:         $18,234.56 MXN
Nivel de Riesgo:          🟡 MODERADO

--- Comisión de Garantía ---
Comisión NAFIN:           $21,881.47 MXN
                          (5.47% del monto garantizado)

--- Pago Mensual ---
PAGO MENSUAL:             $14,234.56 MXN
============================================================
```

### 5. Generar Análisis Completo

```bash
python analyze_results.py
```

Esto genera visualizaciones en la carpeta `graficos/`.

## 📊 Variables del Modelo

### Variables Numéricas
- `GrAppv`: Monto bruto aprobado (MXN)
- `NAFIN_Appv`: Monto garantizado por NAFIN (MXN)
- `NAFIN_Portion`: Proporción del préstamo garantizada
- `Loan_per_Employee`: Monto por empleado
- `Term_Years`: Plazo en años
- `Debt_to_NAFIN`: Porción no garantizada
- `Log_GrAppv`: Log del monto
- `IsNewBusiness`: Negocio nuevo vs existente
- `HasRealEstate`: Tiene garantía inmobiliaria
- `InRecession`: Período de crisis (COVID)
- `IsUrban`: Ubicación urbana

### Variables Categóricas
- `SCIAN`: Código de clasificación industrial (equivalente a NAICS)
- `State`: Estado de la República Mexicana

## 🎓 Detalles del Modelo

### Criterios de Selección del Modelo PD
- **Métrica**: AUC-ROC
- **Calibración**: Regresión isotónica
- **Rendimiento típico**: AUC ~0.75-0.80

### Criterios de Selección del Modelo LGD
- **Métrica**: Error Absoluto Medio (MAE)
- **Conjunto de entrenamiento**: Solo préstamos en default
- **Rendimiento típico**: MAE ~$10,000-$20,000 MXN

### Estrategia de Calibración

1. **Calibración Global**: Ajusta el nivel general de predicción
   ```
   Factor = Pérdidas Reales Totales / Pérdidas Predichas Totales
   ```

2. **Calibración por Segmento**: Ajuste fino por características
   - Buckets de monto: <$200K, $200K-500K, $500K-1M, >$1M
   - Buckets de plazo: <2 años, 2-5 años, 5-10 años, >10 años
   - Sectores: Por código SCIAN

## 📈 Métricas de Rendimiento

### Modelo PD
- AUC-ROC (Test): Capacidad de discriminación
- Gráfico de calibración: Tasas de default predichas vs reales
- Matriz de confusión

### Modelo LGD
- MAE (Test): Error promedio en pesos
- RMSE: Raíz del error cuadrático medio
- R²: Varianza explicada

### Modelo EL General
- Total predicho vs pérdidas reales
- Precisión por segmento
- Análisis de rentabilidad de comisiones

## 🔧 Personalización

### Ajustar Margen de Comisión

En `quoter.py`, modifica el margen:

```python
# margen = 0.20 significa 20% sobre la pérdida esperada
guarantee_fee = el_pred * 1.20
```

### Ajustar Porcentajes de Garantía NAFIN

En `quoter.py`, modifica la función:

```python
def calculate_nafin_guarantee(approved_amount):
    if approved_amount <= 2_000_000:
        return approved_amount * 0.80  # 80% para préstamos pequeños
    else:
        return approved_amount * 0.70  # 70% para préstamos grandes
```

## 📚 Contexto: Programa de Garantías NAFIN

Nacional Financiera (NAFIN) proporciona garantías de crédito para reducir el riesgo de los prestamistas:

1. **El banco aprueba** un préstamo a una PyME
2. **NAFIN garantiza** 70-80% del monto del préstamo
3. **El banco cobra** intereses al prestatario
4. **NAFIN cobra** una comisión de garantía inicial
5. Si el préstamo incumple:
   - El banco intenta recuperar fondos
   - **NAFIN paga** la porción garantizada de la pérdida neta
   - Esta es la **pérdida real** que NAFIN incurre

### Por qué Importa este Modelo

- **Gestión de Riesgo**: NAFIN necesita entender las pérdidas esperadas
- **Fijación de Comisiones**: Las comisiones deben cubrir pérdidas esperadas + margen
- **Sostenibilidad del Programa**: Precios adecuados aseguran viabilidad a largo plazo
- **Análisis de Cartera**: Identificar segmentos de alto riesgo

## 🇲🇽 Adaptación al Mercado Mexicano

Este modelo fue adaptado del modelo SBA de Estados Unidos con las siguientes consideraciones:

- **SCIAN en lugar de NAICS**: Compatible a nivel de 2 dígitos
- **NAFIN_Appv en lugar de SBA_Appv**: Garantía de Nacional Financiera
- **Estados mexicanos**: 32 entidades federativas
- **Moneda**: Todos los montos en MXN
- **Tasas de default calibradas**: Basadas en datos del IMOR de CNBV
- **Períodos de crisis**: Incluye COVID-19 (2020-2021)

## 📝 Fuentes de Datos para Calibración

- **CNBV**: Índice de Morosidad (IMOR) por sector
- **INEGI**: Censos Económicos, distribución de empresas por estado
- **ENAFIN**: Encuesta Nacional de Financiamiento de las Empresas
- **Banxico**: Indicadores de crédito PyME

## 📄 Licencia

MIT License - Ver archivo LICENSE para detalles.
