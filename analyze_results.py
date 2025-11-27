"""
SBA Mexico Expected Loss Model - Complete Analysis with Visualizations
Análisis Completo del Modelo de Pérdida Esperada para PyME México

Ejecuta este script después de entrenar para generar todos los gráficos
de rendimiento y análisis para tu reporte.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import mean_absolute_error, r2_score

from features import prepare_data, create_preprocessor, transform_data, SECTORES_SCIAN, ESTADOS_MEXICO
from models import train_pd_model, train_lgd_model

# Configuración de estilo
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Configurar para mostrar pesos en español
import locale
try:
    locale.setlocale(locale.LC_ALL, 'es_MX.UTF-8')
except:
    pass

print("="*60)
print("MODELO DE PÉRDIDA ESPERADA PYME MÉXICO - ANÁLISIS COMPLETO")
print("="*60)

# Buscar archivo de datos
data_files = ['sba_mexico_sintetico.csv', 'data/sba_mexico_sintetico.csv', 
              '../sba_mexico_sintetico.csv']
filepath = None
for f in data_files:
    if os.path.exists(f):
        filepath = f
        break

if filepath is None:
    print("❌ No se encontró el archivo de datos.")
    print("   Coloca 'sba_mexico_sintetico.csv' en el directorio actual.")
    exit(1)

# ============================================
# CARGAR DATOS
# ============================================
print("\n[1/8] Cargando datos...")
X, y_pd, y_loss, df = prepare_data(filepath)
print(f"   Dataset: {len(df):,} préstamos")
print(f"   Tasa de default: {y_pd.mean()*100:.2f}%")
print(f"   Pérdida total: ${y_loss.sum():,.0f} MXN")

# Dividir datos
print("\n[2/8] Dividiendo datos...")
X_train, X_test, y_pd_train, y_pd_test, y_loss_train, y_loss_test = train_test_split(
    X, y_pd, y_loss, test_size=0.20, random_state=42, stratify=y_pd
)

# Preprocesar
print("\n[3/8] Preprocesando...")
preprocessor = create_preprocessor()
preprocessor.fit(X_train)
X_train_t = transform_data(preprocessor, X_train)
X_test_t = transform_data(preprocessor, X_test)

# Cargar o entrenar modelos
print("\n[4/8] Cargando/entrenando modelos...")
try:
    model_files = ['sba_mexico_model.pkl', 'sba_model.pkl']
    artifacts = None
    for mf in model_files:
        if os.path.exists(mf):
            with open(mf, 'rb') as f:
                artifacts = pickle.load(f)
            break
    
    if artifacts is None:
        raise FileNotFoundError
    
    pd_model = artifacts['pd_model']
    lgd_model = artifacts['lgd_model']
    calibration_factor = artifacts['calibration_factor']
    print("   ✓ Modelos cargados exitosamente")
except:
    print("   Entrenando nuevos modelos...")
    pd_model = train_pd_model(X_train_t, y_pd_train)
    
    train_defaults = y_pd_train == 1
    lgd_model = train_lgd_model(X_train_t[train_defaults], y_loss_train[train_defaults])
    
    pd_pred = pd_model.predict_proba(X_train_t)[:, 1]
    lgd_pred = lgd_model.predict(X_train_t)
    el_pred = pd_pred * lgd_pred
    calibration_factor = y_loss_train.sum() / el_pred.sum()
    print("   ✓ Modelos entrenados")

# Hacer predicciones
print("\n[5/8] Generando predicciones...")
pd_test = pd_model.predict_proba(X_test_t)[:, 1]
lgd_test = lgd_model.predict(X_test_t)
el_test = pd_test * lgd_test * calibration_factor

# ============================================
# VISUALIZACIONES
# ============================================

print("\n[6/8] Creando visualizaciones...")

# Crear directorio para imágenes si no existe
os.makedirs('graficos', exist_ok=True)

# ============================================
# FIGURA 1: PANORAMA DE LOS DATOS
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Panorama del Dataset de Créditos PyME México', fontsize=14, fontweight='bold')

# Default rate por monto
df['monto_bucket'] = pd.cut(df['GrAppv'], 
                            bins=[0, 200000, 500000, 1000000, float('inf')],
                            labels=['<$200K', '$200K-500K', '$500K-1M', '>$1M'])
default_by_amount = df.groupby('monto_bucket', observed=True).agg({
    'Default': 'mean'
}) * 100

colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
axes[0, 0].bar(range(len(default_by_amount)), default_by_amount['Default'], 
               color=colors, edgecolor='black')
axes[0, 0].set_xticks(range(len(default_by_amount)))
axes[0, 0].set_xticklabels(default_by_amount.index, rotation=45)
axes[0, 0].set_ylabel('Tasa de Default (%)')
axes[0, 0].set_title('Tasa de Default por Monto del Préstamo', fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)

# Negocio nuevo vs existente
business_stats = df.groupby('IsNewBusiness', observed=True).agg({
    'Default': 'mean'
}) * 100
business_stats.index = ['Existente', 'Nuevo']

axes[0, 1].bar(range(len(business_stats)), business_stats['Default'],
               color=['#27ae60', '#c0392b'], edgecolor='black')
axes[0, 1].set_xticks(range(len(business_stats)))
axes[0, 1].set_xticklabels(business_stats.index, rotation=0)
axes[0, 1].set_ylabel('Tasa de Default (%)')
axes[0, 1].set_title('Negocio Nuevo vs Existente', fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)

# Top sectores con mayor riesgo
df['SCIAN_2d'] = df['SCIAN'].astype(str).str[:2]
industry_stats = df.groupby('SCIAN_2d', observed=True).agg({
    'Default': [('count', 'size'), ('default_rate', 'mean')]
})
industry_stats.columns = ['count', 'default_rate']
industry_stats['default_rate'] *= 100
industry_stats = industry_stats[industry_stats['count'] >= 100]
top_industries = industry_stats.nlargest(8, 'default_rate')

# Agregar nombres de sector
sector_names = [SECTORES_SCIAN.get(idx, idx)[:20] for idx in top_industries.index]

axes[1, 0].barh(range(len(top_industries)), top_industries['default_rate'],
                color='coral', edgecolor='black')
axes[1, 0].set_yticks(range(len(top_industries)))
axes[1, 0].set_yticklabels(sector_names)
axes[1, 0].set_xlabel('Tasa de Default (%)')
axes[1, 0].set_title('Top 8 Sectores con Mayor Riesgo (SCIAN)', fontweight='bold')
axes[1, 0].grid(axis='x', alpha=0.3)
axes[1, 0].invert_yaxis()

# Distribución de montos
axes[1, 1].hist(df['GrAppv']/1e6, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[1, 1].axvline(df['GrAppv'].mean()/1e6, color='red', linestyle='--', 
                   label=f"Promedio: ${df['GrAppv'].mean()/1e6:.2f}M")
axes[1, 1].set_xlabel('Monto del Préstamo (Millones MXN)')
axes[1, 1].set_ylabel('Frecuencia')
axes[1, 1].set_title('Distribución de Montos de Préstamo', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].set_xlim(0, 2)

plt.tight_layout()
plt.savefig('graficos/01_panorama_datos.png', dpi=300, bbox_inches='tight')
print("   ✓ Guardado: graficos/01_panorama_datos.png")
plt.close()

# ============================================
# FIGURA 2: RENDIMIENTO DEL MODELO PD
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Rendimiento del Modelo de Probabilidad de Default (PD)', fontsize=14, fontweight='bold')

# Curva ROC
fpr, tpr, _ = roc_curve(y_pd_test, pd_test)
auc = roc_auc_score(y_pd_test, pd_test)

axes[0, 0].plot(fpr, tpr, linewidth=2, color='#3498db', label=f'AUC = {auc:.4f}')
axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Aleatorio')
axes[0, 0].fill_between(fpr, tpr, alpha=0.2, color='#3498db')
axes[0, 0].set_xlabel('Tasa de Falsos Positivos')
axes[0, 0].set_ylabel('Tasa de Verdaderos Positivos')
axes[0, 0].set_title('Curva ROC - Modelo PD', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Matriz de confusión
cm = confusion_matrix(y_pd_test, (pd_test >= 0.5).astype(int))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1], square=True,
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'])
axes[0, 1].set_xlabel('Predicción')
axes[0, 1].set_ylabel('Real')
axes[0, 1].set_title('Matriz de Confusión', fontweight='bold')

# Distribución de PD
axes[1, 0].hist(pd_test, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
axes[1, 0].axvline(pd_test.mean(), color='red', linestyle='--',
                   label=f'Promedio: {pd_test.mean():.2%}')
axes[1, 0].set_xlabel('Probabilidad de Default Predicha')
axes[1, 0].set_ylabel('Frecuencia')
axes[1, 0].set_title('Distribución de Predicciones PD', fontweight='bold')
axes[1, 0].legend()

# PD por resultado real
axes[1, 1].hist(pd_test[y_pd_test == 0], bins=30, alpha=0.6, 
                label='Sin Default', color='green', edgecolor='black')
axes[1, 1].hist(pd_test[y_pd_test == 1], bins=30, alpha=0.6,
                label='Con Default', color='red', edgecolor='black')
axes[1, 1].set_xlabel('PD Predicha')
axes[1, 1].set_ylabel('Frecuencia')
axes[1, 1].set_title('Distribución de PD por Resultado Real', fontweight='bold')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('graficos/02_modelo_pd.png', dpi=300, bbox_inches='tight')
print("   ✓ Guardado: graficos/02_modelo_pd.png")
plt.close()

# ============================================
# FIGURA 3: RENDIMIENTO DEL MODELO LGD
# ============================================
test_defaults = y_pd_test == 1
lgd_test_defaults = lgd_test[test_defaults]
y_loss_test_defaults = y_loss_test[test_defaults]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Rendimiento del Modelo de Pérdida Dado el Default (LGD)', fontsize=14, fontweight='bold')

# Predicho vs Real
axes[0, 0].scatter(lgd_test_defaults/1e6, y_loss_test_defaults/1e6, alpha=0.5, s=30, color='steelblue')
max_val = max(lgd_test_defaults.max(), y_loss_test_defaults.max())/1e6
axes[0, 0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
axes[0, 0].set_xlabel('Pérdida Predicha (Millones MXN)')
axes[0, 0].set_ylabel('Pérdida Real (Millones MXN)')
axes[0, 0].set_title('Pérdida Predicha vs Real', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Residuales
residuals = y_loss_test_defaults.values - lgd_test_defaults
axes[0, 1].scatter(lgd_test_defaults/1e6, residuals/1e6, alpha=0.5, s=30, color='purple')
axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Pérdida Predicha (Millones MXN)')
axes[0, 1].set_ylabel('Residual (Millones MXN)')
axes[0, 1].set_title('Gráfico de Residuales', fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# Distribución
axes[1, 0].hist(y_loss_test_defaults/1e6, bins=50, alpha=0.6, 
                label='Real', color='blue', edgecolor='black')
axes[1, 0].hist(lgd_test_defaults/1e6, bins=50, alpha=0.6,
                label='Predicha', color='orange', edgecolor='black')
axes[1, 0].set_xlabel('Pérdida (Millones MXN)')
axes[1, 0].set_ylabel('Frecuencia')
axes[1, 0].set_title('Distribución de Pérdidas Real vs Predicha', fontweight='bold')
axes[1, 0].legend()

# Métricas
mae = mean_absolute_error(y_loss_test_defaults, lgd_test_defaults)
r2 = r2_score(y_loss_test_defaults, lgd_test_defaults)
metrics_text = f"""
MAE: ${mae:,.0f} MXN
R²: {r2:.4f}

Promedio Real: ${y_loss_test_defaults.mean():,.0f} MXN
Promedio Predicho: ${lgd_test_defaults.mean():,.0f} MXN
"""
axes[1, 1].text(0.5, 0.5, metrics_text, fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                family='monospace')
axes[1, 1].set_title('Métricas del Modelo LGD', fontweight='bold')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('graficos/03_modelo_lgd.png', dpi=300, bbox_inches='tight')
print("   ✓ Guardado: graficos/03_modelo_lgd.png")
plt.close()

# ============================================
# FIGURA 4: ANÁLISIS DE PÉRDIDA ESPERADA
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Análisis de Pérdida Esperada (EL)', fontsize=14, fontweight='bold')

# Distribución de EL
axes[0, 0].hist(el_test/1e6, bins=50, edgecolor='black', alpha=0.7, color='purple')
axes[0, 0].axvline(el_test.mean()/1e6, color='red', linestyle='--',
                   label=f'Promedio: ${el_test.mean()/1e6:.3f}M')
axes[0, 0].set_xlabel('Pérdida Esperada (Millones MXN)')
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].set_title('Distribución de Pérdida Esperada', fontweight='bold')
axes[0, 0].legend()

# EL vs Pérdida Real
axes[0, 1].scatter(el_test/1e6, y_loss_test/1e6, alpha=0.3, s=20, color='teal')
max_val = max(el_test.max(), y_loss_test.max())/1e6
axes[0, 1].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfecta')
axes[0, 1].set_xlabel('EL Predicha (Millones MXN)')
axes[0, 1].set_ylabel('Pérdida Real (Millones MXN)')
axes[0, 1].set_title('EL Predicha vs Pérdida Real', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Pérdida acumulada
sorted_idx = np.argsort(el_test)[::-1]
cumsum_pred = np.cumsum(el_test[sorted_idx])
cumsum_actual = np.cumsum(y_loss_test.values[sorted_idx])
percentiles = np.arange(1, len(sorted_idx) + 1) / len(sorted_idx) * 100

axes[1, 0].plot(percentiles, cumsum_pred/1e6, label='EL Predicha', linewidth=2, color='orange')
axes[1, 0].plot(percentiles, cumsum_actual/1e6, label='Pérdida Real', linewidth=2, color='blue')
axes[1, 0].set_xlabel('Percentil de Préstamos (ordenados por EL)')
axes[1, 0].set_ylabel('Pérdida Acumulada (Millones MXN)')
axes[1, 0].set_title('Distribución de Pérdida Acumulada', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# EL por bucket de monto
test_df = X_test.copy()
test_df['predicted_el'] = el_test
test_df['actual_loss'] = y_loss_test.values
test_df['bucket'] = pd.cut(test_df['GrAppv'],
                            bins=[0, 200000, 500000, 1000000, float('inf')],
                            labels=['<$200K', '$200K-500K', '$500K-1M', '>$1M'])

bucket_stats = test_df.groupby('bucket', observed=True).agg({'predicted_el': 'sum', 'actual_loss': 'sum'})

x_pos = np.arange(len(bucket_stats))
width = 0.35
axes[1, 1].bar(x_pos - width/2, bucket_stats['predicted_el']/1e6, width, label='Predicha', color='orange')
axes[1, 1].bar(x_pos + width/2, bucket_stats['actual_loss']/1e6, width, label='Real', color='blue')
axes[1, 1].set_xlabel('Monto del Préstamo')
axes[1, 1].set_ylabel('Pérdida Total (Millones MXN)')
axes[1, 1].set_title('Pérdida por Segmento de Monto', fontweight='bold')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(bucket_stats.index, rotation=45)
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('graficos/04_perdida_esperada.png', dpi=300, bbox_inches='tight')
print("   ✓ Guardado: graficos/04_perdida_esperada.png")
plt.close()

# ============================================
# FIGURA 5: SEGMENTACIÓN DE RIESGO
# ============================================
test_df['risk_score'] = el_test
try:
    test_df['risk_category'] = pd.qcut(test_df['risk_score'], q=5,
                                        labels=['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto'],
                                        duplicates='drop')
except ValueError:
    test_df['risk_category'] = pd.qcut(test_df['risk_score'], q=3,
                                        labels=['Bajo', 'Medio', 'Alto'],
                                        duplicates='drop')

risk_summary = test_df.groupby('risk_category', observed=True).agg({
    'GrAppv': 'count',
    'predicted_el': 'mean',
    'actual_loss': lambda x: (x > 0).mean()
})
risk_summary.columns = ['Cantidad', 'EL Promedio', 'Tasa Default']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Segmentación de Riesgo Crediticio', fontsize=14, fontweight='bold')

colors_risk = ['#27ae60', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c'][:len(risk_summary)]

# Cantidad por categoría
risk_summary['Cantidad'].plot(kind='bar', ax=axes[0, 0], color=colors_risk, edgecolor='black')
axes[0, 0].set_title('Cantidad de Préstamos por Categoría de Riesgo', fontweight='bold')
axes[0, 0].set_ylabel('Número de Préstamos')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(axis='y', alpha=0.3)

# EL promedio
risk_summary['EL Promedio'].plot(kind='bar', ax=axes[0, 1], color=colors_risk, edgecolor='black')
axes[0, 1].set_title('Pérdida Esperada Promedio por Categoría', fontweight='bold')
axes[0, 1].set_ylabel('EL Promedio (MXN)')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(axis='y', alpha=0.3)

# Tasa de default
(risk_summary['Tasa Default'] * 100).plot(kind='bar', ax=axes[1, 0], color=colors_risk, edgecolor='black')
axes[1, 0].set_title('Tasa de Default Real por Categoría', fontweight='bold')
axes[1, 0].set_ylabel('Tasa de Default (%)')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(axis='y', alpha=0.3)

# Tabla resumen
table_data = risk_summary.copy()
table_data['EL Promedio'] = table_data['EL Promedio'].apply(lambda x: f'${x:,.0f}')
table_data['Tasa Default'] = table_data['Tasa Default'].apply(lambda x: f'{x*100:.1f}%')
table_data = table_data.reset_index().values

axes[1, 1].axis('tight')
axes[1, 1].axis('off')
table = axes[1, 1].table(cellText=table_data, 
                         colLabels=['Categoría', 'Cantidad', 'EL Promedio', 'Tasa Default'],
                         cellLoc='center', loc='center',
                         colColours=['#3498db']*4)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
axes[1, 1].set_title('Resumen de Segmentación de Riesgo', fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('graficos/05_segmentacion_riesgo.png', dpi=300, bbox_inches='tight')
print("   ✓ Guardado: graficos/05_segmentacion_riesgo.png")
plt.close()

# ============================================
# FIGURA 6: ANÁLISIS POR ESTADO (MÉXICO)
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Análisis de Riesgo por Estado de la República', fontsize=14, fontweight='bold')

# Default por estado (top 15)
state_stats = df.groupby('State', observed=True).agg({
    'Default': ['count', 'mean']
})
state_stats.columns = ['count', 'default_rate']
state_stats['default_rate'] *= 100
state_stats = state_stats[state_stats['count'] >= 50]
state_stats = state_stats.sort_values('count', ascending=False).head(15)

# Agregar nombres de estado
state_names = [ESTADOS_MEXICO.get(idx, idx)[:15] for idx in state_stats.index]

axes[0].bar(range(len(state_stats)), state_stats['default_rate'], color='steelblue', edgecolor='black')
axes[0].axhline(y=y_pd.mean()*100, color='red', linestyle='--', label='Promedio Nacional')
axes[0].set_xticks(range(len(state_stats)))
axes[0].set_xticklabels(state_names, rotation=45, ha='right')
axes[0].set_ylabel('Tasa de Default (%)')
axes[0].set_title('Tasa de Default por Estado (Top 15)', fontweight='bold')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Monto promedio por estado
state_amount = df.groupby('State', observed=True).agg({
    'GrAppv': 'mean'
}).loc[state_stats.index]

axes[1].bar(range(len(state_amount)), state_amount['GrAppv']/1e6, color='coral', edgecolor='black')
axes[1].set_xticks(range(len(state_amount)))
axes[1].set_xticklabels(state_names, rotation=45, ha='right')
axes[1].set_ylabel('Monto Promedio (Millones MXN)')
axes[1].set_title('Monto Promedio de Préstamo por Estado', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('graficos/06_analisis_estados.png', dpi=300, bbox_inches='tight')
print("   ✓ Guardado: graficos/06_analisis_estados.png")
plt.close()

# ============================================
# RESUMEN FINAL
# ============================================
print("\n[7/8] Resumen de Rendimiento del Modelo")
print("="*60)
print(f"  • AUC-ROC del Modelo PD: {auc:.4f}")
print(f"  • MAE del Modelo LGD: ${mae:,.2f} MXN")
print(f"  • R² del Modelo LGD: {r2:.4f}")
print(f"  • Factor de Calibración: {calibration_factor:.4f}")
print(f"  • Pérdida Total Predicha: ${el_test.sum():,.2f} MXN")
print(f"  • Pérdida Total Real: ${y_loss_test.sum():,.2f} MXN")
ratio = el_test.sum() / y_loss_test.sum() if y_loss_test.sum() > 0 else 0
print(f"  • Ratio Predicho/Real: {ratio:.4f}")

print("\n[8/8] ¡Análisis completado!")
print("="*60)
print("\nArchivos generados en carpeta 'graficos/':")
print("  01_panorama_datos.png")
print("  02_modelo_pd.png")
print("  03_modelo_lgd.png")
print("  04_perdida_esperada.png")
print("  05_segmentacion_riesgo.png")
print("  06_analisis_estados.png")
print("\n✓ Todas las visualizaciones guardadas exitosamente!")
