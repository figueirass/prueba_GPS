"""
Train SBA Mexico Expected Loss Models
Entrena los modelos de Pérdida Esperada para créditos PyME en México

Este script entrena los modelos de PD (Probabilidad de Default) y 
LGD (Pérdida Dado el Default) y los guarda para uso en producción.
"""

import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_absolute_error

from features import prepare_data, create_preprocessor, transform_data
from models import train_pd_model, train_lgd_model, calculate_calibration_factor


def main():
    print("="*60)
    print("ENTRENAMIENTO - MODELO DE PÉRDIDA ESPERADA PYME MÉXICO")
    print("="*60)
    
    # =========================================
    # 1. CARGAR DATOS
    # =========================================
    print("\n[1/6] Cargando datos...")
    
    # Buscar el archivo de datos
    data_files = ['sba_mexico_sintetico.csv', 'data/sba_mexico_sintetico.csv', 
                  '../sba_mexico_sintetico.csv', 'SBAcase.csv']
    
    filepath = None
    for f in data_files:
        if os.path.exists(f):
            filepath = f
            break
    
    if filepath is None:
        print("❌ Error: No se encontró el archivo de datos.")
        print("   Coloca 'sba_mexico_sintetico.csv' en el directorio actual.")
        return
    
    X, y_pd, y_loss, df = prepare_data(filepath)
    
    print(f"   ✓ Cargados {len(X):,} préstamos")
    print(f"   ✓ Tasa de default: {y_pd.mean()*100:.2f}%")
    print(f"   ✓ Pérdida promedio (defaults): ${y_loss[y_loss > 0].mean():,.2f} MXN")
    print(f"   ✓ Pérdida total: ${y_loss.sum():,.2f} MXN")
    
    # =========================================
    # 2. DIVIDIR DATOS
    # =========================================
    print("\n[2/6] Dividiendo datos en train/test...")
    
    X_train, X_test, y_pd_train, y_pd_test, y_loss_train, y_loss_test = train_test_split(
        X, y_pd, y_loss, 
        test_size=0.20, 
        random_state=42, 
        stratify=y_pd  # Mantener proporción de defaults
    )
    
    print(f"   ✓ Entrenamiento: {len(X_train):,} préstamos ({y_pd_train.mean()*100:.2f}% defaults)")
    print(f"   ✓ Prueba: {len(X_test):,} préstamos ({y_pd_test.mean()*100:.2f}% defaults)")
    
    # =========================================
    # 3. PREPROCESAMIENTO
    # =========================================
    print("\n[3/6] Preprocesando datos...")
    
    preprocessor = create_preprocessor()
    preprocessor.fit(X_train)
    
    X_train_t = transform_data(preprocessor, X_train)
    X_test_t = transform_data(preprocessor, X_test)
    
    print(f"   ✓ Features originales: {X_train.shape[1]}")
    print(f"   ✓ Features después de transformación: {X_train_t.shape[1]}")
    
    # =========================================
    # 4. ENTRENAR MODELO PD
    # =========================================
    print("\n[4/6] Entrenando modelo de Probabilidad de Default (PD)...")
    
    pd_model = train_pd_model(X_train_t, y_pd_train, model_type='xgboost')
    
    # Evaluar en test
    pd_pred_test = pd_model.predict_proba(X_test_t)[:, 1]
    pd_auc = roc_auc_score(y_pd_test, pd_pred_test)
    
    print(f"   ✓ Modelo PD entrenado")
    print(f"   ✓ AUC-ROC en test: {pd_auc:.4f}")
    
    # =========================================
    # 5. ENTRENAR MODELO LGD
    # =========================================
    print("\n[5/6] Entrenando modelo de Pérdida Dado el Default (LGD)...")
    
    # Filtrar solo préstamos en default para entrenar LGD
    defaults_mask_train = y_pd_train == 1
    X_defaults = X_train_t[defaults_mask_train]
    y_loss_defaults = y_loss_train[defaults_mask_train]
    
    print(f"   ✓ Préstamos en default para entrenamiento: {len(X_defaults):,}")
    
    lgd_model = train_lgd_model(X_defaults, y_loss_defaults, model_type='randomforest')
    
    # Evaluar en test (solo defaults)
    defaults_mask_test = y_pd_test == 1
    X_test_defaults = X_test_t[defaults_mask_test]
    y_loss_test_defaults = y_loss_test[defaults_mask_test]
    
    if len(X_test_defaults) > 0:
        lgd_pred_test = lgd_model.predict(X_test_defaults)
        lgd_mae = mean_absolute_error(y_loss_test_defaults, lgd_pred_test)
        print(f"   ✓ Modelo LGD entrenado")
        print(f"   ✓ MAE en test (defaults): ${lgd_mae:,.2f} MXN")
    else:
        print("   ✓ Modelo LGD entrenado")
        lgd_mae = 0
    
    # =========================================
    # 6. CALCULAR FACTOR DE CALIBRACIÓN
    # =========================================
    print("\n[6/6] Calculando factor de calibración...")
    
    # Predicciones en train para calibración
    pd_pred_train = pd_model.predict_proba(X_train_t)[:, 1]
    lgd_pred_train = lgd_model.predict(X_train_t)
    el_pred_train = pd_pred_train * lgd_pred_train
    
    calibration_factor = calculate_calibration_factor(y_loss_train, el_pred_train)
    
    print(f"   ✓ Factor de calibración: {calibration_factor:.4f}")
    
    # Verificar calibración
    el_calibrated = el_pred_train * calibration_factor
    print(f"   ✓ Pérdida total real (train): ${y_loss_train.sum():,.2f} MXN")
    print(f"   ✓ Pérdida total predicha (calibrada): ${el_calibrated.sum():,.2f} MXN")
    
    # =========================================
    # GUARDAR MODELOS
    # =========================================
    print("\n" + "-"*60)
    print("Guardando modelos...")
    
    artifacts = {
        'preprocessor': preprocessor,
        'pd_model': pd_model,
        'lgd_model': lgd_model,
        'calibration_factor': calibration_factor,
        'metadata': {
            'n_train': len(X_train),
            'n_test': len(X_test),
            'default_rate': y_pd.mean(),
            'pd_auc': pd_auc,
            'lgd_mae': lgd_mae,
            'total_loss': y_loss.sum(),
            'currency': 'MXN',
            'country': 'Mexico',
        }
    }
    
    # Guardar en archivo pickle
    model_path = 'sba_mexico_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(artifacts, f)
    
    # También guardar con nombre genérico para compatibilidad
    with open('sba_model.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    
    # =========================================
    # RESUMEN FINAL
    # =========================================
    print("\n" + "="*60)
    print("✓ ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print(f"\nModelos guardados en: {model_path}")
    print(f"\nResumen de rendimiento:")
    print(f"  • PD Model AUC-ROC: {pd_auc:.4f}")
    print(f"  • LGD Model MAE: ${lgd_mae:,.2f} MXN")
    print(f"  • Factor de calibración: {calibration_factor:.4f}")
    print("\nPara generar cotizaciones, ejecuta: python quoter.py")
    print("Para análisis completo, ejecuta: python analyze_results.py")
    print("="*60)


if __name__ == "__main__":
    main()
