"""
PD and LGD Models for SBA Mexico Expected Loss
Modelos de Probabilidad de Default y Pérdida Dado el Default

Adaptado para el mercado de créditos PyME en México con garantías NAFIN.
"""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression


try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠ XGBoost no instalado. Usando alternativas de sklearn.")


def train_pd_model(X_train, y_train, model_type='xgboost'):
    """
    Train Probability of Default (PD) model
    Entrena modelo de Probabilidad de Incumplimiento
    
    Parámetros:
    -----------
    X_train : DataFrame/array con features de entrenamiento
    y_train : Series/array con variable objetivo (0/1)
    model_type : str, tipo de modelo ('random_forest', 'logistic', 'xgboost')
    
    Retorna:
    --------
    Modelo calibrado listo para predicción
    """
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=50,
            min_samples_split=100,
            class_weight='balanced',  # Importante para datos desbalanceados
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'logistic':
        model = LogisticRegression(
            C=0.1,
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'xgboost' and HAS_XGBOOST:

        model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.02,
        max_depth=4,
        min_child_weight=5,
        gamma=0.2,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_alpha=1.0,
        reg_lambda=2.0,
        scale_pos_weight=3,
        max_delta_step=1,
        tree_method="hist",
        eval_metric="auc",
        random_state=2024)

        
    else:
        # Default a Random Forest si XGBoost no está disponible
        model = RandomForestClassifier(n_estimators=2500,
                                       max_depth=18,
                                       min_samples_split=5,
                                       min_samples_leaf=2,
                                       max_features="sqrt",
                                       bootstrap=True,
                                       class_weight="balanced",
                                       n_jobs=-1,
                                       random_state=2024)
    
    # Entrenar modelo base
    model.fit(X_train, y_train)
    
    # Calibrar probabilidades con Isotonic Regression
    # Esto mejora la precisión de las probabilidades predichas
    calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    calibrated.fit(X_train, y_train)
    
    return calibrated


def train_lgd_model(X_train, y_train, model_type='xgboost'):
    """
    Train Loss Given Default (LGD) model
    Entrena modelo de Pérdida Dado el Default
    
    IMPORTANTE: Este modelo se entrena SOLO con préstamos que hicieron default
    
    Parámetros:
    -----------
    X_train : DataFrame/array con features de préstamos en default
    y_train : Series/array con pérdidas reales (ChgOffPrinGr en MXN)
    model_type : str, tipo de modelo ('xgboost', 'random_forest', 'gradient_boosting')
    
    Retorna:
    --------
    Modelo de regresión entrenado
    """
    
    if model_type == 'xgboost' and HAS_XGBOOST:
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=20,
            min_samples_split=50,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=42
        )
    else:
        # Default a Random Forest si XGBoost no está disponible
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        )
    
    model.fit(X_train, y_train)
    return model


def calculate_expected_loss(pd_pred, lgd_pred, calibration_factor=1.0):
    """
    Calculate Expected Loss: EL = PD × LGD × Calibration Factor
    Calcula la Pérdida Esperada
    
    Fórmula: EL = Probabilidad de Default × Pérdida Dado el Default × Factor de Calibración
    
    Parámetros:
    -----------
    pd_pred : array con probabilidades de default (0 a 1)
    lgd_pred : array con pérdidas predichas dado el default (en MXN)
    calibration_factor : float, factor de ajuste global
    
    Retorna:
    --------
    array con pérdidas esperadas en MXN
    """
    return pd_pred * lgd_pred * calibration_factor


def calculate_calibration_factor(y_loss_actual, el_predicted):
    """
    Calculate calibration factor to adjust predictions
    Calcula el factor de calibración para ajustar predicciones
    
    El factor de calibración asegura que la suma de pérdidas predichas
    sea igual a la suma de pérdidas reales en el conjunto de entrenamiento.
    
    Parámetros:
    -----------
    y_loss_actual : array con pérdidas reales
    el_predicted : array con pérdidas esperadas predichas (PD × LGD)
    
    Retorna:
    --------
    float, factor de calibración
    """
    total_actual = y_loss_actual.sum()
    total_predicted = el_predicted.sum()
    
    if total_predicted > 0:
        return total_actual / total_predicted
    else:
        return 1.0


def calculate_segment_calibration(df, y_loss, el_pred, segment_col):
    """
    Calculate calibration factors by segment
    Calcula factores de calibración por segmento
    
    Parámetros:
    -----------
    df : DataFrame con datos
    y_loss : pérdidas reales
    el_pred : pérdidas predichas
    segment_col : nombre de columna para segmentación
    
    Retorna:
    --------
    dict con factores por segmento
    """
    df_temp = df.copy()
    df_temp['actual_loss'] = y_loss.values
    df_temp['predicted_el'] = el_pred
    
    segment_factors = {}
    for segment in df_temp[segment_col].unique():
        mask = df_temp[segment_col] == segment
        actual_sum = df_temp.loc[mask, 'actual_loss'].sum()
        pred_sum = df_temp.loc[mask, 'predicted_el'].sum()
        
        if pred_sum > 0:
            segment_factors[segment] = actual_sum / pred_sum
        else:
            segment_factors[segment] = 1.0
    
    return segment_factors


# =========================================
# MÉTRICAS DE EVALUACIÓN
# =========================================

def evaluate_pd_model(y_true, y_pred_proba):
    """
    Evaluate PD model performance
    Evalúa el rendimiento del modelo de PD
    """
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
    
    metrics = {
        'AUC-ROC': roc_auc_score(y_true, y_pred_proba),
        'Brier Score': brier_score_loss(y_true, y_pred_proba),
        'Log Loss': log_loss(y_true, y_pred_proba),
        'Default Rate Actual': y_true.mean(),
        'Default Rate Predicted': y_pred_proba.mean(),
    }
    
    return metrics


def evaluate_lgd_model(y_true, y_pred):
    """
    Evaluate LGD model performance
    Evalúa el rendimiento del modelo de LGD
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'Mean Actual': y_true.mean(),
        'Mean Predicted': y_pred.mean(),
    }
    
    return metrics
