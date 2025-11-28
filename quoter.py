"""
Calculadora de Garantía de Crédito PyME México
SBA Mexico Loan Guarantee Fee Calculator

Implementación de la lógica de Garantía Premium Select (GPS).
La garantía interna de la SOFOM se ajusta según el PD predicho. 
"""

import pickle
import os
import numpy as np
import pandas as pd
from features import create_preprocessor, transform_data, SECTORES_SCIAN, ESTADOS_MEXICO

# --- NUEVA FUNCIÓN GPS ---
def apply_premium_select_guarantee(pd_pred, approved_amount):
    """
    Asigna la categoría de Garantía Premium Select (GPS) y define la garantía interna
    de la SOFOM (en el contexto del proyecto original, esta sería la garantía interna).

    La Garantía FINTECH (externa) se mantiene sin cambios, pero la PD define la categoría.
    
    Retorna: diccionario con categoria, garantia_interna_soform (en %), y accion.
    """
    if pd_pred < 0.01:
        # Menos de 1% PD
        return {
            'category': 'Ultra–Oro',
            'soform_guarantee_pct': 0.85, 
            'action': 'Aprobación Rápida con Beneficio',
            'max_guarantee_limit': approved_amount * 0.8 
        }
    elif pd_pred < 0.03:
        # Entre 1% y 3% PD
        return {
            'category': 'Oro',
            'soform_guarantee_pct': 0.70, # 70%
            'action': 'Aprobación con Condiciones Preferenciales',
            'max_guarantee_limit': approved_amount * 0.75
        }
    else:
        # 9% PD o más
        # 3% PD o más
        return {
            'category': 'Rechazo (Riesgo Alto)',
            'soform_guarantee_pct': 0.00, # 0%
            'action': 'Rechazo por Política de Riesgo',
            'max_guarantee_limit': 0.0
        }
# -------------------------

def calculate_nafin_guarantee(approved_amount):
    """
    Calculate FINTECH guarantee amount based on loan size
    Calcula el monto de garantía FINTECH según el tamaño del préstamo
    """
    if approved_amount <= 2_000_000:
        return approved_amount * 0.80
    else:
        return approved_amount * 0.70

# El resto de las funciones (calculate_monthly_payment, create_loan_features,
# load_models, show_scian_codes, show_state_codes, main) se mantienen casi igual,
# pero se modifica `calculate_quote` y `print_quote`.

def calculate_monthly_payment(principal, annual_rate, term_months):
    """
    Calculate monthly loan payment (amortización francesa)
    Calcula el pago mensual del préstamo
    """
    if annual_rate == 0:
        return principal / term_months

    monthly_rate = (annual_rate / 100) / 12
    payment = principal * (monthly_rate * (1 + monthly_rate)**term_months) / \
              ((1 + monthly_rate)**term_months - 1)
    return payment


def create_loan_features(approved_amount, term_months, num_employees, 
                         is_new_business, scian_code, state_code):
    """
    Create feature vector for a loan
    Crea el vector de features para un préstamo
    """
    nafin_guaranteed = calculate_nafin_guarantee(approved_amount)

    features = {
        'GrAppv': approved_amount,
        'NAFIN_Appv': nafin_guaranteed,
        'Term': term_months,
        'NoEmp': num_employees,
        'IsNewBusiness': 1 if is_new_business else 0,
        'NewExist': 1.0 if is_new_business else 2.0,
        'SCIAN': str(scian_code)[:2],
        'State': state_code.upper(),
        'NAFIN_Portion': nafin_guaranteed / approved_amount if approved_amount > 0 else 0,
        'Loan_per_Employee': approved_amount / (num_employees + 1),
        'Term_Years': term_months / 12.0,
        'Debt_to_NAFIN': approved_amount - nafin_guaranteed,
        'Log_GrAppv': np.log1p(approved_amount),
        'HasRealEstate': 0, # Default: sin garantía inmobiliaria
        'InRecession': 0,   # Default: no en recesión
        'IsUrban': 1,       # Default: urbano
    }

    return pd.DataFrame([features])


def load_models():
    """
    Load trained models or train if they don't exist
    Carga los modelos entrenados o entrena si no existen
    """
    model_files = ['sba_mexico_model.pkl', 'sba_model.pkl']

    for model_file in model_files:
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                return pickle.load(f)

    print("⚠ Modelos no encontrados. Entrenando ahora...")
    print("  Esto tomará unos minutos...\n")
    import train
    train.main()
    print("\n")

    # Cargar modelos recién entrenados
    with open('sba_mexico_model.pkl', 'rb') as f:
        return pickle.load(f)

# --- FUNCIÓN MODIFICADA: calculate_quote ---
# --- CONSTANTES DE MERCADO ---
TIIE_ACTUAL = 11.25  # Tasa de referencia aproximada actual en México

def calculate_quote(approved_amount, term_months, num_employees, 
                    is_new_business, scian_code, state_code, market_bank_rate,
                    has_real_estate=False, in_recession=False):
    """
    Calcula la cotización aplicando ingeniería inversa a la tasa del banco
    para reducirla proporcionalmente al riesgo cubierto por la garantía.
    """

    # Cargar modelos
    artifacts = load_models()

    # Crear features
    loan_df = create_loan_features(
        approved_amount, term_months, num_employees,
        is_new_business, scian_code, state_code
    )

    # Ajustar features adicionales
    loan_df['HasRealEstate'] = 1 if has_real_estate else 0
    loan_df['InRecession'] = 1 if in_recession else 0

    # Preprocesar y Predecir
    X_processed = transform_data(artifacts['preprocessor'], loan_df)
    pd_pred = artifacts['pd_model'].predict_proba(X_processed)[:, 1][0]
    lgd_pred = artifacts['lgd_model'].predict(X_processed)[0]
    el_pred = pd_pred * lgd_pred * artifacts['calibration_factor']

    # === APLICAR LÓGICA GPS ===
    gps_info = apply_premium_select_guarantee(pd_pred, approved_amount)
    category = gps_info['category']
    soform_guarantee_pct = gps_info['soform_guarantee_pct']  # Ej: 0.80 u 0.70

    # -------------------------------------------------------------------------
    # 1. CÁLCULO DE LA NUEVA TASA BANCARIA (INGENIERÍA INVERSA DE RIESGO)
    # -------------------------------------------------------------------------
    if category == 'Rechazo (Riesgo Alto)':
        final_bank_rate = 0.0
    else:
        original_spread = max(market_bank_rate - TIIE_ACTUAL, 2.0)

        risk_premium_component = original_spread * 0.65
        operational_component = original_spread * 0.35

        risk_reduction_factor = soform_guarantee_pct
        new_risk_premium = risk_premium_component * (1 - risk_reduction_factor)

        calculated_rate = TIIE_ACTUAL + operational_component + new_risk_premium

        final_bank_rate = max(calculated_rate, TIIE_ACTUAL + 1.5)

    # -------------------------------------------------------------------------
    # 2. CÁLCULO DE COMISIÓN FINTECH Y PAGOS
    # -------------------------------------------------------------------------
    if category == 'Rechazo (Riesgo Alto)':
        guarantee_fee = 0.0
        total_financed = approved_amount
        monthly_payment = 0.0
        nafin_guaranteed = 0.0
    else:
        nafin_guaranteed = approved_amount * soform_guarantee_pct

        guarantee_fee = nafin_guaranteed * 0.05

        total_financed = approved_amount + guarantee_fee
        monthly_payment = calculate_monthly_payment(total_financed, final_bank_rate, term_months)

    return {
        'approved_amount': approved_amount,
        'nafin_guaranteed': nafin_guaranteed,
        'pd': pd_pred,
        'expected_loss': el_pred,
        'guarantee_fee': guarantee_fee,
        'total_financed': total_financed,
        'monthly_payment': monthly_payment,
        'term_months': term_months,
        'original_rate': market_bank_rate,
        'final_bank_rate': final_bank_rate,
        'scian_code': scian_code,
        'state': state_code,
        'gps_category': category,
        'soform_guarantee_pct': soform_guarantee_pct,
        'action': gps_info['action']
    }

def print_quote(quote):
    print("\n" + "="*70)
    print("                COTIZACIÓN - ESQUEMA DE GARANTÍA")
    print("="*70)

    approved = quote['approved_amount']
    term = quote['term_months']
    rate_before = quote['original_rate']
    rate_after = quote['final_bank_rate']
    pd = quote['pd']
    el = quote['expected_loss']
    guarantee_fee = quote['guarantee_fee']
    monthly_payment = quote['monthly_payment']
    soform_pct = quote['soform_guarantee_pct'] * 100
    category = quote['gps_category']

    # Pago mensual tradicional con tasa original
    payment_before = calculate_monthly_payment(approved, rate_before, term)

    # ------------------------
    #   ENCABEZADO GENERAL
    # ------------------------
    print(f"\n Monto aprobado: ${approved:,.2f} MXN")
    print(f" Plazo: {term} meses")
    print(f" Categoría GPS: {category} (Garantía interna: {soform_pct:.0f}%)")

    # ------------------------
    # TASA ANTES VS DESPUÉS
    # ------------------------
    print("\n" + "-"*70)
    print("IMPACTO DE LA GARANTÍA FINTECH EN LA TASA")
    print("-"*70)

    print(f"➡️ Tasa del banco (antes):    {rate_before:.2f}%")
    print(f"➡️ Tasa recalculada (después): {rate_after:.2f}%")

    if rate_after == 0:
        print("\n❌ Este cliente pertenece a la categoría de rechazo por riesgo.")
        print("   No aplica tasa ni cálculo de pago mensual.")
    else:
        print(f"\n   Ajuste realizado conforme al porcentaje de garantía aportada.")

    # ------------------------
    # COMPARACIÓN DE ESQUEMAS DE GARANTÍA
    # ------------------------
    print("\n" + "-"*70)
    print("COMPARACIÓN DE ESQUEMAS DE GARANTÍA")
    print("-"*70)

    print("\n ESQUEMA TRADICIONAL (BANCO)")
    print(f"   • Pago mensual:            ${payment_before:,.2f}")
    print(f"   • Garantía requerida:      Sí, activo pignorado")
    print(f"   • Tipo de garantía:        Activo fijo o bien inmueble")

    print("\n ESQUEMA FINTECH (GPS)")
    if rate_after > 0:
        print(f"   • Pago mensual:            ${monthly_payment:,.2f}")
    else:
        print("   • Pago mensual:            No aplica")
    print(f"   • Garantía requerida:      No")
    print(f"   • Garantía aportada:       {soform_pct:.0f}% por Fintech")

    if rate_after > 0:
        difference = monthly_payment - payment_before
        print("\n DIFERENCIA")
        print(f"   • Variación en pago mensual: ${difference:,.2f}")
        print("   • Este esquema evita dejar activos en garantía real.")

    # ------------------------
    # INFORMACIÓN DE RIESGO
    # ------------------------
    print("\n" + "-"*70)
    print("MODELO DE RIESGO")
    print("-"*70)
    print(f"   • Probabilidad de Incumplimiento (PD): {pd*100:.2f}%")
    print(f"   • Pérdida Esperada (EL): ${el:,.2f} MXN")

    # ------------------------
    # COSTO DE LA GARANTÍA
    # ------------------------
    print("\n" + "-"*70)
    print("COSTO DE GARANTÍA FINTECH")
    print("-"*70)
    print(f"   • Comisión de Garantía Fintech: ${guarantee_fee:,.2f}")
    print(f"   • Monto garantizado por Fintech: ${quote['nafin_guaranteed']:,.2f}")

    # ------------------------
    # PAGO FINAL
    # ------------------------
    print("\n" + "="*70)
    print("PAGO MENSUAL FINAL DEL CRÉDITO")
    print("="*70)

    if rate_after > 0:
        print(f"   Pago mensual total: ${monthly_payment:,.2f}\n")
    else:
        print("   No aplica, cliente rechazado.\n")

    print("="*70)



# Se mantienen las funciones show_scian_codes, show_state_codes, main, y quick_quote

def show_scian_codes():
    """Muestra los códigos SCIAN disponibles"""
    print("\n--- Códigos SCIAN (Sectores Económicos) ---")
    for code, name in sorted(SECTORES_SCIAN.items()):
        print(f"  {code}: {name}")
    print()


def show_state_codes():
    """Muestra los códigos de estados disponibles"""
    print("\n--- Códigos de Estados ---")
    for code, name in sorted(ESTADOS_MEXICO.items()):
        print(f"  {code}: {name}")
    print()


def main():
    """
    Main interactive quoter
    Cotizador interactivo principal
    """
    print("\n" + "="*60)
    print("CALCULADORA DE GARANTÍA FINTECH - CRÉDITO PYME MÉXICO")
    print("="*60)
    print("\nIngresa los datos del préstamo:\n")

    try:
        # Monto del préstamo
        approved_amount = float(input("1. Monto del préstamo (MXN): $"))

        # Plazo
        term_months = int(input("2. Plazo (meses, ej: 36): "))

        # Empleados
        num_employees = int(input("3. Número de empleados: "))

        # Negocio nuevo
        is_new = input("4. ¿Es negocio nuevo? (s/n): ").lower() in ['s', 'si', 'sí', 'y', 'yes']

        # SCIAN
        show_codes = input("5. ¿Ver códigos SCIAN? (s/n): ").lower() in ['s', 'si', 'sí', 'y', 'yes']
        if show_codes:
            show_scian_codes()
        scian = input("  Código SCIAN (2 dígitos, ej: 46 para comercio): ").strip()

        # Estado
        show_states = input("6. ¿Ver códigos de estados? (s/n): ").lower() in ['s', 'si', 'sí', 'y', 'yes']
        if show_states:
            show_state_codes()
        state = input("  Estado (ej: JAL, CDMX, NL): ").strip().upper()

        # Tasa de interés
        bank_rate = float(input("7. Tasa de interés del banco (%, ej: 12.5): "))

        # Garantía inmobiliaria (opcional)
        has_real_estate = input("8. ¿Tiene garantía inmobiliaria? (s/n): ").lower() in ['s', 'si', 'sí', 'y', 'yes']

        print("\nCalculando cotización...")

        quote = calculate_quote(
            approved_amount, term_months, num_employees,
            is_new, scian, state, bank_rate, has_real_estate
        )

        print_quote(quote)

    except ValueError:
        print("\n Error: Por favor ingresa valores numéricos válidos.")
    except KeyboardInterrupt:
        print("\n\n ¡Hasta luego!")
    except Exception as e:
        print(f"\n Error: {e}")


def quick_quote(amount, term, employees, is_new, scian, state, rate):
    """
    Quick quote for programmatic use
    Cotización rápida para uso programático
    """
    quote = calculate_quote(amount, term, employees, is_new, scian, state, rate)
    print_quote(quote)
    return quote


if __name__ == "__main__":
    main()
