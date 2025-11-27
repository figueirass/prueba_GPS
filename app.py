"""
Aplicación Web Streamlit - Cotizador de Créditos PyME México
Interfaz gráfica para el sistema de Garantía Premium Select (GPS)
"""

import streamlit as st
import pandas as pd
import numpy as np
from quoter import calculate_quote, load_models
from features import SECTORES_SCIAN, ESTADOS_MEXICO

# Configuración de la página
st.set_page_config(
    page_title="Cotizador PyME México",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .category-ultra-oro {
        background-color: #FFD700;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .category-oro {
        background-color: #FFA500;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .category-estandar {
        background-color: #FFE66D;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .category-rechazo {
        background-color: #FF6B6B;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.title("💰 Cotizador de Créditos PyME México")
    st.markdown("### Sistema de Garantía Premium Select (GPS)")
    st.markdown("---")

    # Sidebar con información
    with st.sidebar:
        st.header("ℹ️ Información del Sistema")
        st.markdown("""
        **Garantía Premium Select (GPS)**
        
        Sistema de evaluación de riesgo crediticio que clasifica préstamos en categorías según la Probabilidad de Default (PD):
        
        - ✨ **Ultra-Oro**: PD < 1%
        - ⭐ **Oro**: PD < 3%
        - 🟡 **Estándar**: PD < 9%
        - 🔴 **Rechazo**: PD ≥ 9%
        - 🔴 **Rechazo**: PD ≥ 3%
        
        La garantía FINTECH se mantiene, pero la garantía interna de la SOFOM varía según la categoría.
        """)

        st.markdown("---")
        st.markdown("**Desarrollado por:**")
        st.markdown("Equipo de Ingeniería Financiera ITESO")

    # Crear dos columnas para el formulario
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Datos del Préstamo")

        # Monto del préstamo
        approved_amount = st.number_input(
            "Monto del Préstamo (MXN)",
            min_value=50000,
            max_value=50000000,
            value=500000,
            step=10000,
            format="%d"
        )

        # Plazo
        term_months = st.selectbox(
            "Plazo (meses)",
            options=[6, 12, 18, 24, 36, 48, 60, 84, 120],
            index=4  # Default: 36 meses
        )

        # Número de empleados
        num_employees = st.number_input(
            "Número de Empleados",
            min_value=1,
            max_value=1000,
            value=12,
            step=1
        )

        # Negocio nuevo
        is_new_business = st.checkbox("¿Es negocio nuevo? (< 2 años)")

    with col2:
        st.subheader("🏢 Información del Negocio")

        # Sector SCIAN
        scian_options = {f"{code} - {nombre[:40]}": code 
                        for code, nombre in SECTORES_SCIAN.items()}
        scian_selected = st.selectbox(
            "Sector Económico (SCIAN)",
            options=list(scian_options.keys()),
            index=8  # Default: Comercio al por menor
        )
        scian_code = scian_options[scian_selected]

        # Estado
        estado_options = {f"{code} - {nombre}": code 
                         for code, nombre in ESTADOS_MEXICO.items()}
        estado_selected = st.selectbox(
            "Estado",
            options=list(estado_options.keys()),
            index=13  # Default: JAL
        )
        state_code = estado_options[estado_selected]

        # Tasa de interés
        bank_rate = st.slider(
            "Tasa de Interés Anual (%)",
            min_value=8.0,
            max_value=25.0,
            value=14.5,
            step=0.5
        )

        # Opciones adicionales
        st.markdown("**Opciones Adicionales:**")
        has_real_estate = st.checkbox("Tiene garantía inmobiliaria")
        in_recession = st.checkbox("En período de crisis económica")

    # Botón para calcular
    st.markdown("---")

    if st.button("🔍 Calcular Cotización", type="primary", use_container_width=True):
        with st.spinner("Calculando cotización..."):
            try:
                # Calcular cotización
                quote = calculate_quote(
                    approved_amount=approved_amount,
                    term_months=term_months,
                    num_employees=num_employees,
                    is_new_business=is_new_business,
                    scian_code=scian_code,
                    state_code=state_code,
                    bank_rate=bank_rate,
                    has_real_estate=has_real_estate,
                    in_recession=in_recession
                )

                # Mostrar resultados
                st.markdown("---")
                st.markdown("## 📊 Resultados de la Cotización")

                # Categoría GPS
                category = quote['gps_category']
                if category == 'Ultra–Oro':
                    st.markdown('<div class="category-ultra-oro">✨ CATEGORÍA: ULTRA-ORO (PD < 1%)</div>', 
                              unsafe_allow_html=True)
                elif category == 'Oro':
                    st.markdown('<div class="category-oro">⭐ CATEGORÍA: ORO (PD < 3%)</div>', 
                              unsafe_allow_html=True)
                elif category == 'Estándar':
                    st.markdown('<div class="category-estandar">🟡 CATEGORÍA: ESTÁNDAR (PD < 9%)</div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown('<div class="category-rechazo">🔴 CATEGORÍA: RECHAZO (PD ≥ 9%)</div>', 
                              unsafe_allow_html=True)

                st.markdown("")

                # Crear columnas para métricas
                metric_col1, metric_col2, metric_col3 = st.columns(3)

                with metric_col1:
                    st.metric(
                        "Probabilidad de Default",
                        f"{quote['pd']*100:.2f}%",
                        delta=None
                    )

                with metric_col2:
                    st.metric(
                        "Pérdida Esperada",
                        f"${quote['expected_loss']:,.0f} MXN",
                        delta=None
                    )

                with metric_col3:
                    st.metric(
                        "Garantía Interna SOFOM",
                        f"{quote['soform_guarantee_pct']*100:.0f}%",
                        delta=None
                    )

                # Detalles financieros
                if category != 'Rechazo (Riesgo Alto)':
                    st.markdown("---")
                    st.subheader("💵 Detalles Financieros")

                    fin_col1, fin_col2 = st.columns(2)

                    with fin_col1:
                        st.markdown(f"""
                        **Monto Solicitado:** ${quote['approved_amount']:,.2f} MXN  
                        **Garantía FINTECH:** ${quote['nafin_guaranteed']:,.2f} MXN  
                        **Comisión FINTECH:** ${quote['guarantee_fee']:,.2f} MXN  
                        """)

                    with fin_col2:
                        fee_pct = (quote['guarantee_fee'] / quote['nafin_guaranteed']) * 100 if quote['nafin_guaranteed'] > 0 else 0
                        st.markdown(f"""
                        **Total a Financiar:** ${quote['total_financed']:,.2f} MXN  
                        **Plazo:** {quote['term_months']} meses  
                        **Tasa Anual:** {quote['bank_rate']:.2f}%  
                        """)

                    # Pago mensual destacado
                    st.markdown("---")
                    st.markdown("### 💳 Pago Mensual Estimado")
                    st.markdown(f"<h1 style='text-align: center; color: #2E86AB;'>${quote['monthly_payment']:,.2f} MXN</h1>", 
                              unsafe_allow_html=True)

                    # Acción sugerida
                    st.info(f"**Acción Sugerida:** {quote['action']}")

                else:
                    # Caso de rechazo
                    st.error(f"""
                    **❌ SOLICITUD RECHAZADA**
                    
                    La probabilidad de default ({quote['pd']*100:.2f}%) supera el umbral aceptable (9%).
                    
                    La pérdida esperada de ${quote['expected_loss']:,.2f} MXN es superior al límite operativo.
                    
                    **Acción:** {quote['action']}
                    """)

                # Información adicional en expander
                with st.expander("📈 Ver Información Técnica Detallada"):
                    st.json({
                        "Monto Aprobado": f"${quote['approved_amount']:,.2f} MXN",
                        "Probabilidad de Default (PD)": f"{quote['pd']*100:.4f}%",
                        "Pérdida Dado el Default (LGD)": f"${quote['lgd']:,.2f} MXN",
                        "Pérdida Esperada (EL)": f"${quote['expected_loss']:,.2f} MXN",
                        "Categoría GPS": quote['gps_category'],
                        "Garantía SOFOM": f"{quote['soform_guarantee_pct']*100:.0f}%",
                        "Sector": f"{scian_code} - {SECTORES_SCIAN[scian_code]['nombre']}",
                        "Estado": f"{state_code} - {ESTADOS_MEXICO[state_code]}"
                    })

            except Exception as e:
                st.error(f"Error al calcular la cotización: {str(e)}")
                st.exception(e)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <small>Sistema de Cotización de Créditos PyME México v1.0 | 
    Desarrollado con Machine Learning | 
    ITESO Universidad Jesuita de Guadalajara</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Verificar que los modelos estén entrenados
    try:
        load_models()
    except:
        st.warning("""
        ⚠️ **Modelos no encontrados**
        
        Por favor, entrena los modelos primero ejecutando:
        ```bash
        python train.py
        ```
        """)

    main()