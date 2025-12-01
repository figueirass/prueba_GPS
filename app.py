"""
Aplicación Web Streamlit - Cotizador de Créditos PyME México
Interfaz gráfica para el sistema de Garantía Premium Select (GPS)
"""

import streamlit as st
import pandas as pd
import numpy as np
from quoter import calculate_quote, load_models, calculate_monthly_payment
from features import SECTORES_SCIAN, ESTADOS_MEXICO

# Configuración de la página
st.set_page_config(
    page_title="Cotizador PyME México - GPS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .category-ultra-oro {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        color: #000;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .category-oro {
        background: linear-gradient(135deg, #FFA500 0%, #FF8C00 100%);
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        color: #fff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .category-estandar {
        background: linear-gradient(135deg, #FFE66D 0%, #FFDB4D 100%);
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        color: #000;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .category-rechazo {
        background: linear-gradient(135deg, #FF6B6B 0%, #C92A2A 100%);
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .comparison-box {
        background-color: #1e2936;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin: 10px 0;
        color: #ffffff;
    }
    .savings-highlight {
        background-color: #1e3a28;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
        color: #ffffff;
    }
    .cost-highlight {
        background-color: #3a1e1e;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 10px 0;
        color: #ffffff;
    }
    .info-box {
        background-color: #1e2936;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #0066cc;
        margin: 10px 0;
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<p class="main-header">Sistema de Garantía Premium Select</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Cotizador de Créditos PyME México</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar con información
    with st.sidebar:
        st.header("Información del Sistema")
        st.markdown("""
        **Garantía Premium Select (GPS)**
        
        Sistema de evaluación de riesgo crediticio que clasifica préstamos 
        según la Probabilidad de Default (PD):
        
        **Categorías:**
        - **Ultra-Oro**: PD < 1% | Garantía 85%
        - **Oro**: PD < 3% | Garantía 70%
        - **Rechazo**: PD ≥ 3%
        
        **Beneficios:**
        - Reducción en tasa de interés
        - Sin garantías reales requeridas
        - Aprobación basada en datos
        """)

        st.markdown("---")
        st.markdown("""
        **Desarrollado por:**  
        Equipo de Ingeniería Financiera  
        ITESO Universidad Jesuita de Guadalajara
        """)

    # Crear dos columnas para el formulario
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Datos del Préstamo")

        # Monto del préstamo
        approved_amount = st.number_input(
            "Monto del Préstamo (MXN)",
            min_value=50000,
            max_value=50000000,
            value=500000,
            step=10000,
            format="%d",
            help="Ingrese el monto total que desea solicitar"
        )

        # Plazo
        term_months = st.selectbox(
            "Plazo (meses)",
            options=[6, 12, 18, 24, 36, 48, 60, 84, 120],
            index=4,
            help="Seleccione el plazo del préstamo"
        )

        # Número de empleados
        num_employees = st.number_input(
            "Número de Empleados",
            min_value=1,
            max_value=1000,
            value=12,
            step=1,
            help="Número total de empleados en la empresa"
        )

        # Negocio nuevo
        is_new_business = st.checkbox(
            "Es negocio nuevo (< 2 años)",
            help="Marque si la empresa tiene menos de 2 años de operación"
        )

    with col2:
        st.subheader("Información del Negocio")

        # Sector SCIAN
        scian_options = {f"{code} - {nombre[:40]}": code 
                        for code, nombre in SECTORES_SCIAN.items()}
        scian_selected = st.selectbox(
            "Sector Económico (SCIAN)",
            options=list(scian_options.keys()),
            index=8,
            help="Seleccione el sector industrial de su empresa"
        )
        scian_code = scian_options[scian_selected]

        # Estado
        estado_options = {f"{code} - {nombre}": code 
                         for code, nombre in ESTADOS_MEXICO.items()}
        estado_selected = st.selectbox(
            "Estado",
            options=list(estado_options.keys()),
            index=13,
            help="Seleccione el estado donde opera la empresa"
        )
        state_code = estado_options[estado_selected]

        # Tasa de interés del mercado (banco)
        market_bank_rate = st.slider(
            "Tasa de Interés Ofrecida por el Banco (%)",
            min_value=8.0,
            max_value=35.0,
            value=18.5,
            step=0.5,
            help="Ingrese la tasa que el banco le está ofreciendo actualmente"
        )

        # Opciones adicionales
        st.markdown("**Opciones Adicionales:**")
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            has_real_estate = st.checkbox("Garantía inmobiliaria")
        with col_opt2:
            in_recession = st.checkbox("Período de crisis")

    # Botón para calcular
    st.markdown("---")

    if st.button("Calcular Cotización", type="primary", use_container_width=True):
        with st.spinner("Analizando perfil crediticio..."):
            try:
                # Calcular cotización
                quote = calculate_quote(
                    approved_amount=approved_amount,
                    term_months=term_months,
                    num_employees=num_employees,
                    is_new_business=is_new_business,
                    scian_code=scian_code,
                    state_code=state_code,
                    market_bank_rate=market_bank_rate,
                    has_real_estate=has_real_estate,
                    in_recession=in_recession
                )

                # Mostrar resultados
                st.markdown("---")
                st.markdown("## Resultados de la Evaluación")

                # Categoría GPS
                category = quote['gps_category']
                if category == 'Ultra–Oro':
                    st.markdown('<div class="category-ultra-oro">CATEGORÍA: ULTRA-ORO (PD < 1%)<br>Garantía Interna: 85%</div>', 
                              unsafe_allow_html=True)
                elif category == 'Oro':
                    st.markdown('<div class="category-oro">CATEGORÍA: ORO (PD < 3%)<br>Garantía Interna: 70%</div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown('<div class="category-rechazo">CATEGORÍA: RECHAZO (PD ≥ 3%)<br>No procede</div>', 
                              unsafe_allow_html=True)

                st.markdown("")

                # Métricas principales
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

                with metric_col1:
                    st.metric(
                        "Probabilidad de Default",
                        f"{quote['pd']*100:.2f}%",
                        help="Probabilidad estimada de incumplimiento"
                    )

                with metric_col2:
                    st.metric(
                        "Pérdida Esperada",
                        f"${quote['expected_loss']:,.0f}",
                        help="Pérdida esperada en caso de incumplimiento"
                    )

                with metric_col3:
                    st.metric(
                        "Garantía Interna",
                        f"{quote['soform_guarantee_pct']*100:.0f}%",
                        help="Porcentaje de garantía aportada por GPS"
                    )

                with metric_col4:
                    if category != 'Rechazo (Riesgo Alto)':
                        rate_reduction = quote['original_rate'] - quote['final_bank_rate']
                        st.metric(
                            "Reducción de Tasa",
                            f"{rate_reduction:.2f}%",
                            delta=f"-{rate_reduction:.2f}%",
                            help="Reducción en puntos porcentuales"
                        )
                    else:
                        st.metric("Reducción de Tasa", "N/A")

                # Detalles según categoría
                if category != 'Rechazo (Riesgo Alto)':
                    st.markdown("---")
                    
                    # === COMPARACIÓN DE TASAS ===
                    st.subheader("Impacto de la Garantía GPS en la Tasa de Interés")
                    
                    comp_col1, comp_col2 = st.columns(2)
                    
                    with comp_col1:
                        st.markdown("""
                        <div class="cost-highlight">
                        <h4 style="margin-top:0; color: #ffffff;">Esquema Tradicional (Banco)</h4>
                        <p style="color: #ffffff;"><strong>Tasa de Interés:</strong> {:.2f}%</p>
                        <p style="color: #ffffff;"><strong>Garantía Requerida:</strong> Activo fijo o inmueble</p>
                        </div>
                        """.format(quote['original_rate']), unsafe_allow_html=True)
                    
                    with comp_col2:
                        st.markdown("""
                        <div class="savings-highlight">
                        <h4 style="margin-top:0; color: #ffffff;">Esquema GPS (Con Garantía)</h4>
                        <p style="color: #ffffff;"><strong>Tasa de Interés:</strong> {:.2f}%</p>
                        <p style="color: #ffffff;"><strong>Garantía Requerida:</strong> No se requiere garantía real</p>
                        </div>
                        """.format(quote['final_bank_rate']), unsafe_allow_html=True)

                    # === COMPARACIÓN DE PAGOS MENSUALES ===
                    st.markdown("---")
                    st.subheader("Comparación de Pagos Mensuales")
                    
                    # Calcular pago tradicional
                    traditional_payment = calculate_monthly_payment(
                        approved_amount, 
                        quote['original_rate'], 
                        term_months
                    )
                    
                    payment_col1, payment_col2, payment_col3 = st.columns(3)
                    
                    with payment_col1:
                        st.markdown("""
                        <div class="comparison-box">
                        <h5 style="color: #ffffff;">Pago Mensual Tradicional</h5>
                        <h3 style="color: #ff6b6b;">${:,.2f}</h3>
                        <p style="font-size: 0.9em; color: #cccccc;">Con tasa {:.2f}%</p>
                        </div>
                        """.format(traditional_payment, quote['original_rate']), unsafe_allow_html=True)
                    
                    with payment_col2:
                        st.markdown("""
                        <div class="comparison-box">
                        <h5 style="color: #ffffff;">Pago Mensual con GPS</h5>
                        <h3 style="color: #5dca88;">${:,.2f}</h3>
                        <p style="font-size: 0.9em; color: #cccccc;">Con tasa {:.2f}%</p>
                        </div>
                        """.format(quote['monthly_payment'], quote['final_bank_rate']), unsafe_allow_html=True)
                    
                    with payment_col3:
                        monthly_savings = traditional_payment - quote['monthly_payment']
                        total_savings = monthly_savings * term_months
                        st.markdown("""
                        <div class="savings-highlight">
                        <h5 style="color: #ffffff;">Ahorro Total</h5>
                        <h3 style="color: #5dca88;">${:,.2f}</h3>
                        <p style="font-size: 0.9em; color: #cccccc;">Durante {} meses</p>
                        <p style="font-size: 0.85em; margin-top: 5px; color: #cccccc;">(${:,.2f}/mes)</p>
                        </div>
                        """.format(total_savings, term_months, monthly_savings), unsafe_allow_html=True)

                    # === DETALLES FINANCIEROS ===
                    st.markdown("---")
                    st.subheader("Detalles Financieros")

                    fin_col1, fin_col2 = st.columns(2)

                    with fin_col1:
                        st.markdown("""
                        <div class="info-box">
                        <h5 style="color: #ffffff;">Características del Préstamo</h5>
                        <table style="width:100%; margin-top: 10px; color: #ffffff;">
                        <tr><td><strong>Monto Solicitado:</strong></td><td style="text-align:right;">${:,.2f}</td></tr>
                        <tr><td><strong>Plazo:</strong></td><td style="text-align:right;">{} meses</td></tr>
                        <tr><td><strong>Garantía GPS:</strong></td><td style="text-align:right;">${:,.2f}</td></tr>
                        <tr><td><strong>Comisión GPS:</strong></td><td style="text-align:right;">${:,.2f}</td></tr>
                        </table>
                        </div>
                        """.format(
                            quote['approved_amount'],
                            quote['term_months'],
                            quote['nafin_guaranteed'],
                            quote['guarantee_fee']
                        ), unsafe_allow_html=True)

                    with fin_col2:
                        st.markdown("""
                        <div class="info-box">
                        <h5 style="color: #ffffff;">Análisis de Riesgo</h5>
                        <table style="width:100%; margin-top: 10px; color: #ffffff;">
                        <tr><td><strong>Categoría GPS:</strong></td><td style="text-align:right;">{}</td></tr>
                        <tr><td><strong>Probabilidad Default:</strong></td><td style="text-align:right;">{:.2f}%</td></tr>
                        <tr><td><strong>Pérdida Esperada:</strong></td><td style="text-align:right;">${:,.2f}</td></tr>
                        <tr><td><strong>Sector:</strong></td><td style="text-align:right;">{}</td></tr>
                        </table>
                        </div>
                        """.format(
                            quote['gps_category'],
                            quote['pd'] * 100,
                            quote['expected_loss'],
                            SECTORES_SCIAN.get(scian_code, 'No especificado')[:25]
                        ), unsafe_allow_html=True)

                    # Acción sugerida
                    st.markdown("---")
                    st.success(f"**Recomendación:** {quote['action']}")

                else:
                    # Caso de rechazo
                    st.markdown("---")
                    st.error(f"""
                    **SOLICITUD NO PROCEDE**
                    
                    La evaluación de riesgo indica una probabilidad de incumplimiento de {quote['pd']*100:.2f}%, 
                    superior al umbral aceptable del programa GPS (3%).
                    
                    **Pérdida Esperada Estimada:** ${quote['expected_loss']:,.2f} MXN
                    
                    **Recomendación:** {quote['action']}
                    
                    Sugerimos revisar las condiciones del préstamo o considerar otras alternativas de financiamiento.
                    """)

                # Información técnica detallada (expandible)
                with st.expander("Ver Información Técnica Detallada"):
                    technical_data = {
                        "Monto Aprobado": f"${quote['approved_amount']:,.2f} MXN",
                        "Plazo": f"{quote['term_months']} meses",
                        "Probabilidad de Default (PD)": f"{quote['pd']*100:.4f}%",
                        "Pérdida Esperada (EL)": f"${quote['expected_loss']:,.2f} MXN",
                        "Categoría GPS": quote['gps_category'],
                        "Garantía GPS (%)": f"{quote['soform_guarantee_pct']*100:.0f}%",
                        "Garantía GPS (Monto)": f"${quote['nafin_guaranteed']:,.2f} MXN",
                        "Comisión GPS": f"${quote['guarantee_fee']:,.2f} MXN",
                        "Tasa Original": f"{quote['original_rate']:.2f}%",
                        "Tasa Ajustada": f"{quote['final_bank_rate']:.2f}%",
                        "Reducción de Tasa": f"{quote['original_rate'] - quote['final_bank_rate']:.2f} puntos porcentuales",
                        "Pago Mensual": f"${quote['monthly_payment']:,.2f} MXN" if category != 'Rechazo (Riesgo Alto)' else "N/A",
                        "Sector": f"{scian_code} - {SECTORES_SCIAN.get(scian_code, 'No especificado')}",
                        "Estado": f"{state_code} - {ESTADOS_MEXICO.get(state_code, state_code)}"
                    }
                    
                    st.json(technical_data)

            except Exception as e:
                st.error(f"Error al calcular la cotización: {str(e)}")
                st.exception(e)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>Sistema de Cotización de Créditos PyME México v2.0</p>
    <p>Desarrollado con Machine Learning | ITESO Universidad Jesuita de Guadalajara</p>
    <p>© 2025 - Todos los derechos reservados</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Verificar que los modelos estén entrenados
    try:
        load_models()
    except:
        st.warning("""
        **Modelos no encontrados**
        
        Por favor, entrena los modelos primero ejecutando:
        ```bash
        python train.py
        ```
        """)

    main()
