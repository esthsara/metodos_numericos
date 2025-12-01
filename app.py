import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from metodos_raices import *

# Configuración de página
st.set_page_config(page_title="Métodos Numéricos para Raíces", layout="wide")
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stApp {background-color: #14532d;}
    h1, h2, h3 {color: #22c55e !important;}
    .css-1d391kg {color: #16a34a;}
    .stSelectbox, .stTextInput, .stNumberInput {background-color: #166534; color: white;}
</style>
""", unsafe_allow_html=True)

st.title(" Métodos Numéricos para Encontrar Raíces")
st.markdown("<h3 style='color:#22c55e;'>Bisección • Newton-Raphson • Secante</h3>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📝 Configuración")
    funcion_str = st.text_input("Ecuación f(x) = 0", value="x**3 - 2*x - 5", help="Ejemplos: x**2 - 4, exp(x) - 3*x, sin(x) - x/2")
    metodo = st.selectbox("Método", ["Bisección", "Newton-Raphson", "Secante"])
    
    col1, col2 = st.columns(2)
    with col1:
        a = st.number_input("a (o x0)", value=-5.0 if metodo == "Bisección" else 2.0)
    with col2:
        b = st.number_input("b (o x1)", value=5.0 if metodo == "Bisección" else 3.0, disabled=metodo != "Bisección" and metodo != "Secante")
    
    x0_nr = st.number_input("x0 (Newton)", value=2.0, disabled=metodo != "Newton-Raphson")
    
    tol = st.slider("Tolerancia", 1e-15, 1e-5, 1e-10)
    st.markdown("---")
    if st.button("Calcular Raíz"):
        st.session_state.calcular = True

# Main
if 'calcular' in st.session_state:
    try:
        f, df, expr_sym = crear_funcion(funcion_str)
        exacta_raw = solucion_exacta(expr_sym)  # Esto ya es float gracias a .evalf()
        exacta = float(exacta_raw) if exacta_raw is not None else None  # Asegura float
        
        if metodo == "Bisección":
            resultado = biseccion(f, a, b, tol)
        elif metodo == "Newton-Raphson":
            resultado = newton_raphson(f, df, x0_nr, tol)
        else:
            resultado = secante(f, a if metodo == "Secante" else x0_nr, b, tol)
        
        raiz = resultado['raiz']
        its = resultado['iteraciones']
        
        # Resultados
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Raíz aproximada:** {raiz:.12f}")
            if exacta is not None:
                error_real = abs(raiz - exacta)
                st.info(f"**Solución exacta (SymPy):** {exacta:.12f}\n\n**Error real:** {error_real:.2e}")
            else:
                st.warning("No se pudo calcular solución exacta (ecuación compleja). Usa aproximación.")
        
        with col2:
            st.metric("Iteraciones", len(its))
            st.metric("Convergencia", "Sí" if resultado['convergio'] else "No")
        
        # Tabla de iteraciones
        st.markdown("### 📊 Iteraciones")
        df_its = pd.DataFrame(its)
        st.dataframe(df_its.round(10), use_container_width=True)
        
        # Gráfico (sin cambios)
        st.markdown("### 📈 Visualización de la Aproximación")
        rango_min = min(a, x0_nr if metodo == "Newton-Raphson" else a)
        rango_max = max(b, x0_nr if metodo == "Newton-Raphson" else b)
        x_vals = np.linspace(rango_min-2, rango_max+2, 1000)
        y_vals = f(x_vals)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, name="f(x)", line=dict(color="#22c55e")))
        fig.add_trace(go.Scatter(x=[raiz], y=[0], mode="markers", name="Raíz", marker=dict(color="yellow", size=12)))
        
        if metodo == "Bisección":
            xs = [it['c'] for it in its]
        elif metodo == "Newton-Raphson":
            xs = [it['x_new'] for it in its]
        else:
            xs = [it['x_{n+1}'] for it in its]
        
        ys = [f(x) for x in xs]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+lines", name="Iteraciones", marker=dict(color="#f59e0b")))
        
        fig.add_hline(y=0, line_dash="dash", line_color="white")
        fig.update_layout(template="plotly_dark", title=f"Método: {metodo}", xaxis_title="x", yaxis_title="f(x)")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.code(traceback.format_exc())  # Agrega esto para debug: importa traceback arriba
st.markdown("---")
st.caption("Desarrollado con usando Streamlit + SymPy + Plotly")