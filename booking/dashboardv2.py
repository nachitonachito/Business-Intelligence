from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

CARPETA = Path(__file__).parent

# Same colours as the slides
TINTA = "#1B2430"
VERDE = "#0F766E"
ROJO = "#E4572E"
GRIS = "#B8C2CE"
AZUL = "#3B6FA0"
TEXTO_SUAVE = "#64748B"
OPCIONES = [GRIS, VERDE, AZUL]

st.set_page_config(page_title="Hotel cancellations", page_icon="🏨", layout="wide")

st.markdown(f"""
<style>
.block-container {{ padding-top: 2rem; max-width: 1250px; }}
.hero {{
    background: linear-gradient(120deg, {TINTA} 0%, #22324A 100%);
    color: white; border-radius: 18px; padding: 28px 34px; margin-bottom: 18px;
}}
.hero .tag {{ letter-spacing: .18em; font-size: .78rem; font-weight: 700; color: #5EC2B4; text-transform: uppercase; }}
.hero h1 {{ color: white; font-size: 2.2rem; margin: 6px 0 4px 0; padding: 0; }}
.hero p {{ color: #C9D3DF; margin: 0; font-size: 1rem; }}
.kpi {{
    border-radius: 14px; padding: 18px 20px; height: 100%;
    background: rgba(15, 118, 110, .07); border: 1px solid rgba(15, 118, 110, .18);
}}
.kpi.rojo {{ background: rgba(228, 87, 46, .07); border-color: rgba(228, 87, 46, .22); }}
.kpi.gris {{ background: rgba(100, 116, 139, .07); border-color: rgba(100, 116, 139, .2); }}
.kpi .valor {{ font-size: 2.1rem; font-weight: 800; line-height: 1.1; color: {VERDE}; }}
.kpi.rojo .valor {{ color: {ROJO}; }}
.kpi.gris .valor {{ color: inherit; }}
.kpi .nombre {{ font-size: .92rem; font-weight: 600; margin-top: 6px; }}
.kpi .nota {{ font-size: .8rem; color: {TEXTO_SUAVE}; margin-top: 2px; }}
.caja {{
    border-radius: 12px; padding: 16px 20px; margin: 10px 0 6px 0;
    border-left: 5px solid {VERDE}; background: rgba(15, 118, 110, .07);
}}
.caja.rojo {{ border-left-color: {ROJO}; background: rgba(228, 87, 46, .07); }}
.caja.gris {{ border-left-color: {TEXTO_SUAVE}; background: rgba(100, 116, 139, .07); }}
.seccion {{ font-size: 1.25rem; font-weight: 700; margin: 6px 0 2px 0; }}
.sub {{ color: {TEXTO_SUAVE}; font-size: .92rem; margin-bottom: 12px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{
    border-radius: 10px 10px 0 0; padding: 10px 16px; font-weight: 600;
}}
.stTabs [aria-selected="true"] {{ color: {VERDE} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {VERDE}; }}
</style>
""", unsafe_allow_html=True)

datos = pd.read_csv(CARPETA / "datos_limpios.csv")
test = pd.read_csv(CARPETA / "resultados_test.csv")
datos["ingreso"] = datos["adr"] * (datos["stays_in_weekend_nights"] + datos["stays_in_week_nights"])


def recall(marcada):
    return (marcada & (test["is_canceled"] == 1)).sum() / test["is_canceled"].sum()


def precision(marcada):
    return test.loc[marcada, "is_canceled"].mean() if marcada.any() else 0


def kpi(columna, valor, nombre, nota="", estilo=""):
    columna.markdown(f'<div class="kpi {estilo}"><div class="valor">{valor}</div>'
                     f'<div class="nombre">{nombre}</div><div class="nota">{nota}</div></div>', unsafe_allow_html=True)


def caja(texto, estilo=""):
    st.markdown(f'<div class="caja {estilo}">{texto}</div>', unsafe_allow_html=True)


def titulo(texto, sub=""):
    st.markdown(f'<div class="seccion">{texto}</div><div class="sub">{sub}</div>', unsafe_allow_html=True)


def estilo(grafico, alto=420):
    grafico.update_layout(
        template="plotly_white", height=alto, margin=dict(l=10, r=10, t=60, b=10),
        font=dict(family="Source Sans Pro, Segoe UI, sans-serif", size=13, color="#334155"),
        title=dict(font=dict(size=16, color=TINTA), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, x=0, title=None),
        hoverlabel=dict(bgcolor="white", font_size=13),
    )
    grafico.update_xaxes(showgrid=False, linecolor="#CBD5E1")
    grafico.update_yaxes(gridcolor="#E2E8F0", zeroline=False)
    return grafico


def envolver(grafico, nombres):
    # Show long option names on two lines instead of tilted
    def partir(texto):
        palabras, lineas, actual = texto.split(), [], ""
        for p in palabras:
            if actual and len(actual) + len(p) > 20:
                lineas.append(actual)
                actual = p
            else:
                actual = (actual + " " + p).strip()
        return "<br>".join(lineas + [actual])
    grafico.update_xaxes(tickangle=0, tickvals=list(nombres), ticktext=[partir(n) for n in nombres])
    return grafico


def resaltar(fila, nombre):
    color = "background-color: rgba(15, 118, 110, .12); font-weight: 700" if fila.name == nombre else ""
    return [color] * len(fila)


st.markdown("""
<div class="hero">
  <div class="tag">IIB415A · Business Intelligence</div>
  <h1>Which bookings will be cancelled?</h1>
  <p>Two hotels, arrivals from July 2015 to August 2017. Results from the notebook's test set (29,823 bookings).</p>
</div>
""", unsafe_allow_html=True)

vista1, vista2, vista3, vista4, vista5, vista6 = st.tabs(["📊 1. Overview", "🔍 2. Who cancels", "🤖 3. Models", "🎚️ 4. Threshold", "💶 5. Business impact", "🧪 6. Deposit (illustration)"])

with vista1:
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, f"{len(datos):,}", "Bookings", "after cleaning", "gris")
    kpi(c2, f"{datos['is_canceled'].mean():.1%}", "Cancelled", "cancelled or no-show", "rojo")
    kpi(c3, f"{datos.loc[datos['is_canceled'] == 1, 'ingreso'].sum() / 1e6:.1f} M", "Revenue of cancelled bookings", "price per night × nights", "rojo")
    kpi(c4, f"{1 - datos['is_canceled'].mean():.1%}", "Baseline accuracy", "Always predicting 'not cancelled'. It finds no cancellations.", "gris")
    st.write("")
    caja("<b>The decision:</b> the hotel loses money when bookings cancel. Can a model tell it which bookings to act on, "
         "and is that better than contacting every guest?")

    # The answer in one minute, with the default assumptions of tabs 5 and 6
    marcada = test["prob_forest"] >= 0.3
    cancela = test["is_canceled"] == 1
    por_mes = len(datos) / 26 / len(test)
    llamada = 10 / 60 * 10
    mes_modelo = (0.10 * test.loc[marcada & cancela, "ingreso"].sum() - llamada * marcada.sum()) * por_mes
    mes_todos = (0.10 * test.loc[cancela, "ingreso"].sum() - llamada * len(test)) * por_mes
    estricta = test["prob_forest"] >= 0.5
    dep_modelo = (0.10 * test.loc[estricta & cancela, "ingreso"].sum() - 0.10 * test.loc[estricta & ~cancela, "ingreso"].sum()) * por_mes
    dep_todos = (0.10 * test.loc[cancela, "ingreso"].sum() - 0.10 * test.loc[~cancela, "ingreso"].sum()) * por_mes

    titulo("The answer", "With the default assumptions of tabs 5 and 6, per month for both hotels.")
    c1, c2, c3 = st.columns(3)
    kpi(c1, f"{recall(marcada):.0%}", "1. The model predicts well",
        f"It finds {recall(marcada):.0%} of cancellations calling {marcada.mean():.0%} of bookings (tabs 3-4).")
    kpi(c2, f"+{mes_todos:,.0f}", "2. For a cheap call, call everyone",
        f"Calling everyone earns {mes_todos:,.0f} vs {mes_modelo:,.0f} with the model: a wrong guess costs a few minutes (tab 5).", "gris")
    kpi(c3, f"{dep_modelo:+,.0f}", "3. For a deposit, use the model",
        f"Model {dep_modelo:+,.0f} vs {dep_todos:+,.0f} asking everyone, if 10% of good guests leave. Test it first with an A/B test (tab 6).")

with vista2:
    datos["anticipacion"] = pd.cut(datos["lead_time"], bins=[-1, 7, 30, 90, 180, 365, 1000],
                                   labels=["0-7", "8-30", "31-90", "91-180", "181-365", "366+"])
    izquierda, derecha = st.columns(2)

    tabla = datos.groupby("anticipacion", observed=True)["is_canceled"].mean().mul(100).reset_index()
    grafico = px.bar(tabla, x="anticipacion", y="is_canceled", range_y=[0, 100], text_auto=".0f",
                     labels={"anticipacion": "Days between booking and arrival", "is_canceled": "% cancelled"},
                     title="Bookings made far ahead cancel more")
    grafico.update_traces(marker_color=VERDE, marker_cornerradius=6, texttemplate="%{y:.0f}%", textposition="outside")
    with izquierda.container(border=True):
        st.plotly_chart(estilo(grafico), width="stretch")

    segmentos = datos.groupby("market_segment")["is_canceled"].agg(["size", "mean"])
    segmentos = segmentos[segmentos["size"] >= 1000].sort_values("mean", ascending=False)
    segmentos["mean"] = segmentos["mean"] * 100
    grafico = px.bar(segmentos.reset_index(), x="market_segment", y="mean", range_y=[0, 100],
                     labels={"market_segment": "How the booking was made", "mean": "% cancelled"},
                     title="Online travel agency bookings cancel twice as often as direct ones")
    grafico.update_traces(marker_color=[ROJO if s == "Online TA" else GRIS for s in segmentos.index],
                          marker_cornerradius=6, texttemplate="%{y:.0f}%", textposition="outside")
    with derecha.container(border=True):
        st.plotly_chart(estilo(grafico), width="stretch")

with vista3:
    filas = [{"Model": "Baseline (always 'not cancelled')", "Cancellations found": 0.0, "Calls that were right": None, "Bookings to call": 0.0}]
    for nombre, columna in [("Logistic regression", "prob_logistic"), ("Random forest", "prob_forest")]:
        marcada = test[columna] >= 0.3
        filas.append({"Model": nombre, "Cancellations found": recall(marcada),
                      "Calls that were right": precision(marcada), "Bookings to call": marcada.mean()})
    titulo("Each model at the chosen threshold (0.3), on bookings it never saw")
    tabla = pd.DataFrame(filas).set_index("Model")
    st.dataframe(tabla.style.format("{:.1%}", na_rep="-").apply(resaltar, nombre="Random forest", axis=1), width="stretch")

    bosque = tabla.loc["Random forest"]
    c1, c2, c3 = st.columns(3)
    kpi(c1, f"{bosque['Cancellations found']:.1%}", "Cancellations found", "random forest, recall")
    kpi(c2, f"{bosque['Calls that were right']:.1%}", "Calls that were right", "random forest, precision")
    kpi(c3, f"{bosque['Bookings to call']:.1%}", "Bookings to call", "random forest, flagged bookings")
    st.write("")
    caja("The random forest finds more cancellations with fewer calls, so it is the model we use.")
    st.caption("Model choice in the notebook: ROC AUC with 5-fold cross-validation, random forest 0.785 ± 0.002 vs logistic regression 0.734 ± 0.004.")

with vista4:
    with st.container(border=True):
        umbral = st.slider("Call bookings with a risk of at least...", 0.10, 0.90, 0.30, 0.05)
    marcada = test["prob_forest"] >= umbral
    c1, c2, c3 = st.columns(3)
    kpi(c1, f"{recall(marcada):.1%}", "Cancellations found (recall)", f"at threshold {umbral:.2f}")
    kpi(c2, f"{precision(marcada):.1%}", "Calls that were right (precision)", f"at threshold {umbral:.2f}", "rojo")
    kpi(c3, f"{marcada.mean():.1%}", "Bookings to call", f"at threshold {umbral:.2f}", "gris")
    st.write("")

    curva = pd.DataFrame([{"Threshold": u, "Cancellations found": recall(test["prob_forest"] >= u),
                           "Calls that were right": precision(test["prob_forest"] >= u)}
                          for u in [i / 20 for i in range(2, 19)]])
    curva = curva.melt(id_vars="Threshold", var_name="Metric", value_name="Value")
    grafico = px.line(curva, x="Threshold", y="Value", color="Metric", markers=True, color_discrete_sequence=[VERDE, ROJO],
                      range_y=[0, 1], title="Lowering the threshold finds more cancellations, with more false alarms")
    grafico.update_traces(line_width=3, marker_size=8)
    grafico.add_vline(x=umbral, line_dash="dash", line_color=TEXTO_SUAVE,
                      annotation_text=f"  {umbral:.2f}", annotation_position="top right")
    grafico.update_yaxes(tickformat=".0%")
    with st.container(border=True):
        st.plotly_chart(estilo(grafico, 460), width="stretch")

with vista5:
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        recupera = c1.slider("% of revenue at risk recovered by a call", 1, 30, 10) / 100
        minutos = c2.slider("Minutes per call", 2, 30, 10)
        costo_hora = c3.slider("Staff cost per hour (0 = staff with free time)", 0, 40, 10)
    costo_llamada = minutos / 60 * costo_hora
    por_mes = len(datos) / 26 / len(test)

    marcada = test["prob_forest"] >= 0.3
    cancela = test["is_canceled"] == 1
    llamadas = [0, marcada.sum(), len(test)]
    tabla = pd.DataFrame({
        "Option": ["Do nothing (today)", "Call flagged bookings (model)", "Call every booking"],
        "Calls": llamadas,
        "Staff hours": [n * minutos / 60 for n in llamadas],
        "Money recovered": [0, recupera * test.loc[marcada & cancela, "ingreso"].sum(), recupera * test.loc[cancela, "ingreso"].sum()],
        "Cost of calls": [n * costo_llamada for n in llamadas],
    }).set_index("Option")
    tabla["Net benefit"] = tabla["Money recovered"] - tabla["Cost of calls"]
    tabla = tabla * por_mes

    modelo = tabla.loc["Call flagged bookings (model)", "Net benefit"]
    todos = tabla.loc["Call every booking", "Net benefit"]
    ganador = "Call every booking" if todos > modelo else "Call flagged bookings (model)"

    titulo("Per month, for both hotels",
           f"Each call costs {costo_llamada:.2f} ({minutos} minutes at {costo_hora} per hour). Net benefit = money recovered − cost of the calls.")
    st.dataframe(tabla.style.format("{:,.0f}").apply(resaltar, nombre=ganador, axis=1), width="stretch")

    izquierda, derecha = st.columns(2)
    grafico = px.bar(tabla.reset_index(), x="Option", y="Calls", text_auto=",.0f", color="Option",
                     color_discrete_sequence=OPCIONES, title="Calls per month")
    grafico.update_traces(marker_cornerradius=6, textposition="outside")
    grafico.update_layout(showlegend=False, xaxis_title="")
    envolver(grafico, tabla.index)
    with izquierda.container(border=True):
        st.plotly_chart(estilo(grafico), width="stretch")

    grafico = px.bar(tabla.reset_index(), x="Option", y="Net benefit", text_auto=",.0f", color="Option",
                     color_discrete_sequence=OPCIONES, title="Net benefit per month")
    grafico.update_traces(marker_cornerradius=6, textposition="outside")
    grafico.update_layout(showlegend=False, xaxis_title="")
    envolver(grafico, tabla.index)
    with derecha.container(border=True):
        st.plotly_chart(estilo(grafico), width="stretch")

    mejor = "calling every booking" if todos > modelo else "using the model"
    if todos > modelo:
        caja(f"<b>Best option with these assumptions: {mejor}</b> (difference of {abs(todos - modelo):,.0f} a month).<br>"
             "<b>Why:</b> a call is cheap, so a wrong guess costs almost nothing and calling everyone pays off. "
             "The model is worth it when a wrong guess is costly, like asking a guest who was going to stay for a deposit.")
    else:
        caja(f"<b>Best option with these assumptions: {mejor}</b> (difference of {abs(todos - modelo):,.0f} a month).<br>"
             "<b>Why:</b> at this cost, staff time is expensive, so calling only the riskiest bookings pays off more than calling everyone.")
    st.caption("Scaled from the notebook's test set to an average month (99,156 bookings over 26 months), threshold 0.3. "
               "The three sliders are assumptions, not data. The currency is not stated in the data (most likely euros).")

with vista6:
    caja("<b>Illustration, not a forecast.</b> With a call, a wrong guess costs a few minutes. With a deposit, a guest who was going "
         "to stay may refuse it and book elsewhere. We do not know how many would leave: that is what an A/B test would measure.", "gris")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        recupera_d = c1.slider("% of revenue at risk recovered by the deposit", 1, 30, 10, key="recupera_d") / 100
        se_va = c2.slider("% of good guests who leave because of the deposit", 0, 30, 10, key="se_va") / 100
        umbral_d = c3.slider("Ask a deposit from bookings with a risk of at least...", 0.10, 0.90, 0.50, 0.05, key="umbral_d")
    por_mes = len(datos) / 26 / len(test)
    cancela = test["is_canceled"] == 1

    def deposito(pedir):
        ganado = recupera_d * test.loc[pedir & cancela, "ingreso"].sum()
        perdido = se_va * test.loc[pedir & ~cancela, "ingreso"].sum()
        return [pedir.sum(), (pedir & ~cancela).sum(), ganado, perdido, ganado - perdido]

    nada = pd.Series(False, index=test.index)
    tabla = pd.DataFrame(
        [deposito(nada), deposito(test["prob_forest"] >= umbral_d), deposito(~nada)],
        index=pd.Index(["Do nothing (today)", "Deposit for flagged bookings (model)", "Deposit for every booking"], name="Option"),
        columns=["Deposits asked", "Asked from guests who would stay", "Money recovered", "Money lost (guests who leave)", "Net benefit"],
    ) * por_mes

    modelo = tabla.loc["Deposit for flagged bookings (model)", "Net benefit"]
    todos = tabla.loc["Deposit for every booking", "Net benefit"]
    ganador = tabla["Net benefit"].idxmax()

    titulo("Per month, for both hotels", "Net benefit = money recovered − revenue of the good guests who leave.")
    st.dataframe(tabla.style.format("{:,.0f}").apply(resaltar, nombre=ganador, axis=1), width="stretch")

    izquierda, derecha = st.columns(2)
    grafico = px.bar(tabla.reset_index(), x="Option", y="Net benefit", text_auto=",.0f", color="Option",
                     color_discrete_sequence=OPCIONES, title="Net benefit per month")
    grafico.update_traces(marker_cornerradius=6, textposition="outside")
    grafico.update_layout(showlegend=False, xaxis_title="")
    grafico.add_hline(y=0, line_color="#94A3B8")
    envolver(grafico, tabla.index)
    with izquierda.container(border=True):
        st.plotly_chart(estilo(grafico), width="stretch")

    curva = pd.DataFrame([{"Threshold": u, "Net benefit": deposito(test["prob_forest"] >= u)[4] * por_mes}
                          for u in [i / 20 for i in range(2, 19)]])
    mejor_umbral = curva.loc[curva["Net benefit"].idxmax(), "Threshold"]
    grafico = px.line(curva, x="Threshold", y="Net benefit", markers=True, color_discrete_sequence=[VERDE],
                      title="Model: net benefit by threshold")
    grafico.update_traces(line_width=3, marker_size=8)
    grafico.add_vline(x=umbral_d, line_dash="dash", line_color=TEXTO_SUAVE,
                      annotation_text=f"  {umbral_d:.2f}", annotation_position="top right")
    grafico.add_hline(y=0, line_color="#94A3B8")
    grafico.add_trace(go.Scatter(x=[mejor_umbral], y=[curva["Net benefit"].max()], mode="markers",
                                 marker=dict(size=16, color=VERDE, line=dict(color="white", width=3)),
                                 name="Best threshold", hoverinfo="skip"))
    with derecha.container(border=True):
        st.plotly_chart(estilo(grafico), width="stretch")

    if max(modelo, todos) <= 0:
        texto = "<b>With these assumptions, no deposit pays off:</b> too many good guests leave."
        tono = "rojo"
    else:
        mejor = "asking every booking" if todos > modelo else "asking only the bookings the model flags"
        texto = f"<b>Best option with these assumptions: {mejor}</b> (difference of {abs(todos - modelo):,.0f} a month)."
        tono = ""
    caja(texto + f"<br><b>The more good guests leave, the stricter the threshold should be.</b> "
                 f"With these assumptions, the best threshold is about {mejor_umbral:.2f}.", tono)
    st.caption("Same test set and monthly scaling as tab 5. The sliders are assumptions, not data. A deposit is a costly action, "
               "so the hotel should test it first: half of the flagged bookings get a deposit, half do not, and it compares the results.")
