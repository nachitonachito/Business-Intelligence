from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

CARPETA = Path(__file__).parent
AZUL = "#2a78d6"
NARANJO = "#eb6834"

st.set_page_config(page_title="Hotel cancellations", layout="wide")

datos = pd.read_csv(CARPETA / "datos_limpios.csv")
test = pd.read_csv(CARPETA / "resultados_test.csv")
datos["ingreso"] = datos["adr"] * (datos["stays_in_weekend_nights"] + datos["stays_in_week_nights"])


def recall(marcada):
    return (marcada & (test["is_canceled"] == 1)).sum() / test["is_canceled"].sum()


def precision(marcada):
    return test.loc[marcada, "is_canceled"].mean() if marcada.any() else 0


st.title("Which bookings will be cancelled?")
st.caption("Two hotels, arrivals from July 2015 to August 2017. Results from the notebook's test set (29,823 bookings).")

vista1, vista2, vista3, vista4, vista5, vista6 = st.tabs(["1. Overview", "2. Who cancels", "3. Models", "4. Threshold", "5. Business impact", "6. Deposit (illustration)"])

with vista1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bookings", f"{len(datos):,}")
    c2.metric("Cancelled", f"{datos['is_canceled'].mean():.1%}")
    c3.metric("Revenue of cancelled bookings", f"{datos.loc[datos['is_canceled'] == 1, 'ingreso'].sum() / 1e6:.1f} M")
    c4.metric("Baseline accuracy", f"{1 - datos['is_canceled'].mean():.1%}", help="Always predicting 'not cancelled'. It finds no cancellations.")
    st.markdown("**The decision:** the hotel loses money when bookings cancel. Can a model tell it which bookings to act on, and is that better than contacting every guest?")

with vista2:
    datos["anticipacion"] = pd.cut(datos["lead_time"], bins=[-1, 7, 30, 90, 180, 365, 1000],
                                   labels=["0-7", "8-30", "31-90", "91-180", "181-365", "366+"])
    izquierda, derecha = st.columns(2)

    tabla = datos.groupby("anticipacion", observed=True)["is_canceled"].mean().mul(100).reset_index()
    grafico = px.bar(tabla, x="anticipacion", y="is_canceled", color_discrete_sequence=[AZUL], range_y=[0, 100],
                     labels={"anticipacion": "Days between booking and arrival", "is_canceled": "% cancelled"},
                     title="Bookings made far ahead cancel more")
    izquierda.plotly_chart(grafico, width="stretch")

    segmentos = datos.groupby("market_segment")["is_canceled"].agg(["size", "mean"])
    segmentos = segmentos[segmentos["size"] >= 1000].sort_values("mean", ascending=False)
    segmentos["mean"] = segmentos["mean"] * 100
    grafico = px.bar(segmentos.reset_index(), x="market_segment", y="mean", color_discrete_sequence=[AZUL], range_y=[0, 100],
                     labels={"market_segment": "How the booking was made", "mean": "% cancelled"},
                     title="Online travel agency bookings cancel twice as often as direct ones")
    derecha.plotly_chart(grafico, width="stretch")

with vista3:
    filas = [{"Model": "Baseline (always 'not cancelled')", "Cancellations found": 0.0, "Calls that were right": None, "Bookings to call": 0.0}]
    for nombre, columna in [("Logistic regression", "prob_logistic"), ("Random forest", "prob_forest")]:
        marcada = test[columna] >= 0.3
        filas.append({"Model": nombre, "Cancellations found": recall(marcada),
                      "Calls that were right": precision(marcada), "Bookings to call": marcada.mean()})
    st.markdown("**Each model at the chosen threshold (0.3), on bookings it never saw:**")
    st.dataframe(pd.DataFrame(filas).set_index("Model").style.format("{:.1%}", na_rep="-"), width="stretch")
    st.markdown("The random forest finds more cancellations with fewer calls, so it is the model we use.")
    st.caption("Model choice in the notebook: ROC AUC with 5-fold cross-validation, random forest 0.785 ± 0.002 vs logistic regression 0.734 ± 0.004.")

with vista4:
    umbral = st.slider("Call bookings with a risk of at least...", 0.10, 0.90, 0.30, 0.05)
    marcada = test["prob_forest"] >= umbral
    c1, c2, c3 = st.columns(3)
    c1.metric("Cancellations found (recall)", f"{recall(marcada):.1%}")
    c2.metric("Calls that were right (precision)", f"{precision(marcada):.1%}")
    c3.metric("Bookings to call", f"{marcada.mean():.1%}")

    curva = pd.DataFrame([{"Threshold": u, "Cancellations found": recall(test["prob_forest"] >= u),
                           "Calls that were right": precision(test["prob_forest"] >= u)}
                          for u in [i / 20 for i in range(2, 19)]])
    curva = curva.melt(id_vars="Threshold", var_name="Metric", value_name="Value")
    grafico = px.line(curva, x="Threshold", y="Value", color="Metric", markers=True, color_discrete_sequence=[AZUL, NARANJO],
                      range_y=[0, 1], title="Lowering the threshold finds more cancellations, with more false alarms")
    grafico.add_vline(x=umbral, line_dash="dash", line_color="gray")
    grafico.update_yaxes(tickformat=".0%")
    st.plotly_chart(grafico, width="stretch")

with vista5:
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

    st.markdown(f"**Per month, for both hotels.** Each call costs {costo_llamada:.2f} ({minutos} minutes at {costo_hora} per hour). "
                "Net benefit = money recovered − cost of the calls.")
    st.dataframe(tabla.style.format("{:,.0f}"), width="stretch")

    colores = ["#9a9a9a", AZUL, NARANJO]
    izquierda, derecha = st.columns(2)
    grafico = px.bar(tabla.reset_index(), x="Option", y="Calls", text_auto=",.0f", color="Option",
                     color_discrete_sequence=colores, title="Calls per month")
    grafico.update_layout(showlegend=False, xaxis_title="")
    izquierda.plotly_chart(grafico, width="stretch")

    grafico = px.bar(tabla.reset_index(), x="Option", y="Net benefit", text_auto=",.0f", color="Option",
                     color_discrete_sequence=colores, title="Net benefit per month")
    grafico.update_layout(showlegend=False, xaxis_title="")
    derecha.plotly_chart(grafico, width="stretch")

    modelo = tabla.loc["Call flagged bookings (model)", "Net benefit"]
    todos = tabla.loc["Call every booking", "Net benefit"]
    mejor = "calling every booking" if todos > modelo else "using the model"
    st.markdown(f"**Best option with these assumptions: {mejor}** (difference of {abs(todos - modelo):,.0f} a month).")
    if todos > modelo:
        st.markdown("**Why:** a call is cheap, so a wrong guess costs almost nothing and calling everyone pays off. "
                    "The model is worth it when a wrong guess is costly, like asking a guest who was going to stay for a deposit.")
    else:
        st.markdown("**Why:** at this cost, staff time is expensive, so calling only the riskiest bookings pays off more than calling everyone.")
    st.caption("Scaled from the notebook's test set to an average month (99,156 bookings over 26 months), threshold 0.3. "
               "The three sliders are assumptions, not data. The currency is not stated in the data (most likely euros).")

with vista6:
    st.info("**Illustration, not a forecast.** With a call, a wrong guess costs a few minutes. With a deposit, a guest who was going "
            "to stay may refuse it and book elsewhere. We do not know how many would leave: that is what an A/B test would measure.")
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

    st.markdown("**Per month, for both hotels.** Net benefit = money recovered − revenue of the good guests who leave.")
    st.dataframe(tabla.style.format("{:,.0f}"), width="stretch")

    izquierda, derecha = st.columns(2)
    grafico = px.bar(tabla.reset_index(), x="Option", y="Net benefit", text_auto=",.0f", color="Option",
                     color_discrete_sequence=["#9a9a9a", AZUL, NARANJO], title="Net benefit per month")
    grafico.update_layout(showlegend=False, xaxis_title="")
    izquierda.plotly_chart(grafico, width="stretch")

    curva = pd.DataFrame([{"Threshold": u, "Net benefit": deposito(test["prob_forest"] >= u)[4] * por_mes}
                          for u in [i / 20 for i in range(2, 19)]])
    grafico = px.line(curva, x="Threshold", y="Net benefit", markers=True, color_discrete_sequence=[AZUL],
                      title="Model: net benefit by threshold")
    grafico.add_vline(x=umbral_d, line_dash="dash", line_color="gray")
    grafico.add_hline(y=0, line_color="lightgray")
    derecha.plotly_chart(grafico, width="stretch")

    modelo = tabla.loc["Deposit for flagged bookings (model)", "Net benefit"]
    todos = tabla.loc["Deposit for every booking", "Net benefit"]
    mejor_umbral = curva.loc[curva["Net benefit"].idxmax(), "Threshold"]
    if max(modelo, todos) <= 0:
        st.markdown("**With these assumptions, no deposit pays off:** too many good guests leave.")
    else:
        mejor = "asking every booking" if todos > modelo else "asking only the bookings the model flags"
        st.markdown(f"**Best option with these assumptions: {mejor}** (difference of {abs(todos - modelo):,.0f} a month).")
    st.markdown(f"**The more good guests leave, the stricter the threshold should be.** With these assumptions, the best threshold is about {mejor_umbral:.2f}.")
    st.caption("Same test set and monthly scaling as tab 5. The sliders are assumptions, not data. A deposit is a costly action, "
               "so the hotel should test it first: half of the flagged bookings get a deposit, half do not, and it compares the results.")
