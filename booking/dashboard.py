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

vista1, vista2, vista3, vista4, vista5 = st.tabs(["1. Overview", "2. Who cancels", "3. Models", "4. Threshold", "5. Business impact"])

with vista1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bookings", f"{len(datos):,}")
    c2.metric("Cancelled", f"{datos['is_canceled'].mean():.1%}")
    c3.metric("Revenue of cancelled bookings", f"{datos.loc[datos['is_canceled'] == 1, 'ingreso'].sum() / 1e6:.1f} M")
    c4.metric("Baseline accuracy", f"{1 - datos['is_canceled'].mean():.1%}", help="Always predicting 'not cancelled'. It finds no cancellations.")
    st.markdown("**The decision:** which bookings should the front office call to reconfirm, so the hotel can act before losing the room?")

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
    izquierda, derecha = st.columns(2)
    recupera = izquierda.slider("% of revenue at risk recovered by a call", 1, 30, 10) / 100
    minutos = derecha.slider("Minutes per call (staff time at 30 per hour)", 2, 30, 10)
    costo_llamada = minutos / 60 * 30

    marcada = test["prob_forest"] >= 0.3
    cancela = test["is_canceled"] == 1
    opciones = pd.DataFrame({
        "Option": ["Call flagged bookings (model)", "Call every booking"],
        "Staff hours": [marcada.sum() * minutos / 60, len(test) * minutos / 60],
        "Net benefit": [
            recupera * test.loc[marcada & cancela, "ingreso"].sum() - costo_llamada * marcada.sum(),
            recupera * test.loc[cancela, "ingreso"].sum() - costo_llamada * len(test),
        ],
    })

    c1, c2, c3 = st.columns(3)
    c1.metric("Calls with the model", f"{marcada.sum():,}", f"{marcada.sum() - len(test):,} vs calling everyone")
    c2.metric("Staff hours saved", f"{opciones['Staff hours'][1] - opciones['Staff hours'][0]:,.0f}")
    c3.metric("Net benefit with the model", f"{opciones['Net benefit'][0]:,.0f}",
              f"{opciones['Net benefit'][0] - opciones['Net benefit'][1]:,.0f} vs calling everyone")

    izquierda, derecha = st.columns(2)
    grafico = px.bar(opciones, x="Option", y="Staff hours", text_auto=",.0f", color="Option",
                     color_discrete_sequence=[AZUL, NARANJO], title="Staff hours needed")
    grafico.update_layout(showlegend=False)
    izquierda.plotly_chart(grafico, width="stretch")

    grafico = px.bar(opciones, x="Option", y="Net benefit", text_auto=",.0f", color="Option",
                     color_discrete_sequence=[AZUL, NARANJO], title="Net benefit")
    grafico.update_layout(showlegend=False)
    derecha.plotly_chart(grafico, width="stretch")

    st.markdown("**The model reaches about the same money with less than half the calls.** "
                "If calls are short or staff is plentiful, calling everyone earns slightly more; if staff time is limited, the model is the way to choose whom to call.")
    st.caption("Test set (29,823 bookings), threshold 0.3. Assumption: a call recovers the chosen % of the revenue at risk. "
               "The currency is not stated in the data (most likely euros).")
