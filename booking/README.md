# Hotel booking cancellations

IIB415A Business Intelligence, integrative project.

We predict which hotel bookings will be cancelled and check whether that helps the hotel make money.

## What is in the folder

- `proyecto_booking.ipynb`: the notebook with the full analysis, answering the seven questions in order.
- `hotel_bookings.csv`: the dataset.
- `dashboardv2.py`: the dashboard.
- `dashboard_antiguo(ignorar).py`: an older version, kept only for reference.
- `resultados_test.csv` and `datos_limpios.csv`: files the notebook creates for the dashboard.
- The slides, as a PDF.

## Data

Hotel Booking Demand, from Kaggle. It comes from the article "Hotel Booking Demand Datasets" (Antonio, Almeida & Nunes, *Data in Brief*, 2019). Two hotels, arrivals from July 2015 to August 2017, one row per booking.

## How to run it

We used Python 3.15. Install the libraries:

```
pip install pandas scikit-learn matplotlib streamlit plotly
```

Open the notebook and run all cells from the top. It takes about 10 minutes, mostly the cross-validation. All paths are relative and the seed is fixed (`random_state=42`), so the results are the same on any machine.

The last cell of the notebook saves the two CSV files the dashboard reads. Then, from this folder:

```
streamlit run dashboardv2.py --theme.base light
```

## Use of AI

We used Claude (Anthropic) as an assistant during the project. We used it to understand concepts (metrics, leakage, cross-validation), to review our code and find mistakes, and to write parts of the code, especially the dashboard. We discussed and agreed on every decision, and checked every result in the notebook.
