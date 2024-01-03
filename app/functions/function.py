# importing modules
import main
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine


# The database URL must be in a specific format
db_url = "mysql+mysqlconnector://{USER}:{PWD}@{HOST}/{DBNAME}"

db_url = db_url.format(
    USER="root", PWD="password", HOST="localhost", DBNAME="predictive_modelling"
)

engine_global_prices = create_engine(db_url, echo=False)

# fetching database
def fetch_database(
    start_date: str,
    end_date: str,
    commodity_name: str | None = None,
    state: str | None = None
) -> pd.DataFrame:
    start_date = f"'{start_date}'"
    end_date = f"'{end_date}'"

    query = f"SELECT commodity_name, state, district, market, min_price, max_price, modal_price, date FROM commodity_prices WHERE date BETWEEN {start_date} AND {end_date} "

    if commodity_name:
        commodity_name = f"'{commodity_name}'"
        query += f"AND commodity_name = {commodity_name}"

    if state:
        state = f"'{state}'"
        query += f"AND state = {state}"

    with main.engine.begin() as conn:
        commodity_prices = pd.read_sql_query(sql=query, con=conn, parse_dates=["date"])

    return commodity_prices

def fetch_global_prices(
    year: int,
    month: int,
    commodity_name: str,
) -> pd.DataFrame:
    date = f"'{str(year)}-{str(month).zfill(2)}-01'"
    latest_date = "'2023-09-01'"
    query = f"SELECT date, {commodity_name} FROM global_prices WHERE date BETWEEN {date} AND {latest_date}"

    with engine_global_prices.begin() as conn:
        commodity_prices = pd.read_sql_query(sql=query, con=conn, parse_dates=["date"])

    return commodity_prices

def plot_forecast(forecast: pd.DataFrame):
    fig = go.Figure()

    # Add the forecast line
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Predicted Price'
    ))

    # Add the shaded confidence interval
    fig.add_trace(go.Scatter(
        x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence in price prediction'
    ))

    # Add range slider with buttons
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="7 days",
                         step="day",
                         stepmode="backward"),
                    dict(count=1,
                         label="1 month",
                         step="month",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price")
    
    return fig


def plot_predict(forecast: pd.DataFrame):
    fig = go.Figure()

    # Add the forecast line
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Predicted Price'
    ))

    # Add the shaded confidence interval
    fig.add_trace(go.Scatter(
        x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence in Prediction'
    ))
    
    fig.add_trace(
        go.Scatter(
            x = forecast["ds"],
            y = forecast["actual"],
            mode = 'markers',
            name = "Actual Price",
            marker_color='rgba(0, 0, 0, 1)',
            marker_size = 3
        )
    )

    # Add range slider with buttons
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=7,
                         label="7 days",
                         step="day",
                         stepmode="backward"),
                    dict(count=1,
                         label="1 month",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6 month",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="1 year",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price")
    
    return fig