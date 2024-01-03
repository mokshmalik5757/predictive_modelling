from flask import Flask, request, render_template, jsonify
from sqlalchemy import create_engine
import functions.function as ff
import plotly.express as px
from flask_cors import CORS
import warnings
from prophet import Prophet


warnings.filterwarnings("ignore")

# The database URL must be in a specific format
db_url = "mysql+mysqlconnector://{USER}:{PWD}@{HOST}/{DBNAME}"

db_url = db_url.format(
    USER="root", PWD="password", HOST="localhost", DBNAME="linear_regression"
)

engine = create_engine(db_url, echo=False)

app = Flask(__name__)

CORS(app, resources={r"/charts": {"origins": "http://127.0.0.1:5000"}})


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"response": "Server is fine"})


@app.route("/charts", methods=["GET", "POST"])  # type: ignore
def create_charts():
    if request.method == "GET":
        return render_template("charts.html")

    if request.method == "POST":
        start_date = request.form["start_date"]
        end_date = request.form["end_date"]
        commodity_name = request.form.get("commodity_name")
        state = request.form.get("state")

        df = ff.fetch_database(
            start_date=start_date,
            end_date=end_date,
            state=state,
            commodity_name=commodity_name,
        )
        if commodity_name and state == "":
            state_group_df = (
                df.groupby(by="state")[["modal_price"]].agg("mean").reset_index()
            )
            fig = px.bar(
                data_frame=state_group_df,
                x="state",
                y="modal_price",
                color="state",
                title=f"Prices for {commodity_name} in different states",
            )
            fig.update_xaxes(title_text="States")
            fig.update_yaxes(title_text="Modal Price")
 
            state_chart_json = fig.to_json()
            return jsonify({"state_chart_json": state_chart_json})

        if state and commodity_name == "":
            commodity_name_df = (
                df.groupby(by="commodity_name")[["modal_price"]]
                .agg("mean")
                .reset_index()
            )

            fig = px.bar(
                data_frame=commodity_name_df,
                x="commodity_name",
                y="modal_price",
                title=f"Prices in {state} for different commodities",
            )

            fig.update_xaxes(title_text="Commodity")
            fig.update_yaxes(title_text="Modal Price")
            commodity_name_chart_json = fig.to_json()

            return jsonify({"commodity_name_chart_json": commodity_name_chart_json})

        if not state and not commodity_name:
            fig_hist = px.histogram(
                data_frame=df, x="modal_price", title="Price Distribution"
            )
            fig_box = px.box(
                data_frame=df,
                y="modal_price",
                title="Price Spread and Central Tendency",
            )

            fig_box.update_yaxes(title_text="Modal Price")

            fig_hist.update_xaxes(title_text="Modal Price")
            fig_hist.update_yaxes(title_text="Count")

            return jsonify(
                {
                    "not_state_not_commodity_json": {
                        "fig_hist": fig_hist.to_json(),
                        "fig_box": fig_box.to_json(),
                    }
                }
            )

        if state and commodity_name:
            fig_scatter = px.scatter(
                df,
                x="date",
                y="modal_price",
                color="district",
                title=f"Price Trend of {commodity_name} in {state}",
            )
            fig_bar = px.bar(
                df,
                x="date",
                y="modal_price",
                color="district",
                title=f"Average Prices of {commodity_name} in Different Markets of {state}",
            )
            filtered_df = (
                df[["date", "commodity_name", "state", "modal_price"]]
                .groupby(by=["state", "commodity_name", "date"])[["modal_price"]]
                .agg("mean")
                .reset_index()
            )
            fig_line = px.line(
                data_frame=filtered_df,
                x="date",
                y="modal_price",
                title=f"Price of {commodity_name} in {state} within date range {start_date} to {end_date}",
            )

            fig_scatter.update_xaxes(title_text="Date")
            fig_scatter.update_yaxes(title_text="Modal Price")

            fig_bar.update_xaxes(title_text="Date")
            fig_bar.update_yaxes(title_text="Modal Price")

            fig_line.update_xaxes(title_text="Date")
            fig_line.update_yaxes(title_text="Modal Price")

            # Forecasting models
            model = Prophet()

            prophet_df = filtered_df.drop(["state", "commodity_name"], axis=1).rename(
                columns={"date": "ds", "modal_price": "y"}
            )
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=30, include_history=False)
            forecast = model.predict(future)
            fig_forecast = ff.plot_forecast(forecast=forecast)

            return jsonify(
                {
                    "state_commodity_json": {
                        "fig_scatter": fig_scatter.to_json(),
                        "fig_bar": fig_bar.to_json(),
                        "fig_line": fig_line.to_json(),
                        "fig_forecast": fig_forecast.to_json(),
                    }
                }
            )


@app.route("/predict", methods=["GET", "POST"])  # type: ignore
def predict_view():
    if request.method == "GET":
        return render_template("predict.html")

    if request.method == "POST":
        year = int(request.form["year"])
        month = int(request.form["month"])
        commodity_name = request.form["commodity_name"]

        df = ff.fetch_global_prices(
            year=year, month=month, commodity_name=commodity_name
        )

        fig = px.line(
            data_frame=df,
            x="date",
            y=commodity_name,
            title=f"Trend in Prices of {commodity_name.title()} from {str(year)}-{str(month).zfill(2)}-01 to 2023-09-01",
        )
        fig.update_xaxes(title="Date")
        fig.update_yaxes(title=f"{commodity_name.title()} Prices")

        # Prophet Model
        model = Prophet()
        prophet_df = df.rename(columns={"date": "ds", f"{commodity_name}": "y"})
        model.fit(prophet_df)
        period = 365
        future = model.make_future_dataframe(periods=period)
        forecast = model.predict(future)
        forecast["actual"] = prophet_df["y"]
        fig_forecast = ff.plot_predict(forecast=forecast)

        return jsonify(
            {
                "plots": {
                    "past_trend": fig.to_json(),
                    "predict_forecast": fig_forecast.to_json(),
                }
            }
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
