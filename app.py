from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    send_file
)

import os
import uuid
import traceback

import pandas as pd

from utils.model import train_random_forest
from utils.plots import (
    feature_importance_plot,
    confusion_matrix_plot,
    class_distribution_plot
)
from utils.shap_utils import generate_shap_plots
from utils.report import create_pdf_report


# ======================================================
# Flask App Configuration
# ======================================================

app = Flask(__name__)

app.secret_key = "ExplainXAI_Secret_Key"

UPLOAD_FOLDER = "uploads"
REPORT_FOLDER = "reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["REPORT_FOLDER"] = REPORT_FOLDER


# ======================================================
# Home Page
# ======================================================

@app.route("/")
def home():

    return render_template("index.html")


# ======================================================
# Upload Dataset
# ======================================================

@app.route("/upload", methods=["POST"])
def upload_dataset():

    try:

        if "dataset" not in request.files:

            flash("Please upload a CSV file.")

            return redirect(url_for("home"))

        file = request.files["dataset"]

        if file.filename == "":

            flash("Please choose a dataset.")

            return redirect(url_for("home"))

        if not file.filename.lower().endswith(".csv"):

            flash("Only CSV files are supported.")

            return redirect(url_for("home"))

        filename = f"{uuid.uuid4()}.csv"

        filepath = os.path.join(

            app.config["UPLOAD_FOLDER"],

            filename

        )

        file.save(filepath)

        session["dataset_path"] = filepath

        df = pd.read_csv(filepath)

        session["columns"] = df.columns.tolist()

        preview = df.head(10).to_html(

            classes="table",

            index=False,

            border=0

        )

        return render_template(

            "result.html",

            preview=preview,

            columns=df.columns.tolist(),

            rows=df.shape[0],

            cols=df.shape[1],

            missing=int(df.isnull().sum().sum())

        )

    except Exception:

        traceback.print_exc()

        flash("Error while uploading dataset.")

        return redirect(url_for("home"))
    # ======================================================
# Train Model
# ======================================================

@app.route("/train", methods=["POST"])
def train_model():

    try:

        target = request.form.get("target")

        if not target:

            flash("Please select a target column.")

            return redirect(url_for("home"))

        filepath = session.get("dataset_path")

        if filepath is None or not os.path.exists(filepath):

            flash("Dataset not found. Please upload again.")

            return redirect(url_for("home"))

        df = pd.read_csv(filepath)

        (
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            accuracy,
            precision,
            recall,
            f1,
            feature_names,
            predictions,
            feature_importance_df

        ) = train_random_forest(

            df,

            target

        )

        # =============================
        # Plotly Graphs
        # =============================

        feature_graph = feature_importance_plot(

            feature_importance_df

        )

        confusion_graph = confusion_matrix_plot(

            y_test,

            predictions

        )

        pie_graph = class_distribution_plot(

            y_test

        )

        # =============================
        # SHAP Plots
        # =============================

        (
            shap_summary,
            shap_bar,
            shap_waterfall

        ) = generate_shap_plots(

            model,

            X_train,

            X_test,

            feature_names

        )

        # =============================
        # PDF Report
        # =============================

        report_name = f"ExplainXAI_Report_{uuid.uuid4().hex[:8]}.pdf"

        report_path = os.path.join(

            app.config["REPORT_FOLDER"],

            report_name

        )

        create_pdf_report(

            report_path,

            accuracy,

            precision,

            recall,

            f1,

            feature_importance_df

        )

        session["report_path"] = report_path

        metrics = {

            "Accuracy": round(accuracy * 100, 2),

            "Precision": round(precision * 100, 2),

            "Recall": round(recall * 100, 2),

            "F1 Score": round(f1 * 100, 2)

        }

        top_features = feature_importance_df.head(10).values.tolist()

        return render_template(

            "graph.html",

            metrics=metrics,

            features=top_features,

            feature_graph=feature_graph,

            confusion_graph=confusion_graph,

            pie_graph=pie_graph,

            shap_summary=shap_summary,

            shap_bar=shap_bar,

            shap_waterfall=shap_waterfall

        )

    except Exception:

        traceback.print_exc()

        flash("Model training failed.")

        return redirect(url_for("home"))
    # ======================================================
# Download PDF Report
# ======================================================

@app.route("/download")
def download_report():

    try:

        report_path = session.get("report_path")

        if report_path is None:

            flash("No report available.")

            return redirect(url_for("home"))

        if not os.path.exists(report_path):

            flash("Report file not found.")

            return redirect(url_for("home"))

        return send_file(

            report_path,

            as_attachment=True,

            download_name="ExplainXAI_Report.pdf"

        )

    except Exception:

        traceback.print_exc()

        flash("Unable to download report.")

        return redirect(url_for("home"))


# ======================================================
# Error Pages
# ======================================================

@app.errorhandler(404)
def page_not_found(error):

    return (
        render_template(
            "index.html"
        ),
        404
    )


@app.errorhandler(500)
def internal_server_error(error):

    traceback.print_exc()

    return (
        render_template(
            "index.html"
        ),
        500
    )


# ======================================================
# Run Application
# ======================================================

if __name__ == "__main__":

    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000

    )