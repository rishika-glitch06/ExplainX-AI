import shap
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import base64

from io import BytesIO


def image_to_base64():

    buffer = BytesIO()

    plt.savefig(
        buffer,
        format="png",
        bbox_inches="tight"
    )

    plt.close()

    buffer.seek(0)

    image = base64.b64encode(
        buffer.read()
    ).decode("utf-8")

    buffer.close()

    return image


def generate_shap_plots(

    model,

    X_train,

    X_test,

    feature_names

):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_test)

    # ------------------------------------
    # SHAP Compatibility
    # ------------------------------------

    if isinstance(shap_values, list):

        values = shap_values[1]

        base_value = explainer.expected_value[1]

    elif len(np.array(shap_values).shape) == 3:

        values = shap_values[:, :, 1]

        if isinstance(explainer.expected_value, np.ndarray):

            base_value = explainer.expected_value[1]

        else:

            base_value = explainer.expected_value

    else:

        values = shap_values

        base_value = explainer.expected_value

    # ------------------------------------
    # Summary Plot
    # ------------------------------------

    plt.figure(figsize=(10, 6))

    shap.summary_plot(

        values,

        X_test,

        show=False

    )

    summary_plot = image_to_base64()

    # ------------------------------------
    # Bar Plot
    # ------------------------------------

    plt.figure(figsize=(10, 6))

    shap.summary_plot(

        values,

        X_test,

        plot_type="bar",

        show=False

    )

    bar_plot = image_to_base64()

    # ------------------------------------
    # Waterfall Plot
    # ------------------------------------

    explanation = shap.Explanation(

        values=values[0],

        base_values=base_value,

        data=X_test.iloc[0],

        feature_names=feature_names

    )

    plt.figure(figsize=(10, 8))

    shap.plots.waterfall(

        explanation,

        show=False

    )

    waterfall_plot = image_to_base64()

    return (

        summary_plot,

        bar_plot,

        waterfall_plot

    )