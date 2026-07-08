import plotly.express as px

import plotly.figure_factory as ff

from sklearn.metrics import confusion_matrix

import pandas as pd


def feature_importance_plot(feature_df):

    fig = px.bar(

        feature_df.head(15),

        x="Importance",

        y="Feature",

        orientation="h",

        color="Importance",

        title="Top Feature Importance"

    )

    fig.update_layout(

        template="plotly_white",

        height=550,

        yaxis=dict(autorange="reversed"),

        margin=dict(

            l=40,

            r=20,

            t=60,

            b=40

        )

    )

    return fig.to_html(

        full_html=False,

        include_plotlyjs="cdn"

    )


def confusion_matrix_plot(

    y_true,

    y_pred

):

    labels = sorted(

        list(

            set(y_true)

        )

    )

    matrix = confusion_matrix(

        y_true,

        y_pred

    )

    fig = ff.create_annotated_heatmap(

        z=matrix,

        x=labels,

        y=labels,

        colorscale="Blues",

        showscale=True

    )

    fig.update_layout(

        title="Confusion Matrix",

        xaxis_title="Predicted",

        yaxis_title="Actual",

        height=500

    )

    return fig.to_html(

        full_html=False,

        include_plotlyjs="cdn"

    )


def class_distribution_plot(

    y

):

    counts = pd.Series(

        y

    ).value_counts()

    fig = px.pie(

        names=counts.index.astype(str),

        values=counts.values,

        hole=0.45,

        title="Class Distribution"

    )

    fig.update_layout(

        height=500,

        template="plotly_white"

    )

    return fig.to_html(

        full_html=False,

        include_plotlyjs="cdn"

    )