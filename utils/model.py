import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def train_random_forest(df, target_column):

    df = df.copy()

    # -----------------------------
    # Fill Missing Values
    # -----------------------------

    for column in df.columns:

        if df[column].dtype == "object":

            mode = df[column].mode()

            if len(mode) > 0:
                df[column] = df[column].fillna(mode[0])
            else:
                df[column] = df[column].fillna("Unknown")

        else:

            df[column] = df[column].fillna(
                df[column].median()
            )

    # -----------------------------
    # Encode Categorical Columns
    # -----------------------------

    encoders = {}

    for column in df.columns:

        if df[column].dtype == "object":

            encoder = LabelEncoder()

            df[column] = encoder.fit_transform(
                df[column].astype(str)
            )

            encoders[column] = encoder

    X = df.drop(columns=[target_column])
    y = df[target_column]

    feature_names = X.columns.tolist()

    # -----------------------------
    # Train/Test Split
    # -----------------------------

    try:

        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            test_size=0.20,

            random_state=42,

            stratify=y

        )

    except ValueError:

        # fallback when stratify isn't possible

        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            test_size=0.20,

            random_state=42

        )

    # -----------------------------
    # Model
    # -----------------------------

    model = RandomForestClassifier(

        n_estimators=200,

        random_state=42,

        n_jobs=-1

    )

    model.fit(

        X_train,

        y_train

    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    importance = pd.DataFrame({

        "Feature": feature_names,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(

        by="Importance",

        ascending=False

    )

    return (

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

        importance

    )