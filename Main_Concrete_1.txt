import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Lade das vorher trainierte Modell
geladenes_modell = joblib.load('linear_regression_model.pkl')

# Streamlit App
def main():
    st.title("Lineare Regression Vorhersage")

    # Eingabeformular für Features
    st.sidebar.header("Eingabeformular")
    feature1 = st.sidebar.slider("Feature 1", float(X_test_minmax[:, 0].min()), float(X_test_minmax[:, 0].max()))
    feature2 = st.sidebar.slider("Feature 2", float(X_test_minmax[:, 1].min()), float(X_test_minmax[:, 1].max()))

    # Vorhersage mit dem geladenen Modell
    vorhersage = geladenes_modell.predict([[feature1, feature2]])

    # Ergebnisse anzeigen
    st.write("### Vorhersage:")
    st.write(f"Die vorhergesagte Ausgabe ist: {vorhersage[0]:.2f}")

if __name__ == "__main__":
    main()
