import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Пути
CSV_PATH = "boston.csv"
MODEL_PATH = "model_weights.mw"

# Столбцы
COLUMNS = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
    'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
]


@st.cache_resource
def train_and_save_model():
    df = pd.read_csv(CSV_PATH)
    X = df[COLUMNS]
    y = df['MEDV']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    return model, rmse, df


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return train_and_save_model()
    else:
        df = pd.read_csv(CSV_PATH)
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model, None, df


def main():
    st.set_page_config(page_title="💵 Boston House Price Prediction", layout="wide")

    st.title("🏠 Предсказание стоимости домов в Бостоне")
    st.markdown("**Простой ML-проект на Streamlit с визуализациями и вводом признаков**")

    model, rmse, df = load_model()

    # --- Колонки
    tab1, tab2 = st.tabs(["📈 Анализ данных", "🤖 Машинное обучение"])

    # 📊 Анализ данных
    with tab1:
        st.subheader("🎯 Корреляция признаков с ценой")
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.subheader("💵 Распределение цен (MEDV)")
        fig2, ax2 = plt.subplots()
        sns.histplot(df['MEDV'], bins=30, kde=True, ax=ax2, color='skyblue')
        st.pyplot(fig2)

        st.subheader("📌 Важность признаков")
        importances = model.feature_importances_
        imp_df = pd.DataFrame({
            'Feature': COLUMNS,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        fig3, ax3 = plt.subplots()
        sns.barplot(data=imp_df, x='Importance', y='Feature', ax=ax3, palette='viridis')
        st.pyplot(fig3)

    # 🤖 ML Предсказание
    with tab2:
        st.header("🔢 Введите параметры дома")

        input_data = {}
        col1, col2, col3 = st.columns(3)

        with col1:
            input_data['CRIM'] = st.number_input('CRIM (уровень преступности)', value=0.1)
            input_data['ZN'] = st.number_input('ZN (жилые зоны)', value=0.0)
            input_data['INDUS'] = st.number_input('INDUS (нежилой бизнес)', value=7.0)
            input_data['CHAS'] = st.selectbox('CHAS (возле реки)', [0, 1])
            input_data['NOX'] = st.number_input('NOX (загрязнение воздуха)', value=0.5)

        with col2:
            input_data['RM'] = st.number_input('RM (среднее количество комнат)', value=6.0)
            input_data['AGE'] = st.number_input('AGE (% старых домов)', value=65.0)
            input_data['DIS'] = st.number_input('DIS (расстояние до работы)', value=4.0)
            input_data['RAD'] = st.number_input('RAD (доступ к шоссе)', value=4)
            input_data['TAX'] = st.number_input('TAX (налоги)', value=300.0)

        with col3:
            input_data['PTRATIO'] = st.number_input('PTRATIO (соотношение ученик/учитель)', value=18.0)
            input_data['B'] = st.number_input('B (индекс чернокожих)', value=390.0)
            input_data['LSTAT'] = st.number_input('LSTAT (% бедных)', value=10.0)

        if st.button("🔍 Предсказать цену"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            st.success(f"💰 Предсказанная стоимость дома: **${prediction * 1000:.2f}**")


if __name__ == "__main__":
    main()
