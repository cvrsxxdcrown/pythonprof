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

# Столбцы и описание
COLUMNS = {
    'CRIM': "Уровень преступности в районе (чем выше, тем хуже)",
    'ZN': "Доля жилой земли под большие дома (>25 тыс. кв. футов)",
    'INDUS': "Доля земли под промышленность (чем выше, тем хуже)",
    'CHAS': "Находится ли рядом река Чарльз (1 = да, 0 = нет)",
    'NOX': "Уровень загрязнения воздуха (NOx, чем выше — хуже)",
    'RM': "Среднее количество комнат в доме (чем больше — дороже)",
    'AGE': "% домов, построенных до 1940 (старые дома)",
    'DIS': "Расстояние до деловых центров (ближе — дороже)",
    'RAD': "Доступ к шоссе (чем больше — выше шум, но и удобство)",
    'TAX': "Налоговая ставка на имущество (чем выше — хуже)",
    'PTRATIO': "Соотношение учеников к учителям в школе",
    'B': "Чёрное население (специфический исторический индекс)",
    'LSTAT': "% бедного населения (чем выше — ниже цена)"
}

@st.cache_resource
def train_and_save_model():
    df = pd.read_csv(CSV_PATH)
    X = df[list(COLUMNS.keys())]
    y = df['MEDV']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
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
    st.set_page_config(page_title="💵 Стоимость домов в Бостоне", layout="wide")

    st.title("🏠 Предсказание стоимости дома в Бостоне")
    st.markdown("Модель машинного обучения предсказывает цену дома в $1000 по данным о районе и характеристикам.")

    model, rmse, df = load_model()

    tab1, tab2 = st.tabs(["📈 Анализ и графики", "🤖 Предсказание цены"])

    # 📈 Анализ
    with tab1:
        st.header("🔍 Описание параметров и их влияние")

        for key, desc in COLUMNS.items():
            st.markdown(f"**{key}** — {desc}")

        st.divider()
        st.subheader("📌 Выберите тип графика для анализа")

        analysis_options = [
            "Корреляционная матрица",
            "Распределение каждого признака",
            "Зависимость признаков от цены"
        ]
        selected_analysis = st.selectbox("Выберите график:", analysis_options)

        if selected_analysis == "Корреляционная матрица":
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
            ax_corr.set_title("Корреляционная матрица признаков и цены")
            st.pyplot(fig_corr)

        elif selected_analysis == "Распределение каждого признака":
            selected_feature = st.selectbox("Выберите признак:", list(COLUMNS.keys()))
            fig_feat, ax_feat = plt.subplots()
            sns.histplot(df[selected_feature], bins=30, kde=True, ax=ax_feat, color="lightblue")
            ax_feat.set_title(f"{selected_feature} — {COLUMNS[selected_feature]}")
            st.pyplot(fig_feat)

        elif selected_analysis == "Зависимость признаков от цены":
            selected_feature = st.selectbox("Выберите признак:", list(COLUMNS.keys()))
            fig_scatter, ax_scatter = plt.subplots()
            sns.scatterplot(data=df, x=selected_feature, y='MEDV', ax=ax_scatter)
            ax_scatter.set_title(f"{selected_feature} vs MEDV (цена дома)")
            st.pyplot(fig_scatter)

    # 🤖 Предсказание
    with tab2:
        st.header("📊 Введите данные дома")
        input_data = {}
        col1, col2, col3 = st.columns(3)

        for i, (key, desc) in enumerate(COLUMNS.items()):
            col = [col1, col2, col3][i % 3]
            min_val = float(df[key].min())
            max_val = float(df[key].max())
            mean_val = float(df[key].mean())
            step = (max_val - min_val) / 100

            if key == 'CHAS':
                input_data[key] = col.selectbox(f"{key} — {desc}", [0, 1], index=0)
            else:
                input_data[key] = col.slider(f"{key} — {desc}", min_value=min_val, max_value=max_val, value=mean_val, step=step)

        st.divider()

        if st.button("🔍 Предсказать цену"):
            st.session_state.prediction_result = model.predict(pd.DataFrame([input_data]))[0]
            st.session_state.input_data = input_data

        if "prediction_result" in st.session_state:
            prediction = st.session_state.prediction_result
            input_data = st.session_state.input_data

            st.success(f"💰 Оценочная стоимость: **${prediction * 1000:.2f}**")

            # 🔁 Сравнение с общими данными
            st.subheader("📊 Сравнение введённого значения с распределением признака")

            selected_feature = st.selectbox("Выберите признак для сравнения:", list(input_data.keys()))
            selected_value = input_data[selected_feature]

            fig, ax = plt.subplots()
            sns.histplot(df[selected_feature], bins=30, kde=True, ax=ax, color="lightblue")
            ax.axvline(selected_value, color="red", linestyle="--", label="Ваш ввод")
            ax.set_title(f"{selected_feature} — {COLUMNS[selected_feature]}")
            ax.legend()
            st.pyplot(fig)


if __name__ == "__main__":
    main()
