import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).resolve().parent

LABELS = {
    'fixed_acidity': 'Нелетучие кислоты, г/л',
    'volatile_acidity': 'Летучие кислоты, г/л',
    'citric_acid': 'Лимонная кислота, г/л',
    'residual_sugar': 'Остаточный сахар, г/л',
    'chlorides': 'Хлориды, г/л',
    'free_sulfur_dioxide': 'Свободный SO₂, мг/л',
    'total_sulfur_dioxide': 'Общий SO₂, мг/л',
    'density': 'Плотность, г/см³',
    'pH': 'pH',
    'sulphates': 'Сульфаты, г/л',
    'alcohol': 'Алкоголь, %',
    'is_red': 'Красное вино'
}
WINE_COLOR = '#8B2845'


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


@st.cache_data
def wine_proba(_model, row):
    return float(_model.predict_proba(row)[0, 1])


def validate_batch(frame, feature_names):
    """Возвращает числовые признаки в нужном порядке или понятную ошибку."""
    if frame.empty:
        raise ValueError('В CSV нет строк с данными.')
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError('Не хватает столбцов: ' + ', '.join(missing))
    if 'target' in frame.columns:
        raise ValueError('Удалите столбец target: для прогноза нужны только признаки.')
    try:
        values = frame[feature_names].apply(pd.to_numeric, errors='raise')
    except (ValueError, TypeError):
        raise ValueError('Все признаки должны быть числами. Проверьте разделитель и значения.')
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError('В признаках есть пропуски или бесконечные значения. Заполните их в CSV.')
    if not values['is_red'].isin([0, 1]).all():
        raise ValueError('В столбце is_red допустимы только 0 (белое) и 1 (красное).')
    return values


def batch_prediction(model, frame, feature_names, threshold):
    values = validate_batch(frame, feature_names)
    result = frame.copy()
    result['probability_good'] = model.predict_proba(values)[:, 1]
    result['recommend'] = (result['probability_good'] >= threshold).astype(int)
    return result


def main():
    st.set_page_config(page_title='Отбор вин', page_icon='🍷', layout='wide')
    st.title('Отбор вин')
    st.write('Оценка качества по лабораторному анализу. Хорошее вино — оценка экспертов от 7 баллов.')

    model = load_model()
    data = load_test()
    config = json.loads((APP_DIR / 'config.json').read_text(encoding='utf-8'))
    features = list(model.feature_names_in_)
    X_test = data[features]
    y_test = data['target']
    probabilities = test_proba(model, X_test)

    st.sidebar.header('Правило отбора')
    threshold = st.sidebar.slider('Минимальная вероятность', 0.0, 1.0,
                                  value=float(config['threshold']), step=0.005,
                                  format='%.3f', key='threshold')
    st.sidebar.caption(f"Порог из валидации: {config['threshold']:.3f}")
    st.sidebar.write('Выше порог — строже отбор. Ниже порог — больше вин в подборке.')

    single_tab, metrics_tab, batch_tab, weights_tab = st.tabs(
        ['Одно вино', 'Модельные метрики', 'Загрузка CSV', 'Что выучила модель'])

    with single_tab:
        st.subheader('Лабораторные показатели')
        columns = st.columns(3)
        inputs = {}
        numeric_features = [name for name in features if name != 'is_red']
        for index, name in enumerate(numeric_features):
            with columns[index % 3]:
                inputs[name] = st.number_input(
                    LABELS[name], min_value=float(X_test[name].min()),
                    max_value=float(X_test[name].max()), value=float(X_test[name].median()),
                    step=(0.00001 if name == 'density' else
                          0.001 if name == 'chlorides' else
                          1.0 if 'sulfur_dioxide' in name else 0.01),
                    format='%.5f' if name == 'density' else '%.3f', key=name)
        color = st.radio('Тип вина', ['Белое', 'Красное'], horizontal=True,
                          index=int(X_test['is_red'].median()))
        inputs['is_red'] = int(color == 'Красное')
        row = pd.DataFrame([inputs], columns=features)
        probability = wine_proba(model, row)
        st.metric('Вероятность хорошего вина', f'{probability:.1%}')
        if probability >= threshold:
            st.success(f'Включить в подборку: вероятность не ниже порога {threshold:.3f}.')
        else:
            st.info(f'Пропустить: вероятность ниже порога {threshold:.3f}.')
        st.caption('Диапазоны и начальные значения взяты из тестовой таблицы.')

    with metrics_tab:
        st.subheader('Как меняется подборка')
        predicted = probabilities >= threshold
        precision = metrics.precision_score(y_test, predicted, zero_division=0)
        recall = metrics.recall_score(y_test, predicted, zero_division=0)
        f1 = metrics.f1_score(y_test, predicted, zero_division=0)
        columns = st.columns(4)
        for column, label, value in zip(columns, ['Precision', 'Recall', 'F1', 'Доля отобранных'],
                                         [precision, recall, f1, predicted.mean()]):
            column.metric(label, f'{value:.1%}')
        st.caption(f'Тестовая выборка: {len(data)} вин. Порог: {threshold:.3f}. '
                   f'ROC-AUC: {metrics.roc_auc_score(y_test, probabilities):.3f}; '
                   f'PR-AUC (AP): {metrics.average_precision_score(y_test, probabilities):.3f}.')

        cm = metrics.confusion_matrix(y_test, predicted, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 3.5))
        metrics.ConfusionMatrixDisplay(cm, display_labels=['Обычное', 'Хорошее']).plot(
            ax=ax, cmap='RdPu', colorbar=False)
        ax.set_title('Матрица ошибок')
        st.pyplot(fig)
        plt.close(fig)

        fpr, tpr, _ = metrics.roc_curve(y_test, probabilities)
        pr, re, _ = metrics.precision_recall_curve(y_test, probabilities)
        tn, fp, fn, tp = cm.ravel()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(fpr, tpr, color=WINE_COLOR)
        axes[0].plot([0, 1], [0, 1], '--', color='gray')
        axes[0].scatter(fp / (fp + tn), recall, color=WINE_COLOR, edgecolor='black',
                        s=70, zorder=3, label=f'Порог {threshold:.3f}')
        axes[0].set(xlabel='Доля обычных вин, попавших в отбор',
                    ylabel='Доля найденных хороших вин', title='ROC-кривая')
        axes[1].plot(re, pr, color=WINE_COLOR)
        axes[1].axhline(y_test.mean(), ls='--', color='gray', label='Случайный отбор')
        axes[1].scatter(recall, precision, color=WINE_COLOR, edgecolor='black',
                        s=70, zorder=3, label=f'Порог {threshold:.3f}')
        axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision–recall')
        for ax in axes:
            ax.legend(fontsize=8)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.write('Повышая порог, мы отбираем меньше вин и пропускаем больше хороших. '
                 'Снижая порог, находим больше хороших вин, но чаще добавляем обычные. '
                 'Точность подборки обычно растёт с порогом, хотя на конечной выборке возможны колебания.')
        if not predicted.any():
            st.info('При этом пороге не выбрано ни одного вина. Precision принят равным нулю.')
        st.caption('Этот экран показывает поведение модели на test. Рабочий порог выбран по валидации в ноутбуке.')

    with batch_tab:
        st.subheader('Оценка списка вин')
        st.write('Загрузите CSV с лабораторными признаками. is_red: 0 — белое, 1 — красное. '
                 f'Для всех строк применяется порог {threshold:.3f}.')
        st.download_button('Скачать пример CSV',
                           X_test.head(10).to_csv(index=False).encode('utf-8'),
                           file_name='wine_example.csv', mime='text/csv')
        with st.expander('Нужные столбцы'):
            st.code(', '.join(features), language=None)
        uploaded = st.file_uploader('CSV с признаками вин', type=['csv'])
        if uploaded is not None:
            try:
                uploaded_data = pd.read_csv(uploaded)
                result = batch_prediction(model, uploaded_data, features, threshold)
            except (ValueError, TypeError, pd.errors.ParserError, UnicodeDecodeError) as error:
                st.error(f'Не получилось обработать файл. {error}')
            else:
                st.dataframe(result, hide_index=True)
                st.write(f"В подборку попало {int(result['recommend'].sum())} из {len(result)} вин.")
                st.download_button('Скачать прогнозы', result.to_csv(index=False).encode('utf-8-sig'),
                                   file_name='wine_predictions.csv', mime='text/csv')

    with weights_tab:
        st.subheader('Вклад признаков')
        coefficients = pd.Series(model.named_steps['logisticregression'].coef_[0],
                                  index=features).sort_values()
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = [WINE_COLOR if weight > 0 else '#587B8A' for weight in coefficients]
        ax.barh([LABELS[name] for name in coefficients.index], coefficients, color=colors)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set(xlabel='Коэффициент после StandardScaler')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        positive = ', '.join(LABELS[name] for name in coefficients[coefficients > 0].tail(3).index[::-1])
        negative = ', '.join(LABELS[name] for name in coefficients[coefficients < 0].head(3).index)
        st.write(f'Сильнее всего за хорошее вино голосуют: {positive}.')
        st.write(f'Сильнее всего против: {negative}.')
        st.caption('Коэффициент показывает изменение логарифма шансов при росте признака '
                   'на одно стандартное отклонение, если остальные признаки зафиксированы. '
                   'Это связь в данных, а не доказательство причинного влияния.')


if __name__ == '__main__':
    main()
