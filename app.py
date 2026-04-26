from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Зависимости между логико-лингвистическими переменными",
    layout="wide",
)


# ============================================================
# СПРАВОЧНИКИ И НАСТРОЙКИ
# ============================================================

LINGUISTIC_LEVELS: Dict[int, str] = {
    1: "Не зависят",
    2: "Скорее слабо зависят",
    3: "Слабо зависят",
    4: "Зависят",
    5: "Скорее сильно зависят",
    6: "Сильно зависят",
    7: "Очень сильно зависят",
}

X_VARIABLES: Dict[str, str] = {
    "X1": "Скорость выработки ленты стекла",
    "X2": "Скорость вращения роликов утоняющих устройств",
    "X3": "Температура расплава металла",
    "X4": "Температура кожуха дна",
    "X5": "Температура бортов ванны расплава",
    "X6": "Стабильность положения ленты в ванне расплава",
    "X7": "Расход электроэнергии",
    "X8": "Расход природного газа",
    "X9": "Расход защитной атмосферы",
    "X10": "Расход металла",
    "X11": "Уровень квалификации оператора",
    "X12": "Размер и состав дежурной бригады",
    "X13": "Сложность выбранного режима производства",
}

Z_VARIABLES: Dict[str, str] = {
    "Z1": "Частота отключения электроэнергии",
    "Z2": "Отключение природного газа",
    "Z3": "Прекращение подачи защитной атмосферы или её составляющих",
    "Z4": "Параметры внешней среды",
    "Z5": "Износ и отказ оборудования",
}

OUTPUT_VARIABLES: Dict[str, str] = {
    "Y1": "Качество производимого стекла",
    "Y2": "Количество бракованного стекла",
}


@dataclass(frozen=True)
class Dependency:
    code: str
    source_code: str
    source_name: str
    target_code: str
    target_name: str
    description: str
    default_points: List[Tuple[int, int]]


DEPENDENCIES: Dict[str, Dependency] = {
    "R1": Dependency(
        code="R1",
        source_code="X13",
        source_name=X_VARIABLES["X13"],
        target_code="X2",
        target_name=X_VARIABLES["X2"],
        description=(
            "Экспертная зависимость между сложностью выбранного режима производства "
            "и скоростью вращения роликов утоняющих устройств."
        ),
        default_points=[(1, 2), (2, 3), (4, 4), (6, 6), (7, 7)],
    ),
    "R2": Dependency(
        code="R2",
        source_code="X13",
        source_name=X_VARIABLES["X13"],
        target_code="X3",
        target_name=X_VARIABLES["X3"],
        description=(
            "Экспертная зависимость между сложностью выбранного режима производства "
            "и температурой расплава металла."
        ),
        default_points=[(1, 2), (2, 3), (4, 5), (6, 6), (7, 7)],
    ),
}


# ============================================================
# ФУНКЦИИ
# ============================================================

def level_label(level: int) -> str:
    return f"{level} — {LINGUISTIC_LEVELS[level]}"


def build_scale_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Уровень": list(LINGUISTIC_LEVELS.keys()),
            "Лингвистическое значение": list(LINGUISTIC_LEVELS.values()),
        }
    )


def build_points_df(points: List[Tuple[int, int]], dependency: Dependency) -> pd.DataFrame:
    rows = []
    for index, (x, y) in enumerate(points, start=1):
        rows.append(
            {
                "Точка": index,
                f"{dependency.source_code}: {dependency.source_name}": x,
                "Значение X": LINGUISTIC_LEVELS[x],
                f"{dependency.target_code}: {dependency.target_name}": y,
                "Значение Y": LINGUISTIC_LEVELS[y],
            }
        )
    return pd.DataFrame(rows)


def fit_curve(points: List[Tuple[int, int]], degree: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)

    safe_degree = min(degree, len(points) - 1)
    coeffs = np.polyfit(x, y, safe_degree)

    x_dense = np.linspace(1, 7, 250)
    y_dense = np.polyval(coeffs, x_dense)
    y_dense = np.clip(y_dense, 1, 7)

    return x_dense, y_dense, coeffs


def polynomial_to_string(coeffs: np.ndarray) -> str:
    degree = len(coeffs) - 1
    parts = []

    for i, coef in enumerate(coeffs):
        power = degree - i
        if abs(coef) < 1e-10:
            continue

        sign = "+" if coef >= 0 else "-"
        value = abs(coef)

        if power == 0:
            part = f"{value:.4f}"
        elif power == 1:
            part = f"{value:.4f}x"
        else:
            part = f"{value:.4f}x^{power}"

        parts.append((sign, part))

    if not parts:
        return "0"

    result = ""
    first_sign, first_part = parts[0]
    result += first_part if first_sign == "+" else f"-{first_part}"

    for sign, part in parts[1:]:
        result += f" {sign} {part}"

    return result


def calculate_by_curve(x_value: float, points: List[Tuple[int, int]], degree: int) -> float:
    _, _, coeffs = fit_curve(points, degree)
    y_value = float(np.polyval(coeffs, x_value))
    return float(np.clip(y_value, 1, 7))


def nearest_linguistic_value(value: float) -> Tuple[int, str]:
    nearest = int(np.clip(round(value), 1, 7))
    return nearest, LINGUISTIC_LEVELS[nearest]


def make_dependency_chart(
    dependency: Dependency,
    points: List[Tuple[int, int]],
    degree: int,
    x_test: float | None = None,
    y_test: float | None = None,
) -> Tuple[go.Figure, str]:
    x_dense, y_dense, coeffs = fit_curve(points, degree)

    x_points = [p[0] for p in points]
    y_points = [p[1] for p in points]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_dense,
            y=y_dense,
            mode="lines",
            name="Аппроксимирующая кривая",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_points,
            y=y_points,
            mode="markers+text",
            text=[f"P{i}" for i in range(1, len(points) + 1)],
            textposition="top center",
            marker={"size": 12},
            name="Экспертные точки",
        )
    )

    if x_test is not None and y_test is not None:
        fig.add_trace(
            go.Scatter(
                x=[x_test],
                y=[y_test],
                mode="markers+text",
                text=["Расчёт"],
                textposition="bottom center",
                marker={"size": 14, "symbol": "diamond"},
                name="Расчётная точка",
            )
        )

    tick_values = list(LINGUISTIC_LEVELS.keys())
    tick_text = [LINGUISTIC_LEVELS[i] for i in tick_values]

    fig.update_layout(
        title=f"{dependency.code}: {dependency.source_code} → {dependency.target_code}",
        xaxis_title=f"{dependency.source_code}: {dependency.source_name}",
        yaxis_title=f"{dependency.target_code}: {dependency.target_name}",
        height=650,
        xaxis={"tickmode": "array", "tickvals": tick_values, "ticktext": tick_text, "range": [0.8, 7.2]},
        yaxis={"tickmode": "array", "tickvals": tick_values, "ticktext": tick_text, "range": [0.8, 7.2]},
        margin={"l": 40, "r": 40, "t": 80, "b": 80},
    )

    formula = polynomial_to_string(coeffs)
    return fig, formula


def dependency_calculator(dependency: Dependency, key_prefix: str) -> None:
    st.subheader(f"{dependency.code} — формирование зависимости")
    st.markdown(dependency.description)

    st.markdown(
        f"""
**Входная логико-лингвистическая переменная:**  
`{dependency.source_code}` — {dependency.source_name}

**Выходная логико-лингвистическая переменная:**  
`{dependency.target_code}` — {dependency.target_name}
        """
    )

    st.markdown("### Экспертные точки")
    st.caption("Необходимо задать 5 точек зависимости в прямоугольной системе координат.")

    degree = st.slider(
        "Степень аппроксимирующего полинома",
        min_value=1,
        max_value=4,
        value=2,
        step=1,
        key=f"{key_prefix}_degree",
    )

    cols = st.columns(5)
    points: List[Tuple[int, int]] = []

    for i, col in enumerate(cols, start=1):
        default_x, default_y = dependency.default_points[i - 1]
        with col:
            st.markdown(f"**Точка {i}**")
            x = st.selectbox(
                "X",
                options=list(LINGUISTIC_LEVELS.keys()),
                index=default_x - 1,
                format_func=level_label,
                key=f"{key_prefix}_x_{i}",
            )
            y = st.selectbox(
                "Y",
                options=list(LINGUISTIC_LEVELS.keys()),
                index=default_y - 1,
                format_func=level_label,
                key=f"{key_prefix}_y_{i}",
            )
            points.append((int(x), int(y)))

    points = sorted(points, key=lambda p: p[0])
    points_df = build_points_df(points, dependency)

    st.dataframe(points_df, use_container_width=True)

    st.markdown("### Расчёт значения по построенной зависимости")
    selected_x = st.slider(
        f"Значение {dependency.source_code}",
        min_value=1.0,
        max_value=7.0,
        value=4.0,
        step=0.1,
        key=f"{key_prefix}_calc_x",
    )

    calculated_y = calculate_by_curve(selected_x, points, degree)
    nearest_level, nearest_label = nearest_linguistic_value(calculated_y)

    col1, col2, col3 = st.columns(3)
    col1.metric("Числовое значение Y", f"{calculated_y:.3f}")
    col2.metric("Ближайший уровень", nearest_level)
    col3.metric("Лингвистическая интерпретация", nearest_label)

    fig, formula = make_dependency_chart(
        dependency=dependency,
        points=points,
        degree=degree,
        x_test=selected_x,
        y_test=calculated_y,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Аппроксимирующая зависимость")
    st.code(f"y = {formula}", language="text")


# ============================================================
# ИНТЕРФЕЙС
# ============================================================

st.title("🧊 Формирование зависимостей между логико-лингвистическими переменными")
st.markdown(
    "Веб-приложение предназначено для экспертного задания зависимостей между переменными модели технологического процесса производства листового стекла."
)

st.info(
    "Для выбранной зависимости экспертно задаются 5 точек в прямоугольной системе координат. "
    "На основе этих точек строится аппроксимирующая кривая, отражающая характер связи между двумя "
    "логико-лингвистическими переменными."
)


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Постановка задачи",
        "Калькулятор R1",
        "Калькулятор R2",
        "Переменные и шкала",
    ]
)

with tab1:
    st.subheader("Постановка задачи")
    st.markdown(
        """
В модели рассматриваются логико-лингвистические переменные технологического процесса производства листового стекла.
Для выбранных пар переменных экспертно задаются 5 точек зависимости. После этого через точки проводится кривая,
отражающая характер влияния одной переменной на другую.
        """
    )

    st.markdown("### Выбранные зависимости")
    selected_df = pd.DataFrame(
        [
            {
                "Код": dep.code,
                "Входная переменная": f"{dep.source_code} — {dep.source_name}",
                "Выходная переменная": f"{dep.target_code} — {dep.target_name}",
                "Описание": dep.description,
            }
            for dep in DEPENDENCIES.values()
        ]
    )
    st.dataframe(selected_df, use_container_width=True)

with tab2:
    dependency_calculator(DEPENDENCIES["R1"], "r1")

with tab3:
    dependency_calculator(DEPENDENCIES["R2"], "r2")

with tab4:
    st.subheader("Шкала зависимости")
    st.dataframe(build_scale_df(), use_container_width=True)

    st.subheader("Входные координаты модели")
    st.dataframe(
        pd.DataFrame({"Код": list(X_VARIABLES.keys()), "Переменная": list(X_VARIABLES.values())}),
        use_container_width=True,
    )

    st.subheader("Возмущения")
    st.dataframe(
        pd.DataFrame({"Код": list(Z_VARIABLES.keys()), "Переменная": list(Z_VARIABLES.values())}),
        use_container_width=True,
    )

    st.subheader("Выходные координаты")
    st.dataframe(
        pd.DataFrame({"Код": list(OUTPUT_VARIABLES.keys()), "Переменная": list(OUTPUT_VARIABLES.values())}),
        use_container_width=True,
    )
