import streamlit as st
import numpy as np
import pandas as pd

# Определение иерархии факторов (МАИ, 2 уровня)
# Уровень 1: 6 главных критериев → 15 сравнений
# Уровень 2: 23 субкритерия → 37 сравнений (итого 52)
HIERARCHY = {
    "Культурные":             ["ОКН", "События", "Учреждения", "Ремесла"],                                  # 6 сравнений
    "Инфраструктурные":       ["Транспорт", "Размещение", "Питание", "Соц. услуги"],                        # 6 сравнений
    "Биологические":          ["ООПТ", "Леса", "Красные виды", "Животный мир"],                             # 6 сравнений
    "Гидрометеорологические": ["Плотность озер", "Плотность рек", "Температурный режим", "Снежный покров", "Осадки", "Ветер"],  # 15 сравнений
    "Геологические":          ["Достопримечательности", "Рельеф"],                                          # 1 сравнение
    "Ограничивающие":         ["Заболоченность", "Заболевания", "Опасные процессы"],                        # 3 сравнения
}

MAIN_CRITERIA = list(HIERARCHY.keys())

# Словесные аналоги шкалы Саати
SCORE_LABELS = {
    2: "Слабое превосходство",
    3: "Умеренное превосходство",
    4: "Умеренное+ превосходство",
    5: "Сильное превосходство",
    6: "Сильное+ превосходство",
    7: "Очень сильное превосходство",
    8: "Очень-очень сильное превосходство",
    9: "Абсолютное превосходство",
}


def saaty_consistency_check(matrix):
    n = matrix.shape[0]

    # Для матриц 1x1 и 2x2 индекс согласованности всегда равен 0
    if n <= 2:
        if n == 1:
            return True, 0.0, np.array([1.0]), None
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        weights = np.abs(np.real(eigenvectors[:, np.argmax(np.real(eigenvalues))]))
        weights = weights / np.sum(weights)
        return True, 0.0, weights, None

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_eigenval = np.max(np.real(eigenvalues))

    weights = np.abs(np.real(eigenvectors[:, np.argmax(np.real(eigenvalues))]))
    weights = weights / np.sum(weights)

    ci = (max_eigenval - n) / (n - 1)

    # Табличные значения случайного индекса (RI) по Саати
    ri_dict = {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ri = ri_dict.get(n, 1.49)

    cr = ci / ri
    is_consistent = cr <= 0.1

    problem_pair = None
    if not is_consistent:
        ideal_matrix = np.outer(weights, 1 / weights)
        error_matrix = matrix * (1 / ideal_matrix)
        error_matrix[np.tril_indices(n)] = 0
        i_idx, j_idx = np.unravel_index(np.argmax(error_matrix), error_matrix.shape)
        problem_pair = (i_idx, j_idx)

    return is_consistent, cr, weights, problem_pair


def build_comparison_ui(factors, prefix):
    n = len(factors)
    matrix = np.ones((n, n))

    if n == 1:
        st.info(f"В группе '{factors[0]}' только один фактор. Сравнение не требуется.")
        return matrix

    for i in range(n):
        for j in range(i + 1, n):
            st.markdown(f"**{factors[i]}** vs **{factors[j]}**")
            col1, col2 = st.columns([1, 2])

            with col1:
                choice = st.radio(
                    "Какой фактор важнее?",
                    [factors[i], factors[j], "Равны"],
                    key=f"{prefix}_radio_{i}_{j}",
                    horizontal=False
                )

            with col2:
                score = st.slider(
                    "Степень превосходства (2-9)",
                    min_value=2, max_value=9, value=3,
                    key=f"{prefix}_score_{i}_{j}",
                    disabled=(choice == "Равны")
                )
                if choice != "Равны":
                    st.caption(f"**{score}** — {SCORE_LABELS[score]}")
                else:
                    st.caption("Факторы равнозначны")

            val = 1.0
            if choice == factors[i]:
                val = float(score)
            elif choice == factors[j]:
                val = 1.0 / float(score)

            matrix[i, j] = val
            matrix[j, i] = 1.0 / val
            st.write("---")

    return matrix


def analyze_matrix(matrix, factors, group_name):
    is_consistent, cr, weights, problem_pair = saaty_consistency_check(matrix)

    if is_consistent:
        st.success(f"[{group_name}] Матрица согласована (ОС = {cr:.3f}).")
    else:
        st.error(f"[{group_name}] Матрица противоречива (ОС = {cr:.3f} > 0.10).")
        if problem_pair is not None:
            i, j = problem_pair
            st.warning(f"Рекомендуется пересмотреть оценку в паре: **{factors[i]}** и **{factors[j]}**.")

    return is_consistent, weights


st.set_page_config(page_title="Оценка факторов (МАИ)", layout="wide")
st.title("Многоуровневая оценка по методу Саати")

# Сбор матриц
matrices = {}

with st.expander("Шаг 1. Сравнение главных критериев (Уровень 1)", expanded=True):
    st.write("Оцените приоритет основных групп факторов.")
    matrices["main"] = build_comparison_ui(MAIN_CRITERIA, "main")

st.markdown("### Шаг 2. Сравнение подфакторов (Уровень 2)")
tabs = st.tabs(MAIN_CRITERIA)

for idx, main_crit in enumerate(MAIN_CRITERIA):
    with tabs[idx]:
        sub_factors = HIERARCHY[main_crit]
        matrices[main_crit] = build_comparison_ui(sub_factors, f"sub_{idx}")

st.markdown("### Шаг 3. Анализ и расчет весов")

if st.button("Проверить согласованность и рассчитать результат", type="primary"):
    all_consistent = True
    results = {}

    # Анализ главных критериев
    st.subheader("Проверка уровня 1")
    is_consist, main_weights = analyze_matrix(matrices["main"], MAIN_CRITERIA, "Главные критерии")
    if not is_consist:
        all_consistent = False

    # Анализ подфакторов
    st.subheader("Проверка уровня 2")
    sub_weights = {}
    for main_crit in MAIN_CRITERIA:
        sub_factors = HIERARCHY[main_crit]
        is_consist, weights = analyze_matrix(matrices[main_crit], sub_factors, main_crit)
        sub_weights[main_crit] = weights
        if not is_consist:
            all_consistent = False

    # Итоговый расчет, если нет противоречий
    if all_consistent:
        st.success("Все матрицы согласованы. Вычисление глобальных весов...")

        global_results = []
        for i, main_crit in enumerate(MAIN_CRITERIA):
            w_main = main_weights[i]
            for j, sub_factor in enumerate(HIERARCHY[main_crit]):
                w_sub = sub_weights[main_crit][j]
                global_weight = w_main * w_sub
                global_results.append({
                    "Группа": main_crit,
                    "Фактор": sub_factor,
                    "Локальный вес": round(w_sub, 4),
                    "Глобальный вес": round(global_weight, 4)
                })

        df_results = pd.DataFrame(global_results)
        df_results = df_results.sort_values(by="Глобальный вес", ascending=False).reset_index(drop=True)

        st.subheader("Итоговый рейтинг 23 факторов")
        st.dataframe(df_results, use_container_width=True)

        # Проверка суммы глобальных весов (должна быть равна 1.0)
        total_weight = df_results["Глобальный вес"].sum()
        st.write(f"Сумма глобальных весов: **{total_weight:.4f}**")
    else:
        st.error(
            "Расчет глобальных весов остановлен. Пожалуйста, устраните противоречия, указанные выше, и повторите расчет.")