"""Обучение робота поиску пути на клеточном поле с помощью Q-learning.

Зависимости: numpy, matplotlib
Запуск: python q_learning_robot.py
Скрипт обучает агента, печатает итоговые показатели и сохраняет графики
в папку results рядом с файлом программы.
"""
from __future__ import annotations

from pathlib import Path
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ------------------------- Описание среды -------------------------
ROWS = 8
COLS = 8
START = (0, 0)   # верхний левый угол: (строка, столбец)
GOAL = (7, 7)    # нижний правый угол

# Клетки, на которые робот не может перейти.
OBSTACLES = {
    (0, 3), (1, 3), (2, 3),
    (3, 1), (3, 2), (3, 3), (3, 5),
    (4, 5), (5, 1), (5, 2), (5, 5), (6, 2),
}

# Действия: вверх, вправо, вниз, влево.
ACTIONS = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}
ACTION_NAMES = {0: "вверх", 1: "вправо", 2: "вниз", 3: "влево"}

# Вознаграждения и параметры Q-learning.
STEP_REWARD = -1.0
OBSTACLE_REWARD = -10.0
WALL_REWARD = -5.0
GOAL_REWARD = 100.0
ALPHA = 0.15       # скорость обучения
GAMMA = 0.95       # коэффициент дисконтирования
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.997
EPISODES = 3500
MAX_STEPS = 100
RANDOM_SEED = 42

State = Tuple[int, int]


def valid_cell(state: State) -> bool:
    """Проверяет, находится ли клетка внутри поля и не является ли препятствием."""
    r, c = state
    return 0 <= r < ROWS and 0 <= c < COLS and state not in OBSTACLES


def step(state: State, action: int) -> Tuple[State, float, bool]:
    """Выполняет действие и возвращает (новое состояние, награда, конец эпизода)."""
    dr, dc = ACTIONS[action]
    candidate = (state[0] + dr, state[1] + dc)

    # Столкновение с границей поля: робот остаётся на месте.
    if not (0 <= candidate[0] < ROWS and 0 <= candidate[1] < COLS):
        return state, WALL_REWARD, False

    # Препятствие: робот остаётся на месте и получает сильный штраф.
    if candidate in OBSTACLES:
        return state, OBSTACLE_REWARD, False

    # Цель: эпизод завершается с большой положительной наградой.
    if candidate == GOAL:
        return candidate, GOAL_REWARD, True

    # Обычный переход: небольшой штраф побуждает искать короткий путь.
    return candidate, STEP_REWARD, False


def choose_action(q_table: np.ndarray, state: State, epsilon: float) -> int:
    """Стратегия epsilon-greedy: исследование случайных ходов или лучший ход."""
    if random.random() < epsilon:
        return random.randrange(len(ACTIONS))
    state_index = state[0] * COLS + state[1]
    best_actions = np.flatnonzero(q_table[state_index] == q_table[state_index].max())
    return int(random.choice(best_actions.tolist()))


def greedy_path(q_table: np.ndarray) -> Tuple[List[State], bool]:
    """Строит маршрут по максимальным Q-значениям; обнаружение цикла останавливает проверку."""
    state = START
    path = [state]
    visited = {state}

    for _ in range(ROWS * COLS * 3):
        state_index = state[0] * COLS + state[1]
        # После обучения выбирается действие с наибольшей Q-оценкой.
        action = int(np.argmax(q_table[state_index]))
        next_state, _reward, done = step(state, action)
        if next_state == state:
            return path, False
        path.append(next_state)
        state = next_state
        if done or state == GOAL:
            return path, True
        if state in visited:
            return path, False
        visited.add(state)
    return path, False


def moving_average(values: List[float], window: int = 100) -> np.ndarray:
    """Вычисляет скользящее среднее с указанным окном."""
    values_array = np.asarray(values, dtype=float)
    if len(values_array) < window:
        return values_array
    return np.convolve(values_array, np.ones(window) / window, mode="valid")


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Q-таблица: 64 состояния (по одной строке на клетку) x 4 действия.
    q_table = np.zeros((ROWS * COLS, len(ACTIONS)), dtype=float)
    episode_rewards: List[float] = []
    episode_steps: List[int] = []
    episode_successes: List[int] = []
    epsilon = EPSILON_START

    # ----------------------------- Обучение -----------------------------
    for _episode in range(EPISODES):
        state = START
        total_reward = 0.0
        success = False

        for step_number in range(1, MAX_STEPS + 1):
            action = choose_action(q_table, state, epsilon)
            next_state, reward, done = step(state, action)

            state_index = state[0] * COLS + state[1]
            next_index = next_state[0] * COLS + next_state[1]

            # Обновление Q(s,a): текущая награда + оценка лучшего будущего действия.
            if done:
                target = reward
            else:
                target = reward + GAMMA * float(np.max(q_table[next_index]))
            q_table[state_index, action] += ALPHA * (
                target - q_table[state_index, action]
            )

            state = next_state
            total_reward += reward
            if done:
                success = True
                break

        episode_rewards.append(total_reward)
        episode_steps.append(step_number)
        episode_successes.append(1 if success else 0)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    # ----------------------- Оценка обученного агента -----------------------
    path, path_success = greedy_path(q_table)
    final_100_success = float(np.mean(episode_successes[-100:])) * 100
    final_100_reward = float(np.mean(episode_rewards[-100:]))
    final_100_steps = float(np.mean(episode_steps[-100:]))

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # График скользящей средней награды за 100 эпизодов.
    rewards_avg = moving_average(episode_rewards, 100)
    plt.figure(figsize=(10, 5.5))
    x_values = np.arange(100, EPISODES + 1) if len(rewards_avg) == EPISODES - 99 else np.arange(1, len(rewards_avg) + 1)
    plt.plot(x_values, rewards_avg, linewidth=1.8, label="Средняя награда за 100 эпизодов")
    plt.xlabel("Эпизод обучения")
    plt.ylabel("Средняя суммарная награда")
    plt.title("Обучение робота с помощью Q-learning")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curve.png", dpi=180)
    plt.close()

    # Дополнительный график успешности за 100 эпизодов.
    success_avg = moving_average(episode_successes, 100) * 100
    plt.figure(figsize=(10, 5.5))
    x_values = np.arange(100, EPISODES + 1) if len(success_avg) == EPISODES - 99 else np.arange(1, len(success_avg) + 1)
    plt.plot(x_values, success_avg, linewidth=1.8, label="Доля успешных эпизодов за 100 эпизодов")
    plt.xlabel("Эпизод обучения")
    plt.ylabel("Успешные эпизоды, %")
    plt.title("Изменение успешности обучения")
    plt.ylim(-3, 103)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "success_rate.png", dpi=180)
    plt.close()

    # Визуализация найденного пути.
    grid = np.zeros((ROWS, COLS), dtype=int)
    for r, c in OBSTACLES:
        grid[r, c] = 1
    plt.figure(figsize=(7.5, 7.5))
    plt.imshow(grid, cmap="Greys", vmin=0, vmax=1)
    plt.xticks(range(COLS))
    plt.yticks(range(ROWS))
    plt.grid(which="major", color="lightgray", linewidth=1.2)
    plt.gca().set_xticks(np.arange(-0.5, COLS, 1), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, ROWS, 1), minor=True)
    plt.grid(which="minor", color="white", linewidth=1.2)
    plt.tick_params(which="minor", bottom=False, left=False)

    if path:
        xs = [cell[1] for cell in path]
        ys = [cell[0] for cell in path]
        plt.plot(xs, ys, marker="o", linewidth=2.8, markersize=5, label="Путь агента")
    plt.text(START[1], START[0], "S", ha="center", va="center", fontsize=12, fontweight="bold")
    plt.text(GOAL[1], GOAL[0], "G", ha="center", va="center", fontsize=12, fontweight="bold")
    for r, c in OBSTACLES:
        plt.text(c, r, "×", ha="center", va="center", fontsize=12, fontweight="bold")
    plt.title("Маршрут робота после обучения\nS - старт, G - цель, × - препятствие")
    plt.xlabel("Столбец")
    plt.ylabel("Строка")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    plt.savefig(output_dir / "learned_path.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Текстовый файл с метриками эксперимента.
    report_lines = [
        "Результаты эксперимента Q-learning",
        f"Эпизодов обучения: {EPISODES}",
        f"Скорость обучения alpha: {ALPHA}",
        f"Коэффициент дисконтирования gamma: {GAMMA}",
        f"Минимальное epsilon: {EPSILON_MIN}",
        f"Успешных эпизодов среди последних 100: {final_100_success:.1f}%",
        f"Средняя награда за последние 100 эпизодов: {final_100_reward:.2f}",
        f"Среднее число шагов за последние 100 эпизодов: {final_100_steps:.2f}",
        f"Найден ли путь жадной стратегией: {'да' if path_success else 'нет'}",
        f"Число шагов в найденном пути: {len(path) - 1 if path_success else 'не найден'}",
        "Координаты пути (строка, столбец): " + " -> ".join(map(str, path)),
    ]
    (output_dir / "metrics.txt").write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))
    print(f"\nГрафики и результаты сохранены в: {output_dir}")


if __name__ == "__main__":
    main()
