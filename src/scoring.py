# src/scoring.py v1.1
"""
Логика подсчета очков, роялти, проверки фолов и условий Фантазии
для OFC Pineapple согласно предоставленным правилам.
Реализует Progressive Fantasyland (14-17 карт).
"""
from typing import List, Tuple, Dict, Optional
from collections import Counter
import traceback # Для отладки
import sys # Для вывода ошибок

# --- ИСПРАВЛЕНО: Импорты из src пакета ---
from src.card import (
    Card as CardUtils, card_to_str, RANK_MAP, STR_RANKS,
    INVALID_CARD, INT_RANK_TO_CHAR
)
try:
    # Импортируем эвалуаторы из пакета evaluator
    from src.evaluator.ofc_3card_evaluator import evaluate_3_card_ofc
    from src.evaluator.ofc_5card_evaluator import Evaluator as Evaluator5Card
    # Импортируем LookupTable только для доступа к константам MAX_...
    from src.evaluator.ofc_5card_lookup import LookupTable as LookupTable5Card
except ImportError as e:
     # Заглушки на случай проблем с импортом (хотя их быть не должно)
     print(f"CRITICAL ERROR: Failed to import custom evaluators in scoring.py: {e}", file=sys.stderr)
     traceback.print_exc(file=sys.stderr)
     raise ImportError("Failed to import evaluators, cannot proceed.") from e


# --- Создаем экземпляр 5-карточного эвалуатора ---
# Это нормально делать на уровне модуля, т.к. он stateless после инициализации
try:
    evaluator_5card = Evaluator5Card()
except Exception as e_init:
    print(f"CRITICAL ERROR: Failed to initialize 5-card evaluator: {e_init}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    raise RuntimeError("Could not initialize 5-card evaluator.") from e_init


# --- Константы рангов из 5-карточного эвалуатора ---
# (Используем константы из LookupTable5Card для надежности)
RANK_CLASS_ROYAL_FLUSH: int = 1 # Роял Флеш всегда имеет ранг 1
RANK_CLASS_STRAIGHT_FLUSH: int = LookupTable5Card.MAX_STRAIGHT_FLUSH # 10
RANK_CLASS_QUADS: int = LookupTable5Card.MAX_FOUR_OF_A_KIND # 166
RANK_CLASS_FULL_HOUSE: int = LookupTable5Card.MAX_FULL_HOUSE # 322
RANK_CLASS_FLUSH: int = LookupTable5Card.MAX_FLUSH # 1599
RANK_CLASS_STRAIGHT: int = LookupTable5Card.MAX_STRAIGHT # 1609
RANK_CLASS_TRIPS: int = LookupTable5Card.MAX_THREE_OF_A_KIND # 2467
RANK_CLASS_TWO_PAIR: int = LookupTable5Card.MAX_TWO_PAIR # 3325
RANK_CLASS_PAIR: int = LookupTable5Card.MAX_PAIR # 6185
RANK_CLASS_HIGH_CARD: int = LookupTable5Card.MAX_HIGH_CARD # 7462

# --- Таблицы Роялти (Американские правила) ---
# (Используем RANK_MAP из src.card)
ROYALTY_BOTTOM_POINTS: Dict[str, int] = {
    "Straight": 2, "Flush": 4, "Full House": 6, "Four of a Kind": 10,
    "Straight Flush": 15, # RF обрабатывается отдельно
}
ROYALTY_MIDDLE_POINTS: Dict[str, int] = {
    "Three of a Kind": 2, "Straight": 4, "Flush": 8, "Full House": 12,
    "Four of a Kind": 20, "Straight Flush": 30, # RF обрабатывается отдельно
}
ROYALTY_BOTTOM_POINTS_RF: int = 25 # Отдельно для проверки rank == 1
ROYALTY_MIDDLE_POINTS_RF: int = 50 # Отдельно для проверки rank == 1

ROYALTY_TOP_PAIRS: Dict[int, int] = {
    RANK_MAP['6']: 1, RANK_MAP['7']: 2, RANK_MAP['8']: 3, RANK_MAP['9']: 4,
    RANK_MAP['T']: 5, RANK_MAP['J']: 6, RANK_MAP['Q']: 7, RANK_MAP['K']: 8,
    RANK_MAP['A']: 9
}
ROYALTY_TOP_TRIPS: Dict[int, int] = {
    RANK_MAP['2']: 10, RANK_MAP['3']: 11, RANK_MAP['4']: 12, RANK_MAP['5']: 13,
    RANK_MAP['6']: 14, RANK_MAP['7']: 15, RANK_MAP['8']: 16, RANK_MAP['9']: 17,
    RANK_MAP['T']: 18, RANK_MAP['J']: 19, RANK_MAP['Q']: 20, RANK_MAP['K']: 21,
    RANK_MAP['A']: 22
}

# --- Основные функции ---

def get_hand_rank_safe(cards: List[Optional[int]]) -> int:
    """
    Безопасно вычисляет ранг руки (3 или 5 карт), обрабатывая неполные руки.
    Меньший ранг означает более сильную руку.

    Args:
        cards (List[Optional[int]]): Список карт (int) или None.

    Returns:
        int: Ранг руки. Для неполных или невалидных рук возвращает значение хуже худшей возможной руки.
    """
    if not isinstance(cards, list):
        print(f"Warning: get_hand_rank_safe received non-list input: {type(cards)}")
        return RANK_CLASS_HIGH_CARD + 1000 # Очень плохой ранг

    valid_cards = [c for c in cards if isinstance(c, int) and c is not None and c != INVALID_CARD and c > 0]
    num_valid = len(valid_cards)
    expected_len = len(cards)

    # --- ИСПРАВЛЕНО: Возвращаем плохой ранг вместо ValueError ---
    if expected_len == 3:
        if num_valid != 3:
            # Возвращаем ранг хуже худшей 3-карточной руки (455)
            return 455 + 10 + (3 - num_valid)
        try:
            # Проверка на дубликаты перед вызовом эвалуатора
            if len(valid_cards) != len(set(valid_cards)):
                 print(f"Warning: Duplicate cards found in 3-card hand for ranking: {[card_to_str(c) for c in valid_cards]}")
                 return 455 + 50 # Штраф за дубликат
            rank, _, _ = evaluate_3_card_ofc(valid_cards[0], valid_cards[1], valid_cards[2])
            return rank
        except Exception as e:
            print(f"Error evaluating 3-card hand {[card_to_str(c) for c in valid_cards]}: {e}", file=sys.stderr)
            # traceback.print_exc(file=sys.stderr)
            return 455 + 100 # Значительно хуже худшей руки
    elif expected_len == 5:
        if num_valid != 5:
            # Возвращаем ранг хуже худшей 5-карточной руки (7462)
            return RANK_CLASS_HIGH_CARD + 10 + (5 - num_valid)
        try:
            # Проверка на дубликаты перед вызовом эвалуатора
            if len(valid_cards) != len(set(valid_cards)):
                 print(f"Warning: Duplicate cards found in 5-card hand for ranking: {[card_to_str(c) for c in valid_cards]}")
                 return RANK_CLASS_HIGH_CARD + 50 # Штраф за дубликат
            rank = evaluator_5card.evaluate(valid_cards)
            return rank
        except Exception as e:
            print(f"Error evaluating 5-card hand {[card_to_str(c) for c in valid_cards]}: {e}", file=sys.stderr)
            # traceback.print_exc(file=sys.stderr)
            return RANK_CLASS_HIGH_CARD + 100 # Значительно хуже худшей руки
    else:
        # Обработка неожиданной длины руки
        print(f"Warning: get_hand_rank_safe called with unsupported hand length {expected_len}.")
        return RANK_CLASS_HIGH_CARD + 200 # Возвращаем очень плохой ранг

def get_row_royalty(cards: List[Optional[int]], row_name: str) -> int:
    """
    Вычисляет роялти для указанного ряда (top, middle, bottom) согласно правилам.
    Не начисляет роялти для неполных рядов.

    Args:
        cards (List[Optional[int]]): Карты в ряду.
        row_name (str): Название ряда ('top', 'middle', 'bottom').

    Returns:
        int: Количество очков роялти.
    """
    if not isinstance(cards, list): return 0 # Защита от неверного типа

    valid_cards = [c for c in cards if isinstance(c, int) and c is not None and c != INVALID_CARD and c > 0]
    num_cards = len(valid_cards)
    royalty = 0

    if row_name == "top":
        if num_cards != 3: return 0
        try:
            # Проверка на дубликаты
            if len(valid_cards) != len(set(valid_cards)): return 0

            # Оцениваем 3-карточную руку
            _, type_str, rank_str = evaluate_3_card_ofc(valid_cards[0], valid_cards[1], valid_cards[2])

            # Начисляем роялти за трипс
            if type_str == 'Trips':
                rank_char = rank_str[0]
                rank_index = RANK_MAP.get(rank_char)
                if rank_index is not None:
                    royalty = ROYALTY_TOP_TRIPS.get(rank_index, 0)
            # Начисляем роялти за пару (66 и выше)
            elif type_str == 'Pair':
                pair_rank_char = rank_str[0]
                rank_index = RANK_MAP.get(pair_rank_char)
                # --- ИСПРАВЛЕНО: Проверяем наличие индекса в ROYALTY_TOP_PAIRS ---
                if rank_index is not None and rank_index in ROYALTY_TOP_PAIRS:
                    royalty = ROYALTY_TOP_PAIRS.get(rank_index, 0)
            return royalty
        except Exception as e:
            print(f"Error calculating top row royalty for {[card_to_str(c) for c in valid_cards]}: {e}", file=sys.stderr)
            # traceback.print_exc(file=sys.stderr)
            return 0

    elif row_name in ["middle", "bottom"]:
        if num_cards != 5: return 0
        try:
            # Проверка на дубликаты
            if len(valid_cards) != len(set(valid_cards)): return 0

            # Оцениваем 5-карточную руку
            rank_eval = get_hand_rank_safe(valid_cards)
            # Проверяем на Роял-флеш (ранг 1)
            is_royal = (rank_eval == RANK_CLASS_ROYAL_FLUSH)

            # Получаем класс руки (строку)
            rank_class = evaluator_5card.get_rank_class(rank_eval)
            hand_name = evaluator_5card.class_to_string(rank_class) # e.g., "Straight Flush", "Pair"

            # Выбираем таблицу роялти для среднего или нижнего ряда
            table = ROYALTY_MIDDLE_POINTS if row_name == "middle" else ROYALTY_BOTTOM_POINTS

            # Ищем роялти по названию руки (SF, 4K, FH, Fl, St, 3K)
            royalty = table.get(hand_name, 0)

            # Добавляем бонус за Роял Флеш, если он есть в таблице
            if is_royal:
                 if row_name == "middle":
                     # Используем ROYALTY_MIDDLE_POINTS_RF, а не max
                     royalty = ROYALTY_MIDDLE_POINTS_RF
                 elif row_name == "bottom":
                     # Используем ROYALTY_BOTTOM_POINTS_RF, а не max
                     royalty = ROYALTY_BOTTOM_POINTS_RF

            # Корректировка: Трипс на нижнем ряду не дает роялти
            if row_name == "bottom" and hand_name == "Three of a Kind":
                 royalty = 0

            return royalty
        except Exception as e:
            print(f"Error calculating {row_name} row royalty for {[card_to_str(c) for c in valid_cards]}: {e}", file=sys.stderr)
            # traceback.print_exc(file=sys.stderr)
            return 0
    else:
        # Неизвестное имя ряда
        return 0

def check_board_foul(top: List[Optional[int]], middle: List[Optional[int]], bottom: List[Optional[int]]) -> bool:
    """
    Проверяет, является ли доска "мертвой" (фол) из-за нарушения порядка силы линий.
    Считает фолом только если все ряды полностью заполнены валидными картами.

    Args:
        top (List[Optional[int]]): Карты верхнего ряда.
        middle (List[Optional[int]]): Карты среднего ряда.
        bottom (List[Optional[int]]): Карты нижнего ряда.

    Returns:
        bool: True, если доска "мертвая", иначе False.
    """
    # --- ИСПРАВЛЕНО: Проверка на валидность и полноту ---
    valid_top = [c for c in top if isinstance(c, int) and c is not None and c != INVALID_CARD and c > 0]
    valid_middle = [c for c in middle if isinstance(c, int) and c is not None and c != INVALID_CARD and c > 0]
    valid_bottom = [c for c in bottom if isinstance(c, int) and c is not None and c != INVALID_CARD and c > 0]

    if len(valid_top) != 3 or len(valid_middle) != 5 or len(valid_bottom) != 5:
        return False # Не фол, если доска не полная или содержит невалидные карты

    # Проверка на дубликаты между рядами (и внутри рядов, если get_hand_rank_safe их не ловит)
    all_cards = valid_top + valid_middle + valid_bottom
    if len(all_cards) != len(set(all_cards)):
         print(f"Warning: Duplicate cards detected across rows in check_board_foul. Treating as non-foul for now.")
         # Можно решить, считать ли это фолом или ошибкой данных
         return False # Пока считаем не фолом

    # Получаем ранги всех трех рядов
    rank_t = get_hand_rank_safe(valid_top)
    rank_m = get_hand_rank_safe(valid_middle)
    rank_b = get_hand_rank_safe(valid_bottom)

    # Проверяем условие: Нижний <= Средний <= Верхний (по силе, т.е. >= по рангу)
    # Меньший ранг означает более сильную руку
    is_foul = not (rank_b <= rank_m <= rank_t)
    return is_foul

def get_fantasyland_entry_cards(top: List[Optional[int]]) -> int:
    """
    Проверяет верхний ряд на соответствие условиям входа в Progressive Fantasyland.

    Args:
        top (List[Optional[int]]): Карты верхнего ряда.

    Returns:
        int: Количество карт для раздачи в Фантазии (14, 15, 16, 17) или 0, если условие не выполнено.
    """
    valid_cards = [c for c in top if isinstance(c, int) and c is not None and c != INVALID_CARD and c > 0]
    if len(valid_cards) != 3: return 0
    # Проверка на дубликаты
    if len(valid_cards) != len(set(valid_cards)): return 0

    try:
        # Оцениваем 3-карточную руку
        _, type_str, rank_str = evaluate_3_card_ofc(valid_cards[0], valid_cards[1], valid_cards[2])

        # --- ИСПРАВЛЕНО: Логика Progressive Fantasyland ---
        if type_str == 'Trips':
            # Любой трипс на топе дает 17 карт
            return 17
        elif type_str == 'Pair':
            pair_rank_char = rank_str[0] # Ранг пары
            if pair_rank_char == 'Q': return 14
            if pair_rank_char == 'K': return 15
            if pair_rank_char == 'A': return 16
        # Другие комбинации (включая меньшие пары) не дают Фантазию
        return 0
    except Exception as e:
        print(f"Error checking Fantasyland entry for {[card_to_str(c) for c in valid_cards]}: {e}", file=sys.stderr)
        # traceback.print_exc(file=sys.stderr)
        return 0

def check_fantasyland_stay(top: List[Optional[int]], middle: List[Optional[int]], bottom: List[Optional[int]]) -> bool:
    """
    Проверяет условия удержания Фантазии (Re-Fantasy).
    Условие: Сет (Трипс) на верхнем ряду ИЛИ Каре или лучше на нижнем ряду.
    Доска не должна быть "мертвой".

    Args:
        top (List[Optional[int]]): Карты верхнего ряда.
        middle (List[Optional[int]]): Карты среднего ряда.
        bottom (List[Optional[int]]): Карты нижнего ряда.

    Returns:
        bool: True, если условия удержания Фантазии выполнены, иначе False.
    """
    # --- ИСПРАВЛЕНО: Проверка на валидность и полноту ---
    valid_top = [c for c in top if isinstance(c, int) and c is not None and c != INVALID_CARD and c > 0]
    valid_middle = [c for c in middle if isinstance(c, int) and c is not None and c != INVALID_CARD and c > 0]
    valid_bottom = [c for c in bottom if isinstance(c, int) and c is not None and c != INVALID_CARD and c > 0]

    if len(valid_top) != 3 or len(valid_middle) != 5 or len(valid_bottom) != 5:
        return False # Не полная доска

    # Проверка на дубликаты
    all_cards = valid_top + valid_middle + valid_bottom
    if len(all_cards) != len(set(all_cards)):
         return False # Дубликаты не дают остаться

    # Проверяем на фол
    if check_board_foul(valid_top, valid_middle, valid_bottom):
        return False # Фол

    # --- ИСПРАВЛЕНО: Логика проверки условий ---
    # Проверяем условие для верхнего ряда (Трипс)
    try:
        _, type_str_top, _ = evaluate_3_card_ofc(valid_top[0], valid_top[1], valid_top[2])
        if type_str_top == 'Trips':
            return True # Трипс наверху -> остаемся в ФЛ
    except Exception as e:
        print(f"Error checking top for Fantasyland stay {[card_to_str(c) for c in valid_top]}: {e}", file=sys.stderr)
        # traceback.print_exc(file=sys.stderr)
        # Продолжаем проверку нижнего ряда, если верхний вызвал ошибку

    # Проверяем условие для нижнего ряда (Каре или лучше)
    try:
        rank_b = get_hand_rank_safe(valid_bottom)
        # Сравниваем ранг с максимальным рангом Фулл-хауса
        # Если ранг <= RANK_CLASS_QUADS (166), значит это Каре или Стрит-флеш
        if rank_b <= RANK_CLASS_QUADS:
            return True # Каре или лучше внизу -> остаемся в ФЛ
    except Exception as e:
        print(f"Error checking bottom for Fantasyland stay {[card_to_str(c) for c in valid_bottom]}: {e}", file=sys.stderr)
        # traceback.print_exc(file=sys.stderr)

    # Если ни одно из условий не выполнено
    return False

# Импортируем PlayerBoard только для аннотации типов, чтобы избежать циклического импорта
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.board import PlayerBoard

def calculate_headsup_score(board1: 'PlayerBoard', board2: 'PlayerBoard') -> int:
    """
    Рассчитывает итоговый счет за раунд между двумя игроками (1 на 1).
    Использует метод подсчета 1-6 (1 очко за линию, 3 за скуп).
    Учитывает фолы и роялти.

    Args:
        board1 (PlayerBoard): Доска игрока 1 (считается P0).
        board2 (PlayerBoard): Доска игрока 2 (считается P1).

    Returns:
        int: Разница очков (с точки зрения игрока 1).
             Положительное значение - игрок 1 выиграл очки у игрока 2.
             Отрицательное значение - игрок 2 выиграл очки у игрока 1.
    """
    # --- ИСПРАВЛЕНО: Логика подсчета очков с фолами ---
    # Убедимся, что доски полные перед подсчетом
    if not board1.is_complete() or not board2.is_complete():
         print("Warning: calculate_headsup_score called with incomplete boards.")
         # Можно вернуть 0 или выбросить ошибку, пока возвращаем 0
         return 0

    # Проверяем фолы (check_and_set_foul вызывается внутри get_royalties)
    r1 = board1.get_total_royalty() # Вызов обновит board1.is_foul
    r2 = board2.get_total_royalty() # Вызов обновит board2.is_foul
    foul1 = board1.is_foul
    foul2 = board2.is_foul

    # Обработка сценариев с фолами
    if foul1 and foul2:
        return 0 # Оба фол -> 0 очков за линии и 0 роялти
    if foul1:
        # Игрок 1 фол, Игрок 2 не фол. Игрок 2 получает 6 очков за линии + свои роялти.
        # Игрок 1 не получает роялти.
        return -(6 + r2) # Отрицательное значение, так как P2 выиграл у P1
    if foul2:
        # Игрок 2 фол, Игрок 1 не фол. Игрок 1 получает 6 очков за линии + свои роялти.
        # Игрок 2 не получает роялти.
        return 6 + r1

    # Если фолов нет, сравниваем линии
    line_score_diff = 0 # Очки только за сравнение линий
    wins1 = 0 # Счетчик выигранных линий для игрока 1

    # Сравнение рангов (меньший ранг лучше)
    rank_t1 = board1._get_rank('top')
    rank_m1 = board1._get_rank('middle')
    rank_b1 = board1._get_rank('bottom')
    rank_t2 = board2._get_rank('top')
    rank_m2 = board2._get_rank('middle')
    rank_b2 = board2._get_rank('bottom')

    if rank_t1 < rank_t2: wins1 += 1
    elif rank_t2 < rank_t1: wins1 -= 1

    if rank_m1 < rank_m2: wins1 += 1
    elif rank_m2 < rank_m1: wins1 -= 1

    if rank_b1 < rank_b2: wins1 += 1
    elif rank_b2 < rank_b1: wins1 -= 1

    # Начисляем очки за линии
    line_score_diff += wins1

    # Начисляем бонус за скуп (scoop)
    if wins1 == 3: # Игрок 1 выиграл все 3 линии
        line_score_diff += 3
    elif wins1 == -3: # Игрок 2 выиграл все 3 линии
        line_score_diff -= 3

    # Итоговый счет = очки за линии + разница роялти
    total_score_diff = line_score_diff + (r1 - r2)

    return total_score_diff
