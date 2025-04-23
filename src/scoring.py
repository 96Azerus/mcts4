# src/scoring.py v1.7
"""
Логика подсчета очков, роялти, проверки фолов и условий Фантазии
для OFC Pineapple согласно предоставленным правилам.
Реализует Progressive Fantasyland (14-17 карт).
"""
from typing import List, Tuple, Dict, Optional
from collections import Counter
import traceback
import sys
import logging

# Импорты из src пакета
from src.card import (
    Card as CardUtils, card_to_str, RANK_MAP, STR_RANKS,
    INVALID_CARD, INT_RANK_TO_CHAR
)
try:
    from src.evaluator.ofc_3card_evaluator import evaluate_3_card_ofc
    from src.evaluator.ofc_5card_evaluator import Evaluator as Evaluator5Card
    from src.evaluator.ofc_5card_lookup import LookupTable as LookupTable5Card
except ImportError as e:
     print(f"CRITICAL ERROR: Failed to import custom evaluators in scoring.py: {e}", file=sys.stderr)
     traceback.print_exc(file=sys.stderr)
     raise ImportError("Failed to import evaluators, cannot proceed.") from e

# Получаем логгер
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): logger.setLevel(logging.WARNING)

# --- Создаем экземпляр 5-карточного эвалуатора ---
try:
    evaluator_5card = Evaluator5Card()
except Exception as e_init:
    logger.critical(f"Failed to initialize 5-card evaluator: {e_init}", exc_info=True)
    raise RuntimeError("Could not initialize 5-card evaluator.") from e_init

# --- Константы рангов из 5-карточного эвалуатора ---
RANK_CLASS_ROYAL_FLUSH: int = 1
RANK_CLASS_STRAIGHT_FLUSH: int = LookupTable5Card.MAX_STRAIGHT_FLUSH
RANK_CLASS_QUADS: int = LookupTable5Card.MAX_FOUR_OF_A_KIND
RANK_CLASS_FULL_HOUSE: int = LookupTable5Card.MAX_FULL_HOUSE
RANK_CLASS_FLUSH: int = LookupTable5Card.MAX_FLUSH
RANK_CLASS_STRAIGHT: int = LookupTable5Card.MAX_STRAIGHT
RANK_CLASS_TRIPS: int = LookupTable5Card.MAX_THREE_OF_A_KIND
RANK_CLASS_TWO_PAIR: int = LookupTable5Card.MAX_TWO_PAIR
RANK_CLASS_PAIR: int = LookupTable5Card.MAX_PAIR
RANK_CLASS_HIGH_CARD: int = LookupTable5Card.MAX_HIGH_CARD
WORST_RANK: int = RANK_CLASS_HIGH_CARD + 1000

# --- Таблицы Роялти (Американские правила) ---
ROYALTY_BOTTOM_POINTS: Dict[str, int] = {
    "Straight": 2, "Flush": 4, "Full House": 6, "Four of a Kind": 10,
    "Straight Flush": 15,
}
ROYALTY_MIDDLE_POINTS: Dict[str, int] = {
    "Three of a Kind": 2, "Straight": 4, "Flush": 8, "Full House": 12,
    "Four of a Kind": 20, "Straight Flush": 30,
}
ROYALTY_BOTTOM_POINTS_RF: int = 25
ROYALTY_MIDDLE_POINTS_RF: int = 50

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
        int: Ранг руки. Для неполных или невалидных рук возвращает WORST_RANK.
    """
    if not isinstance(cards, list):
        logger.warning(f"get_hand_rank_safe received non-list input: {type(cards)}")
        return WORST_RANK

    valid_cards = [c for c in cards if isinstance(c, int) and c is not None and c != INVALID_CARD and c > 0]
    num_valid = len(valid_cards)
    expected_len = len(cards)

    if expected_len == 3:
        if num_valid != 3: return WORST_RANK
        if len(valid_cards) != len(set(valid_cards)):
             logger.warning(f"Duplicate cards found in 3-card hand for ranking: {[card_to_str(c) for c in valid_cards]}")
             return WORST_RANK
        try:
            rank, _, _ = evaluate_3_card_ofc(valid_cards[0], valid_cards[1], valid_cards[2])
            return rank
        except Exception as e:
            logger.error(f"Error evaluating 3-card hand {[card_to_str(c) for c in valid_cards]}: {e}", exc_info=True)
            return WORST_RANK
    elif expected_len == 5:
        if num_valid != 5: return WORST_RANK
        if len(valid_cards) != len(set(valid_cards)):
             logger.warning(f"Duplicate cards found in 5-card hand for ranking: {[card_to_str(c) for c in valid_cards]}")
             return WORST_RANK
        try:
            rank = evaluator_5card.evaluate(valid_cards)
            return rank
        except Exception as e:
            logger.error(f"Error evaluating 5-card hand {[card_to_str(c) for c in valid_cards]}: {e}", exc_info=True)
            return WORST_RANK
    else:
        logger.warning(f"get_hand_rank_safe called with unsupported hand length {expected_len}.")
        return WORST_RANK

def get_row_royalty(cards: List[Optional[int]], row_name: str) -> int:
    """
    Вычисляет роялти для указанного ряда (top, middle, bottom) согласно правилам.
    Не начисляет роялти для неполных или невалидных рядов.

    Args:
        cards (List[Optional[int]]): Карты в ряду.
        row_name (str): Название ряда ('top', 'middle', 'bottom').

    Returns:
        int: Количество очков роялти.
    """
    if not isinstance(cards, list): return 0

    valid_cards = [c for c in cards if isinstance(c, int) and c is not None and c != INVALID_CARD and c > 0]
    num_cards = len(valid_cards)
    royalty = 0

    if row_name == "top":
        if num_cards != 3: return 0
        if len(valid_cards) != len(set(valid_cards)): return 0
        try:
            # --- ПЕРЕПИСАНО: Логика роялти для топа v1.7 ---
            rank_3card, type_str, rank_str = evaluate_3_card_ofc(valid_cards[0], valid_cards[1], valid_cards[2])
            if type_str == 'Trips':
                rank_char = rank_str[0] # 'AAA' -> 'A'
                rank_index = RANK_MAP.get(rank_char)
                royalty = ROYALTY_TOP_TRIPS.get(rank_index, 0)
            elif type_str == 'Pair':
                pair_rank_char = rank_str[0] # '66A' -> '6'
                rank_index = RANK_MAP.get(pair_rank_char)
                royalty = ROYALTY_TOP_PAIRS.get(rank_index, 0)
            return royalty
        except Exception as e:
            logger.error(f"Error calculating top row royalty for {[card_to_str(c) for c in valid_cards]}: {e}", exc_info=True)
            return 0

    elif row_name in ["middle", "bottom"]:
        if num_cards != 5: return 0
        if len(valid_cards) != len(set(valid_cards)): return 0
        try:
            rank_eval = get_hand_rank_safe(valid_cards)
            if rank_eval >= WORST_RANK: return 0

            is_royal = (rank_eval == RANK_CLASS_ROYAL_FLUSH)
            rank_class = evaluator_5card.get_rank_class(rank_eval)
            hand_name = evaluator_5card.class_to_string(rank_class)

            table = ROYALTY_MIDDLE_POINTS if row_name == "middle" else ROYALTY_BOTTOM_POINTS

            if is_royal:
                royalty = ROYALTY_MIDDLE_POINTS_RF if row_name == "middle" else ROYALTY_BOTTOM_POINTS_RF
            else:
                royalty = table.get(hand_name, 0)

            if row_name == "bottom" and hand_name == "Three of a Kind":
                 royalty = 0

            return royalty
        except Exception as e:
            logger.error(f"Error calculating {row_name} row royalty for {[card_to_str(c) for c in valid_cards]}: {e}", exc_info=True)
            return 0
    else:
        logger.warning(f"Unknown row name '{row_name}' in get_row_royalty.")
        return 0

def check_board_foul(top: List[Optional[int]], middle: List[Optional[int]], bottom: List[Optional[int]]) -> bool:
    """
    Проверяет, является ли доска "мертвой" (фол) из-за нарушения порядка силы линий.
    Считает фолом только если все ряды полностью заполнены валидными картами без дубликатов.

    Args:
        top (List[Optional[int]]): Карты верхнего ряда.
        middle (List[Optional[int]]): Карты среднего ряда.
        bottom (List[Optional[int]]): Карты нижнего ряда.

    Returns:
        bool: True, если доска "мертвая", иначе False.
    """
    # --- ПЕРЕПИСАНО: v1.7 ---
    try:
        # 1. Проверка полноты и валидности карт в каждом ряду
        valid_top = [c for c in top if isinstance(c, int) and c is not None and c != INVALID_CARD and c > 0]
        valid_middle = [c for c in middle if isinstance(c, int) and c is not None and c != INVALID_CARD and c > 0]
        valid_bottom = [c for c in bottom if isinstance(c, int) and c is not None and c != INVALID_CARD and c > 0]

        if len(valid_top) != 3 or len(valid_middle) != 5 or len(valid_bottom) != 5:
            return False # Не фол, если доска не полная

        # 2. Проверка на дубликаты *между* рядами
        all_cards = valid_top + valid_middle + valid_bottom
        if len(all_cards) != len(set(all_cards)):
             logger.warning(f"Duplicate cards detected across rows in check_board_foul. Hand is invalid, returning False (not foul).")
             return False

        # 3. Получаем ранги
        rank_t = get_hand_rank_safe(valid_top)
        rank_m = get_hand_rank_safe(valid_middle)
        rank_b = get_hand_rank_safe(valid_bottom)

        # 4. Проверяем, что ранги валидны
        if rank_t >= WORST_RANK or rank_m >= WORST_RANK or rank_b >= WORST_RANK:
             logger.warning(f"Invalid rank detected during foul check (T:{rank_t}, M:{rank_m}, B:{rank_b}). Returning False.")
             return False

        # 5. Проверяем условие фола: rank_b <= rank_m <= rank_t (меньше = лучше)
        is_foul = not (rank_b <= rank_m <= rank_t)
        # Добавим логирование для отладки падающих тестов
        if is_foul:
             logger.debug(f"Foul detected: T={rank_t}, M={rank_m}, B={rank_b}")
        # else:
        #      logger.debug(f"No Foul: T={rank_t}, M={rank_m}, B={rank_b}")
        return is_foul
    except Exception as e:
        logger.error(f"Error during check_board_foul: {e}", exc_info=True)
        return False # Считаем не фолом при любой ошибке

def get_fantasyland_entry_cards(top: List[Optional[int]]) -> int:
    """
    Проверяет верхний ряд на соответствие условиям входа в Progressive Fantasyland.

    Args:
        top (List[Optional[int]]): Карты верхнего ряда.

    Returns:
        int: Количество карт для раздачи в Фантазии (14, 15, 16, 17) или 0, если условие не выполнено.
    """
    # --- ПЕРЕПИСАНО: v1.7 ---
    valid_cards = [c for c in top if isinstance(c, int) and c is not None and c != INVALID_CARD and c > 0]
    if len(valid_cards) != 3: return 0
    if len(valid_cards) != len(set(valid_cards)): return 0

    try:
        # Используем исправленный ofc_3card_lookup (v1.2)
        _, type_str, rank_str = evaluate_3_card_ofc(valid_cards[0], valid_cards[1], valid_cards[2])

        if type_str == 'Trips':
            return 17
        elif type_str == 'Pair':
            pair_rank_char = rank_str[0] # 'QQJ' -> 'Q'
            if pair_rank_char == 'A': return 16
            if pair_rank_char == 'K': return 15
            if pair_rank_char == 'Q': return 14
            return 0 # Пары JJ и ниже
        else: # High Card
            return 0
    except Exception as e:
        logger.error(f"Error checking Fantasyland entry for {[card_to_str(c) for c in valid_cards]}: {e}", exc_info=True)
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
    # --- ПЕРЕПИСАНО: v1.7 ---
    try:
        valid_top = [c for c in top if isinstance(c, int) and c is not None and c != INVALID_CARD and c > 0]
        valid_middle = [c for c in middle if isinstance(c, int) and c is not None and c != INVALID_CARD and c > 0]
        valid_bottom = [c for c in bottom if isinstance(c, int) and c is not None and c != INVALID_CARD and c > 0]

        # Проверяем полноту, валидность, отсутствие дубликатов и фола
        if len(valid_top) != 3 or len(valid_middle) != 5 or len(valid_bottom) != 5: return False
        all_cards = valid_top + valid_middle + valid_bottom
        if len(all_cards) != len(set(all_cards)): return False
        # Используем исправленный check_board_foul
        if check_board_foul(valid_top, valid_middle, valid_bottom): return False

        # Проверяем условие для верхнего ряда (Трипс)
        # Используем исправленный ofc_3card_lookup (v1.2)
        _, type_str_top, _ = evaluate_3_card_ofc(valid_top[0], valid_top[1], valid_top[2])
        if type_str_top == 'Trips':
            return True

        # Проверяем условие для нижнего ряда (Каре или лучше)
        # Используем исправленный ofc_5card_lookup (v1.3)
        rank_b = get_hand_rank_safe(valid_bottom)
        # Каре или Стрит-флеш (ранги <= RANK_CLASS_QUADS)
        if rank_b < WORST_RANK and rank_b <= RANK_CLASS_QUADS:
            return True

    except Exception as e:
        logger.error(f"Error checking Fantasyland stay: {e}", exc_info=True)
        return False

    return False

# Импортируем PlayerBoard только для аннотации типов
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
    # --- ПЕРЕПИСАНО: v1.7 ---
    if not board1.is_complete() or not board2.is_complete():
         logger.warning("calculate_headsup_score called with incomplete boards.")
         return 0

    # Проверяем фолы и получаем роялти
    r1 = board1.get_total_royalty()
    r2 = board2.get_total_royalty()
    foul1 = board1.is_foul
    foul2 = board2.is_foul

    # Обработка сценариев с фолами
    if foul1 and foul2:
        return 0
    if foul1:
        # P1 фол, P2 не фол. P1 платит P2 6 очков + роялти P2.
        return -(6 + r2)
    if foul2:
        # P2 фол, P1 не фол. P2 платит P1 6 очков + роялти P1.
        return 6 + r1

    # Если фолов нет, сравниваем линии
    line_score_p1 = 0 # Очки P1 за линии

    try:
        rank_t1 = board1._get_rank('top'); rank_t2 = board2._get_rank('top')
        rank_m1 = board1._get_rank('middle'); rank_m2 = board2._get_rank('middle')
        rank_b1 = board1._get_rank('bottom'); rank_b2 = board2._get_rank('bottom')

        if any(r >= WORST_RANK for r in [rank_t1, rank_m1, rank_b1, rank_t2, rank_m2, rank_b2]):
             logger.error("Invalid rank detected during score calculation. Returning 0.")
             return 0

        # Считаем очки за каждую линию
        if rank_t1 < rank_t2: line_score_p1 += 1
        elif rank_t2 < rank_t1: line_score_p1 -= 1

        if rank_m1 < rank_m2: line_score_p1 += 1
        elif rank_m2 < rank_m1: line_score_p1 -= 1

        if rank_b1 < rank_b2: line_score_p1 += 1
        elif rank_b2 < rank_b1: line_score_p1 -= 1

    except Exception as e_rank:
         logger.error(f"Error getting ranks during score calculation: {e_rank}", exc_info=True)
         return 0

    # Добавляем бонус за скуп
    scoop_bonus = 0
    if line_score_p1 == 3: # P1 scoop
        scoop_bonus = 3
    elif line_score_p1 == -3: # P2 scoop
        scoop_bonus = -3

    # Итоговый счет = очки за линии + бонус за скуп + разница роялти
    total_score_diff = line_score_p1 + scoop_bonus + (r1 - r2)

    return total_score_diff
