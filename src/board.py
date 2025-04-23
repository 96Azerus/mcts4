# src/board.py v1.3
"""
Представление доски одного игрока в OFC Pineapple.
Содержит карты в трех рядах и управляет их размещением,
а также кэширует и вычисляет ранги/роялти, делегируя логику в scoring.py.
"""
from typing import List, Tuple, Dict, Optional
from collections import Counter
import copy
import traceback
import sys
import logging

# Импорты из src пакета
from src.card import Card, card_to_str, RANK_MAP, STR_RANKS, INVALID_CARD, CARD_PLACEHOLDER, SUIT_CHAR_TO_INT
# Импортируем ВСЕ необходимые функции и константы из scoring
from src.scoring import (
    get_hand_rank_safe, check_board_foul,
    get_fantasyland_entry_cards, check_fantasyland_stay,
    get_row_royalty, RANK_CLASS_HIGH_CARD, WORST_RANK # Импортируем WORST_RANK
)

# Получаем логгер
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): logger.setLevel(logging.WARNING)


class PlayerBoard:
    """
    Представляет доску одного игрока с тремя рядами (линиями).
    """
    ROW_CAPACITY: Dict[str, int] = {'top': 3, 'middle': 5, 'bottom': 5}
    ROW_NAMES: List[str] = ['top', 'middle', 'bottom']
    WORST_POSSIBLE_RANK: int = WORST_RANK

    def __init__(self):
        """Инициализирует пустую доску."""
        self.rows: Dict[str, List[Optional[int]]] = {
            name: [None] * capacity for name, capacity in self.ROW_CAPACITY.items()
        }
        self._cards_placed: int = 0
        self.is_foul: bool = False
        self._cached_ranks: Dict[str, Optional[int]] = {name: None for name in self.ROW_NAMES}
        self._cached_royalties: Dict[str, Optional[int]] = {name: None for name in self.ROW_NAMES}
        self._is_complete: bool = False

    def add_card(self, card_int: int, row_name: str, index: int) -> bool:
        """
        Добавляет карту (целочисленное представление) в УКАЗАННЫЙ слот ряда.
        Обновляет счетчик карт и сбрасывает кэши.

        Args:
            card_int (int): Целочисленное представление карты.
            row_name (str): Имя ряда ('top', 'middle', 'bottom').
            index (int): Индекс слота в ряду (0-based).

        Returns:
            bool: True при успехе, False при неудаче (неверный ряд/индекс, слот занят, невалидная карта).
        """
        if row_name not in self.ROW_NAMES:
            logger.debug(f"Add card failed: Invalid row name '{row_name}'")
            return False
        if not isinstance(card_int, int) or card_int == INVALID_CARD or card_int <= 0:
             logger.debug(f"Add card failed: Invalid card value '{card_int}'")
             return False

        capacity = self.ROW_CAPACITY[row_name]
        if not (0 <= index < capacity):
            logger.debug(f"Add card failed: Index {index} out of bounds for row '{row_name}' (0-{capacity-1}).")
            return False

        if self.rows[row_name][index] is not None:
            logger.debug(f"Add card failed: Slot {row_name}[{index}] is already occupied by {card_to_str(self.rows[row_name][index])}. Cannot add {card_to_str(card_int)}.")
            return False

        self.rows[row_name][index] = card_int
        self._cards_placed += 1
        self._is_complete = (self._cards_placed == 13) # Обновляем флаг завершенности

        self._reset_caches()
        self.is_foul = False # Сбрасываем флаг фола при изменении
        logger.debug(f"Added card {card_to_str(card_int)} to {row_name}[{index}]. Total cards: {self._cards_placed}. Complete: {self._is_complete}")
        return True

    def remove_card(self, row_name: str, index: int) -> Optional[int]:
        """
        Удаляет карту из указанного слота (например, для отмены хода в UI).
        Обновляет счетчик карт и сбрасывает кэши.

        Args:
            row_name (str): Имя ряда.
            index (int): Индекс слота.

        Returns:
            Optional[int]: Удаленная карта (int) или None, если слот был пуст или некорректен.
        """
        if row_name not in self.ROW_NAMES or not (0 <= index < self.ROW_CAPACITY[row_name]):
            return None

        card_int = self.rows[row_name][index]
        if card_int is not None:
            self.rows[row_name][index] = None
            self._cards_placed -= 1
            self._is_complete = False # Доска больше не полная
            self._reset_caches()
            self.is_foul = False # Сбрасываем флаг фола
            logger.debug(f"Removed card {card_to_str(card_int)} from {row_name}[{index}]. Total cards: {self._cards_placed}.")
            return card_int
        return None

    def set_full_board(self, top: List[int], middle: List[int], bottom: List[int]):
        """
        Устанавливает всю доску из готовых списков карт (например, для Фантазии).
        Проверяет корректность входных данных и уникальность карт.

        Args:
            top (List[int]): Список из 3 карт для верхнего ряда.
            middle (List[int]): Список из 5 карт для среднего ряда.
            bottom (List[int]): Список из 5 карт для нижнего ряда.

        Raises:
            ValueError: Если количество карт неверно, карты невалидны или есть дубликаты.
            TypeError: Если входные данные не являются списками или содержат не int.
        """
        if not isinstance(top, list) or not isinstance(middle, list) or not isinstance(bottom, list):
             raise TypeError("Input rows must be lists.")
        if len(top) != 3 or len(middle) != 5 or len(bottom) != 5:
            raise ValueError(f"Incorrect number of cards provided for setting full board (top={len(top)}, mid={len(middle)}, bot={len(bottom)}).")

        all_cards: List[int] = []
        card_lists = {'top': top, 'middle': middle, 'bottom': bottom}

        for row_name, card_list in card_lists.items():
             for i, card_int in enumerate(card_list):
                  if not isinstance(card_int, int) or card_int == INVALID_CARD or card_int <= 0:
                       raise ValueError(f"Invalid card value '{card_int}' provided in row '{row_name}' at index {i}.")
                  all_cards.append(card_int)

        if len(all_cards) != len(set(all_cards)):
            counts = Counter(all_cards)
            duplicates = {card_to_str(card): count for card, count in counts.items() if count > 1}
            raise ValueError(f"Duplicate cards provided for setting full board: {duplicates}")

        self.rows['top'] = list(top)
        self.rows['middle'] = list(middle)
        self.rows['bottom'] = list(bottom)

        self._cards_placed = 13
        self._is_complete = True
        self._reset_caches()
        self.check_and_set_foul() # Проверяем фол сразу

    def get_row_cards(self, row_name: str) -> List[int]:
        """Возвращает список действительных карт (int) в указанном ряду."""
        if row_name not in self.rows:
            return []
        return [card for card in self.rows[row_name] if isinstance(card, int) and card is not None and card != INVALID_CARD and card > 0]

    def is_row_full(self, row_name: str) -> bool:
        """Проверяет, заполнен ли указанный ряд валидными картами."""
        if row_name not in self.rows:
            return False
        return all(isinstance(slot, int) and slot is not None and slot != INVALID_CARD and slot > 0 for slot in self.rows[row_name])

    def get_available_slots(self) -> List[Tuple[str, int]]:
        """Возвращает список доступных (пустых) слотов в формате ('row_name', index)."""
        slots = []
        for row_name in self.ROW_NAMES:
            for i, card in enumerate(self.rows[row_name]):
                if card is None:
                    slots.append((row_name, i))
        return slots

    def get_total_cards(self) -> int:
        """Возвращает общее количество карт, размещенных на доске."""
        return self._cards_placed

    def is_complete(self) -> bool:
        """Проверяет, размещены ли все 13 карт на доске."""
        self._is_complete = (self._cards_placed == 13)
        return self._is_complete

    def _reset_caches(self):
        """Сбрасывает внутренние кэши рангов и роялти."""
        self._cached_ranks = {name: None for name in self.ROW_NAMES}
        self._cached_royalties = {name: None for name in self.ROW_NAMES}

    def _get_rank(self, row_name: str) -> int:
        """
        Получает ранг руки для указанного ряда (из кэша или вычисляет).
        Использует get_hand_rank_safe из scoring.py.

        Args:
            row_name (str): Имя ряда.

        Returns:
            int: Ранг руки (меньше = лучше). Возвращает WORST_POSSIBLE_RANK при ошибке или для неполного ряда.
        """
        if row_name not in self.ROW_NAMES:
            return self.WORST_POSSIBLE_RANK

        if self._cached_ranks[row_name] is None:
            cards_with_none = self.rows[row_name]
            self._cached_ranks[row_name] = get_hand_rank_safe(cards_with_none)

        rank = self._cached_ranks[row_name]
        return rank if rank is not None else self.WORST_POSSIBLE_RANK

    def check_and_set_foul(self) -> bool:
        """
        Проверяет доску на фол (нарушение порядка линий) и устанавливает флаг is_foul.
        Вызывать эту функцию имеет смысл только для полной доски.
        Использует check_board_foul из scoring.py.

        Returns:
            bool: Текущее значение флага is_foul.
        """
        if not self.is_complete():
            if self.is_foul:
                 self.is_foul = False
                 self._reset_caches()
            return False

        current_foul_status = check_board_foul(
            self.rows['top'],
            self.rows['middle'],
            self.rows['bottom']
        )

        if current_foul_status != self.is_foul:
            self.is_foul = current_foul_status
            if self.is_foul:
                self._cached_royalties = {'top': 0, 'middle': 0, 'bottom': 0}
            else:
                 self._reset_caches()

        return self.is_foul

    def get_royalties(self) -> Dict[str, int]:
        """
        Считает и возвращает роялти для каждой линии, используя кэш.
        Автоматически проверяет фол для полной доски перед расчетом.
        Использует get_row_royalty из scoring.py.

        Returns:
            Dict[str, int]: Словарь с роялти для 'top', 'middle', 'bottom'.
                            Возвращает нули, если рука "мертвая" (фол).
        """
        if self.is_complete() and self.check_and_set_foul():
            return {'top': 0, 'middle': 0, 'bottom': 0}

        for row_name in self.ROW_NAMES:
            if self._cached_royalties[row_name] is None:
                cards_with_none = self.rows[row_name]
                self._cached_royalties[row_name] = get_row_royalty(cards_with_none, row_name)

        return {
            row: self._cached_royalties.get(row, 0) or 0
            for row in self.ROW_NAMES
        }


    def get_total_royalty(self) -> int:
        """Возвращает сумму роялти по всем линиям, учитывая фол."""
        royalties_dict = self.get_royalties()
        return sum(royalties_dict.values())

    def get_fantasyland_qualification_cards(self) -> int:
        """
        Проверяет, квалифицируется ли доска для входа в Fantasyland (Progressive).
        Возвращает количество карт для ФЛ (0, 14, 15, 16, 17).
        Автоматически проверяет фол.
        Использует get_fantasyland_entry_cards из scoring.py.

        Returns:
            int: Количество карт для Фантазии (0, если нет квалификации или фол).
        """
        if not self.is_complete(): return 0
        if self.check_and_set_foul(): return 0

        return get_fantasyland_entry_cards(self.rows['top'])

    def check_fantasyland_stay_conditions(self) -> bool:
        """
        Проверяет, выполнены ли условия для удержания Fantasyland (Re-Fantasy).
        (Сет на топе или Каре+ на боттоме, без фола).
        Автоматически проверяет фол.
        Использует check_fantasyland_stay из scoring.py.

        Returns:
            bool: True, если условия выполнены, иначе False.
        """
        if not self.is_complete(): return False
        if self.check_and_set_foul(): return False

        return check_fantasyland_stay(
            self.rows['top'],
            self.rows['middle'],
            self.rows['bottom']
        )

    def get_board_state_tuple(self) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
        """
        Возвращает неизменяемое представление доски для использования в качестве ключа словаря или элемента set (например, в MCTS).
        Карты представлены строками (или CARD_PLACEHOLDER).
        Карты внутри каждого ряда сортируются для каноничности представления (от старшей к младшей).

        Returns:
            Tuple[Tuple[str, ...], ...]: Кортеж из трех кортежей (top, middle, bottom),
                                         содержащих отсортированные строки карт.
        """
        # --- ИСПРАВЛЕНО: Ключ сортировки (s > h > d > c) ---
        suit_order = {'s': 0, 'h': 1, 'd': 2, 'c': 3} # Меньше = старше
        def sort_key(card_str: str) -> Tuple[int, int]:
            if card_str == CARD_PLACEHOLDER:
                return (99, 99) # Плейсхолдеры идут в конец
            try:
                rank_char = card_str[0].upper()
                suit_char = card_str[1].lower()
                # Отрицательный ранг для сортировки по убыванию
                rank_val = -RANK_MAP.get(rank_char, -99)
                # Порядок мастей
                suit_val = suit_order.get(suit_char, 99)
                return (rank_val, suit_val)
            except IndexError:
                return (99, 99) # Некорректная строка

        rows_as_str_tuples: Dict[str, Tuple[str, ...]] = {}
        for r_name in self.ROW_NAMES:
            row_str_list: List[str] = [card_to_str(c) for c in self.rows[r_name]]
            sorted_row = tuple(sorted(row_str_list, key=sort_key))
            rows_as_str_tuples[r_name] = sorted_row

        return (rows_as_str_tuples['top'], rows_as_str_tuples['middle'], rows_as_str_tuples['bottom'])

    def copy(self) -> 'PlayerBoard':
        """Создает глубокую копию объекта доски."""
        new_board = PlayerBoard()
        new_board.rows = {r: list(cards) for r, cards in self.rows.items()}
        new_board._cards_placed = self._cards_placed
        new_board.is_foul = self.is_foul
        new_board._is_complete = self._is_complete
        new_board._cached_ranks = self._cached_ranks.copy()
        new_board._cached_royalties = self._cached_royalties.copy()
        return new_board

    def __str__(self) -> str:
        """Возвращает строковое представление доски для вывода."""
        s = ""
        max_len = max(len(self.rows[r_name]) for r_name in self.ROW_NAMES)
        for r_name in self.ROW_NAMES:
            row_str = [card_to_str(c) for c in self.rows[r_name]]
            row_str += [CARD_PLACEHOLDER] * (max_len - len(row_str))
            s += f"{r_name.upper():<6}: " + " ".join(f"{c:^2}" for c in row_str) + "\n"

        s += f"Cards: {self._cards_placed}/13"
        is_comp = self.is_complete()
        current_foul_status = self.is_foul
        if is_comp:
            current_foul_status = self.check_and_set_foul()
            s += f", Complete: Yes, Foul: {current_foul_status}"
            royalties_dict = self.get_royalties()
            total_royalty = sum(royalties_dict.values())
            s += f", Royalties: {total_royalty} {royalties_dict}"
        else:
            s += f", Complete: No"
            s += f", Foul: {current_foul_status}"
        return s.strip()

    def __repr__(self) -> str:
        """Возвращает репрезентацию объекта PlayerBoard."""
        is_comp = self.is_complete()
        return f"PlayerBoard(Cards={self._cards_placed}, Complete={is_comp}, Foul={self.is_foul})"
