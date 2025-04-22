# src/game_state.py v1.2
"""
Определяет класс GameState, управляющий полным состоянием игры
OFC Pineapple для двух игроков, включая Progressive Fantasyland
и инкапсулированную логику управления ходом игры.
"""
import copy
import random
import sys
import traceback
import logging # Добавлено для логирования
import os
from itertools import combinations, permutations
from typing import List, Tuple, Optional, Set, Dict, Any

# Импорты из src пакета
from src.card import Card, card_to_str, card_from_str, INVALID_CARD, CARD_PLACEHOLDER
from src.deck import Deck
from src.board import PlayerBoard
# --- ИСПРАВЛЕНО: Импортируем все нужные функции из scoring ---
from src.scoring import (
    calculate_headsup_score, check_board_foul,
    get_fantasyland_entry_cards, check_fantasyland_stay
)


# Получаем логгер Flask приложения (если доступен) или создаем свой
# Это позволит логам GameState идти в тот же поток, что и логи Flask
try:
    # Попытка получить логгер Flask, если он уже настроен
    logger = logging.getLogger('flask.app')
    if not logger.hasHandlers(): # Если логгер Flask не настроен, используем базовый
        raise ImportError("Flask logger not configured")
except (ImportError, NameError):
    # Базовая настройка логгера, если Flask недоступен или не настроен
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers(): # Настраиваем только если нет обработчиков
        log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
        handler = logging.StreamHandler(sys.stderr) # Вывод в stderr по умолчанию
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(log_level)
        # logger.propagate = False # Опционально: предотвратить дублирование, если корневой логгер настроен

class GameState:
    """
    Представляет полное состояние игры OFC Pineapple для двух игроков.
    Инкапсулирует основную логику перехода состояний.
    """
    NUM_PLAYERS = 2

    def __init__(self,
                 boards: Optional[List[PlayerBoard]] = None,
                 deck: Optional[Deck] = None,
                 private_discard: Optional[List[List[int]]] = None,
                 dealer_idx: int = 0,
                 current_player_idx: Optional[int] = None, # Управляется внутренне
                 street: int = 0,
                 current_hands: Optional[Dict[int, Optional[List[int]]]] = None,
                 fantasyland_status: Optional[List[bool]] = None,
                 next_fantasyland_status: Optional[List[bool]] = None,
                 fantasyland_cards_to_deal: Optional[List[int]] = None,
                 is_fantasyland_round: Optional[bool] = None,
                 fantasyland_hands: Optional[List[Optional[List[int]]]] = None,
                 _player_acted_this_street: Optional[List[bool]] = None,
                 _player_finished_round: Optional[List[bool]] = None,
                 _last_player_acted: Optional[int] = None): # Добавили _last_player_acted
        """Инициализирует состояние игры."""
        self.boards: List[PlayerBoard] = boards if boards is not None else [PlayerBoard() for _ in range(self.NUM_PLAYERS)]
        self.deck: Deck = deck if deck is not None else Deck()
        self.private_discard: List[List[int]] = private_discard if private_discard is not None else [[] for _ in range(self.NUM_PLAYERS)]

        if not (0 <= dealer_idx < self.NUM_PLAYERS):
            logger.error(f"Invalid dealer index provided: {dealer_idx}. Defaulting to 0.")
            dealer_idx = 0
        self.dealer_idx: int = dealer_idx

        # Внутренний индекс игрока, чья очередь ходить
        # Если не задан, определяется как игрок слева от дилера
        self._internal_current_player_idx = (dealer_idx + 1) % self.NUM_PLAYERS if current_player_idx is None else current_player_idx

        self.street: int = street

        self.current_hands: Dict[int, Optional[List[int]]] = current_hands if current_hands is not None else {i: None for i in range(self.NUM_PLAYERS)}
        self.fantasyland_hands: List[Optional[List[int]]] = fantasyland_hands if fantasyland_hands is not None else [None] * self.NUM_PLAYERS

        self.fantasyland_status: List[bool] = fantasyland_status if fantasyland_status is not None else [False] * self.NUM_PLAYERS
        # --- ИСПРАВЛЕНО: next_fantasyland_status по умолчанию копирует fantasyland_status ---
        self.next_fantasyland_status: List[bool] = list(self.fantasyland_status) if next_fantasyland_status is None else list(next_fantasyland_status)
        self.fantasyland_cards_to_deal: List[int] = fantasyland_cards_to_deal if fantasyland_cards_to_deal is not None else [0] * self.NUM_PLAYERS
        self.is_fantasyland_round: bool = any(self.fantasyland_status) if is_fantasyland_round is None else is_fantasyland_round

        self._player_acted_this_street: List[bool] = _player_acted_this_street if _player_acted_this_street is not None else [False] * self.NUM_PLAYERS
        # --- ИСПРАВЛЕНО: _player_finished_round инициализируется на основе is_complete досок ---
        self._player_finished_round: List[bool] = _player_finished_round if _player_finished_round is not None else [b.is_complete() for b in self.boards]

        # Индекс игрока, который ходил последним
        self._last_player_acted: Optional[int] = _last_player_acted


    # --- Публичные методы для получения информации ---

    def get_player_board(self, player_idx: int) -> PlayerBoard:
        """Возвращает доску указанного игрока."""
        if 0 <= player_idx < self.NUM_PLAYERS:
            return self.boards[player_idx]
        else:
            logger.error(f"Invalid player index {player_idx} requested in get_player_board.")
            raise IndexError(f"Invalid player index: {player_idx}")

    def get_player_hand(self, player_idx: int) -> Optional[List[int]]:
        """Возвращает текущую руку игрока (обычную или ФЛ)."""
        if not (0 <= player_idx < self.NUM_PLAYERS):
            logger.error(f"Invalid player index {player_idx} requested in get_player_hand.")
            return None

        if self.is_fantasyland_round and self.fantasyland_status[player_idx]:
            return self.fantasyland_hands[player_idx]
        else:
            return self.current_hands.get(player_idx)

    def is_round_over(self) -> bool:
        """Проверяет, завершили ли все игроки раунд."""
        return all(self._player_finished_round)

    def get_terminal_score(self) -> int:
        """Возвращает счет раунда (P0 vs P1), если он завершен."""
        if not self.is_round_over():
            logger.warning("get_terminal_score called before round is over.")
            return 0
        try:
            # Убедимся, что флаги фола актуальны перед подсчетом
            # Вызов get_total_royalty внутри calculate_headsup_score обновит флаги
            return calculate_headsup_score(self.boards[0], self.boards[1])
        except Exception as e:
            logger.error(f"Error calculating terminal score: {e}", exc_info=True)
            return 0 # Возвращаем 0 при ошибке

    def get_known_dead_cards(self, perspective_player_idx: int) -> Set[int]:
        """Возвращает набор карт, известных игроку как вышедшие из игры."""
        if not (0 <= perspective_player_idx < self.NUM_PLAYERS):
            logger.error(f"Invalid player index {perspective_player_idx} requested in get_known_dead_cards.")
            return set()

        dead_cards: Set[int] = set()
        try:
            # Карты на всех досках
            for board in self.boards:
                for row_name in board.ROW_NAMES:
                    for card_int in board.rows[row_name]:
                        if isinstance(card_int, int) and card_int is not None and card_int != INVALID_CARD and card_int > 0:
                            dead_cards.add(card_int)

            # Карты в руке игрока (обычной или ФЛ)
            player_hand = self.get_player_hand(perspective_player_idx)
            if player_hand:
                dead_cards.update(c for c in player_hand if isinstance(c, int) and c is not None and c != INVALID_CARD and c > 0)

            # Карты из приватного сброса игрока
            if 0 <= perspective_player_idx < len(self.private_discard):
                dead_cards.update(c for c in self.private_discard[perspective_player_idx] if isinstance(c, int) and c is not None and c != INVALID_CARD and c > 0)

        except Exception as e:
            logger.error(f"Error collecting dead cards for player {perspective_player_idx}: {e}", exc_info=True)
            # Возвращаем то, что успели собрать, или пустое множество
            return dead_cards or set()

        return dead_cards

    def get_player_to_move(self) -> int:
        """
        Определяет индекс игрока, который должен ходить следующим.
        Возвращает -1, если раунд завершен или никто не может ходить (ожидание).
        """
        if self.is_round_over():
            return -1

        if self.is_fantasyland_round:
            # В ФЛ раунде ищем первого игрока (ФЛ или нет), кто не закончил и имеет карты
            # Начинаем с игрока слева от дилера
            for i in range(self.NUM_PLAYERS):
                player_idx = (self.dealer_idx + 1 + i) % self.NUM_PLAYERS
                if not self._player_finished_round[player_idx] and self.get_player_hand(player_idx):
                    return player_idx
            return -1 # Никто не может ходить (все закончили или ждут карт)
        else: # Обычный раунд
            current_p = self._internal_current_player_idx
            other_p = (current_p + 1) % self.NUM_PLAYERS

            can_current_act = not self._player_finished_round[current_p] and self.get_player_hand(current_p) is not None
            can_other_act = not self._player_finished_round[other_p] and self.get_player_hand(other_p) is not None

            if can_current_act:
                return current_p # Текущий игрок ходит
            elif can_other_act:
                # Текущий не может, но другой может -> Ожидание (ход еще не перешел)
                return -1
            else:
                # Никто не может ходить (оба ждут карт или закончили)
                return -1

    # --- Методы для изменения состояния ---

    def start_new_round(self, dealer_button_idx: int):
        """Начинает новый раунд."""
        logger.info(f"Starting new round. Requested dealer: {dealer_button_idx}")
        # Сохраняем статус ФЛ для нового раунда
        fl_status_for_new_round = list(self.next_fantasyland_status)
        fl_cards_for_new_round = list(self.fantasyland_cards_to_deal)
        logger.info(f"Carrying over FL status: {fl_status_for_new_round}, cards: {fl_cards_for_new_round}")

        # Сбрасываем состояние через __init__
        self.__init__(dealer_idx=dealer_button_idx,
                      fantasyland_status=fl_status_for_new_round,
                      fantasyland_cards_to_deal=fl_cards_for_new_round)
        self.street = 1
        self._internal_current_player_idx = (self.dealer_idx + 1) % self.NUM_PLAYERS
        self._last_player_acted = None
        logger.info(f"New round state initialized. Dealer: {self.dealer_idx}, Player to start: {self._internal_current_player_idx}, FL Round: {self.is_fantasyland_round}")

        # Раздаем начальные карты
        if self.is_fantasyland_round:
            self._deal_fantasyland_hands()
            # Раздаем улицу 1 не-ФЛ игрокам, если они есть
            for i in range(self.NUM_PLAYERS):
                if not self.fantasyland_status[i]:
                    logger.debug(f"Dealing Street 1 cards to non-FL player {i}")
                    self._deal_street_to_player(i)
        else:
            # --- ИСПРАВЛЕНО: Раздаем только первому игроку ---
            player_to_deal = self._internal_current_player_idx
            logger.debug(f"Dealing Street 1 cards to player {player_to_deal}")
            self._deal_street_to_player(player_to_deal)

        # Раздаем карты второму игроку, если нужно (например, если оба в ФЛ или оба не в ФЛ и улица сменилась)
        # Эта логика теперь в advance_state и _deal_cards_if_needed
        # self._deal_cards_if_needed() # Убрано отсюда, будет вызвано в advance_state
        logger.info("Initial cards dealt for the new round.")


    def apply_action(self, player_idx: int, action: Any) -> 'GameState':
        """
        Применяет легальное действие ОБЫЧНОГО ХОДА к копии состояния.
        Возвращает новое состояние или выбрасывает исключение при ошибке.
        """
        if not (0 <= player_idx < self.NUM_PLAYERS):
            raise IndexError(f"Invalid player index: {player_idx}")
        if self.is_fantasyland_round and self.fantasyland_status[player_idx]:
            raise RuntimeError(f"Player {player_idx} is in Fantasyland. Use apply_fantasyland_placement/foul.")
        if self._player_finished_round[player_idx]:
             raise RuntimeError(f"Player {player_idx} has already finished the round.")
        # --- ИСПРАВЛЕНО: Проверяем, действительно ли ход этого игрока ---
        if self.get_player_to_move() != player_idx:
             raise RuntimeError(f"Attempted to apply action for player {player_idx}, but it's player {self.get_player_to_move()}'s turn.")


        new_state = self.copy()
        board = new_state.boards[player_idx]
        current_hand = new_state.current_hands.get(player_idx)

        if not current_hand:
            raise RuntimeError(f"Player {player_idx} has no hand to apply action.")

        logger.debug(f"Applying action for Player {player_idx} on Street {new_state.street}")

        # --- Применение действия ---
        try:
            if new_state.street == 1:
                if len(current_hand) != 5: raise ValueError(f"Invalid hand size {len(current_hand)} for street 1.")
                if not isinstance(action, tuple) or len(action) != 2 or not isinstance(action[0], tuple) or action[1] != tuple():
                     raise TypeError(f"Invalid action format for street 1: {type(action)}, {action}")
                placements = action[0]
                if len(placements) != 5: raise ValueError("Street 1 action requires exactly 5 placements.")

                placed_cards_in_action = set()
                hand_set = set(current_hand)
                for place_data in placements:
                    if not isinstance(place_data, tuple) or len(place_data) != 3: raise ValueError("Invalid placement format.")
                    card, row, idx = place_data
                    if card not in hand_set: raise ValueError(f"Card {card_to_str(card)} in action not found in hand.")
                    if card in placed_cards_in_action: raise ValueError(f"Duplicate card {card_to_str(card)} in action.")
                    if not board.add_card(card, row, idx):
                        raise RuntimeError(f"Failed to add card {card_to_str(card)} to {row}[{idx}]. Slot might be occupied or invalid.")
                    placed_cards_in_action.add(card)

            elif 2 <= new_state.street <= 5:
                if len(current_hand) != 3: raise ValueError(f"Invalid hand size {len(current_hand)} for street {new_state.street}.")
                if not isinstance(action, tuple) or len(action) != 3: raise TypeError(f"Invalid action format for street {new_state.street}: {type(action)}")
                # Действие: (placement1, placement2, discard_card)
                # placement: (card_int, row_name, index)
                # Проверяем типы элементов действия
                if not (isinstance(action[0], tuple) and len(action[0]) == 3 and isinstance(action[0][0], int) and
                        isinstance(action[1], tuple) and len(action[1]) == 3 and isinstance(action[1][0], int) and
                        isinstance(action[2], int)):
                     raise TypeError(f"Invalid element types in Pineapple action: {action}")

                p1, p2, discard = action
                c1, r1, i1 = p1
                c2, r2, i2 = p2
                action_cards = {c1, c2, discard}
                hand_set = set(current_hand)

                if len(action_cards) != 3: raise ValueError("Duplicate cards specified in action/discard.")
                if action_cards != hand_set: raise ValueError("Action cards do not match player's hand.")

                if not board.add_card(c1, r1, i1):
                    raise RuntimeError(f"Failed to add card {card_to_str(c1)} to {r1}[{i1}].")
                try:
                    if not board.add_card(c2, r2, i2):
                        raise RuntimeError(f"Failed to add card {card_to_str(c2)} to {r2}[{i2}].")
                except Exception as e_add2:
                    board.remove_card(r1, i1) # Откат первой карты
                    raise e_add2

                new_state.private_discard[player_idx].append(discard)
            else:
                raise ValueError(f"Cannot apply action on invalid street {new_state.street}.")

            # Обновляем состояние после успешного применения
            new_state.current_hands[player_idx] = None
            new_state._player_acted_this_street[player_idx] = True
            new_state._last_player_acted = player_idx

            # --- ИСПРАВЛЕНО: Проверяем завершение доски и обновляем статус ФЛ ---
            if board.is_complete():
                logger.info(f"Player {player_idx} completed their board.")
                new_state._player_finished_round[player_idx] = True
                new_state._check_foul_and_update_fl_status(player_idx)

            logger.debug(f"Action applied successfully for Player {player_idx}.")
            return new_state

        except (ValueError, TypeError, RuntimeError, IndexError) as e:
            logger.error(f"Error applying action for Player {player_idx}: {e}", exc_info=True)
            raise # Перевыбрасываем исключение


    def apply_fantasyland_placement(self, player_idx: int, placement: Dict[str, List[int]], discarded: List[int]) -> 'GameState':
        """Применяет результат размещения Fantasyland."""
        if not (0 <= player_idx < self.NUM_PLAYERS): raise IndexError(f"Invalid player index: {player_idx}")
        if not self.is_fantasyland_round or not self.fantasyland_status[player_idx]: raise RuntimeError(f"Player {player_idx} is not in Fantasyland.")
        if self._player_finished_round[player_idx]: raise RuntimeError(f"Player {player_idx} has already finished the round.")
        # В ФЛ раунде ход может быть у любого, кто не закончил

        new_state = self.copy()
        board = new_state.boards[player_idx]
        original_hand = new_state.fantasyland_hands[player_idx]

        if not original_hand: raise RuntimeError(f"Player {player_idx} has no Fantasyland hand.")

        logger.debug(f"Applying Fantasyland placement for Player {player_idx}")

        try:
            # Валидация входных данных
            if not isinstance(placement, dict) or not all(k in PlayerBoard.ROW_NAMES for k in placement): raise TypeError("Invalid placement format.")
            if not isinstance(discarded, list): raise TypeError("Discarded cards must be a list.")

            placed_list: List[int] = []
            for row_name in PlayerBoard.ROW_NAMES:
                 row_cards = placement.get(row_name, [])
                 if len(row_cards) != PlayerBoard.ROW_CAPACITY[row_name]: raise ValueError(f"Incorrect card count for row '{row_name}'.")
                 # Проверяем валидность карт в размещении
                 for c in row_cards:
                      if not isinstance(c, int) or c == INVALID_CARD or c <= 0:
                           raise ValueError(f"Invalid card '{c}' found in placement row '{row_name}'.")
                      placed_list.append(c)

            placed_set = set(placed_list)
            discard_set = set(d for d in discarded if isinstance(d, int) and d is not None and d != INVALID_CARD and d > 0)
            hand_set = set(c for c in original_hand if isinstance(c, int) and c is not None and c != INVALID_CARD and c > 0)
            n_hand = len(hand_set)
            expected_discard = max(0, n_hand - 13)

            if len(placed_list) != 13 or len(placed_set) != 13: raise ValueError(f"Placement must contain exactly 13 unique valid cards (found {len(placed_set)}).")
            # Сравниваем количество валидных карт в сбросе
            if len(discard_set) != expected_discard: raise ValueError(f"Expected {expected_discard} unique valid discarded cards, got {len(discard_set)}. Hand size: {n_hand}")
            if not placed_set.isdisjoint(discard_set): raise ValueError("Overlap between placed and discarded cards.")
            if placed_set.union(discard_set) != hand_set: raise ValueError("Placed and discarded cards do not match the original hand.")

            # Устанавливаем доску
            board.set_full_board(placement['top'], placement['middle'], placement['bottom'])

            # Обновляем состояние
            new_state.private_discard[player_idx].extend(list(discard_set)) # Добавляем валидный сброс
            new_state.fantasyland_hands[player_idx] = None
            new_state._player_finished_round[player_idx] = True
            new_state._check_foul_and_update_fl_status(player_idx) # Проверяем фол и Re-Fantasy
            new_state._last_player_acted = player_idx

            logger.info(f"Fantasyland placement applied for Player {player_idx}. Foul: {board.is_foul}. Next FL: {new_state.next_fantasyland_status[player_idx]}")
            return new_state

        except (ValueError, TypeError, RuntimeError) as e:
            logger.error(f"Error applying Fantasyland placement for Player {player_idx}: {e}", exc_info=True)
            logger.warning(f"Applying Fantasyland foul for Player {player_idx} due to error.")
            # Применяем фол, если размещение не удалось
            return new_state.apply_fantasyland_foul(player_idx, original_hand)

    def apply_fantasyland_foul(self, player_idx: int, hand_to_discard: List[int]) -> 'GameState':
        """Применяет фол в Fantasyland."""
        if not (0 <= player_idx < self.NUM_PLAYERS): raise IndexError(f"Invalid player index: {player_idx}")

        logger.warning(f"Applying Fantasyland foul for Player {player_idx}.")
        new_state = self.copy()
        board = new_state.boards[player_idx]

        # Устанавливаем фол и очищаем доску
        board.is_foul = True
        board.rows = {name: [None] * capacity for name, capacity in PlayerBoard.ROW_CAPACITY.items()}
        board._cards_placed = 0
        board._is_complete = False
        board._reset_caches()
        board._cached_royalties = {'top': 0, 'middle': 0, 'bottom': 0}

        # Добавляем карты руки в сброс
        valid_discard = [c for c in hand_to_discard if isinstance(c, int) and c is not None and c != INVALID_CARD and c > 0]
        new_state.private_discard[player_idx].extend(valid_discard)
        new_state.fantasyland_hands[player_idx] = None
        new_state.current_hands[player_idx] = None

        new_state._player_finished_round[player_idx] = True
        new_state.next_fantasyland_status[player_idx] = False
        new_state.fantasyland_cards_to_deal[player_idx] = 0
        new_state._last_player_acted = player_idx

        return new_state

    def advance_state(self) -> 'GameState':
         """
         Продвигает состояние игры после хода: передает ход, меняет улицу, раздает карты.
         Возвращает НОВЫЙ объект GameState.
         """
         if self.is_round_over():
              # logger.debug("advance_state called on completed round. No change.")
              return self # Возвращаем себя, если раунд уже закончен

         new_state = self.copy()
         initial_state_repr = self.get_state_representation() # Для проверки изменений
         logger.debug(f"Advancing state from Street {new_state.street}, Player {new_state._internal_current_player_idx}, LastActed: {new_state._last_player_acted}")

         # --- Логика перехода хода и улицы ---
         last_acted = new_state._last_player_acted
         deal_both = False # Флаг, нужно ли раздавать обоим (при смене улицы)

         if not new_state.is_fantasyland_round:
              # Обычный раунд
              current_p = new_state._internal_current_player_idx

              # Проверяем, нужно ли менять улицу
              # Улица меняется, если оба игрока сходили на текущей улице
              change_street = all(new_state._player_acted_this_street)

              if change_street and new_state.street < 5:
                   new_state.street += 1
                   logger.info(f"--- Advancing to Street {new_state.street} ---")
                   new_state._player_acted_this_street = [False] * new_state.NUM_PLAYERS
                   # Ход переходит к игроку после дилера
                   new_state._internal_current_player_idx = (new_state.dealer_idx + 1) % new_state.NUM_PLAYERS
                   new_state._last_player_acted = None # Сбрасываем после смены улицы
                   deal_both = True # Раздаем обоим на новой улице
                   logger.debug(f"Street advanced. Next player: {new_state._internal_current_player_idx}. Dealing both.")
              elif last_acted is not None:
                   # Улицу не меняли, передаем ход следующему, если он еще не ходил и не закончил
                   next_p_candidate = (last_acted + 1) % new_state.NUM_PLAYERS
                   if not new_state._player_acted_this_street[next_p_candidate] and not new_state._player_finished_round[next_p_candidate]:
                        new_state._internal_current_player_idx = next_p_candidate
                        logger.debug(f"Turn passed to Player {next_p_candidate}")
                   # else: Ход остается у того же игрока (например, если другой закончил)

         else: # Раунд Фантазии
              # В ФЛ раунде advance_state в основном отвечает за раздачу карт не-ФЛ игрокам
              # и смену улицы для них
              if last_acted is not None and not new_state.fantasyland_status[last_acted]:
                   # Если последним ходил не-ФЛ игрок, проверяем, нужно ли менять улицу для не-ФЛ
                   num_non_fl = sum(1 for status in new_state.fantasyland_status if not status)
                   can_change_street = False
                   if num_non_fl == 1:
                        # Если только один не-ФЛ, улица меняется после каждого его хода
                        can_change_street = new_state._player_acted_this_street[last_acted]
                   elif num_non_fl == 2:
                        # Если два не-ФЛ, улица меняется, когда оба сходили
                        can_change_street = all(new_state._player_acted_this_street[i] for i, status in enumerate(new_state.fantasyland_status) if not status)

                   if can_change_street and new_state.street < 5:
                        new_state.street += 1
                        logger.info(f"--- Advancing non-FL players to Street {new_state.street} ---")
                        # Сбрасываем флаги только для не-ФЛ
                        for i in range(new_state.NUM_PLAYERS):
                             if not new_state.fantasyland_status[i]:
                                  new_state._player_acted_this_street[i] = False
                        deal_both = True # Раздаем всем не-ФЛ игрокам
                        logger.debug(f"Street advanced for non-FL. Dealing cards.")

         # --- Логика раздачи карт ---
         new_state._deal_cards_if_needed(deal_both_on_street_change=deal_both)

         # Логгируем, если состояние не изменилось (может быть нормой, если оба ждут карт)
         if new_state.get_state_representation() == initial_state_repr and not new_state.is_round_over():
              logger.debug("advance_state did not change the game state representation.")

         return new_state

    # --- Внутренние методы ---

    def _deal_cards_if_needed(self, deal_both_on_street_change: bool = False):
        """Раздает карты игрокам, которым они нужны на текущем шаге."""
        players_to_deal: List[int] = []

        if self.is_fantasyland_round:
            # В ФЛ раунде карты раздаются только не-ФЛ игрокам
            for p_idx in range(self.NUM_PLAYERS):
                # Раздаем, если: не в ФЛ И не закончил раунд И нет текущей руки
                if (not self.fantasyland_status[p_idx] and
                        not self._player_finished_round[p_idx] and
                        self.current_hands.get(p_idx) is None):
                    # Раздаем, если улица сменилась ИЛИ если это ход этого игрока
                    if deal_both_on_street_change or self._internal_current_player_idx == p_idx:
                         players_to_deal.append(p_idx)
        else: # Обычный раунд
            # Если улица только что сменилась, раздаем обоим активным игрокам
            if deal_both_on_street_change:
                 for p_idx in range(self.NUM_PLAYERS):
                      if not self._player_finished_round[p_idx] and self.current_hands.get(p_idx) is None:
                           players_to_deal.append(p_idx)
            else:
                 # Иначе раздаем только текущему игроку, если ему нужно
                 p_idx = self._internal_current_player_idx
                 if not self._player_finished_round[p_idx] and self.current_hands.get(p_idx) is None:
                      players_to_deal.append(p_idx)

        if players_to_deal:
            logger.debug(f"Need to deal cards to players: {players_to_deal} on Street {self.street}")
            for p_idx_deal in players_to_deal:
                self._deal_street_to_player(p_idx_deal)

    def _deal_street_to_player(self, player_idx: int):
        """Раздает карты для текущей улицы указанному игроку."""
        if self._player_finished_round[player_idx]:
            # logger.warning(f"Attempted to deal cards to finished player {player_idx}.")
            return
        if self.current_hands.get(player_idx) is not None:
            # logger.warning(f"Attempted to deal cards to player {player_idx} who already has a hand.")
            return
        if self.street > 5 or self.street < 1:
             logger.error(f"Attempted to deal cards on invalid street {self.street}.")
             return

        # --- ИСПРАВЛЕНО: Количество карт зависит от улицы ---
        num_cards = 5 if self.street == 1 else 3
        logger.info(f"Dealing {num_cards} cards to Player {player_idx} for Street {self.street}")
        try:
            dealt_cards = self.deck.deal(num_cards)
            if len(dealt_cards) < num_cards:
                 logger.warning(f"Deck ran out of cards while dealing to Player {player_idx}. Dealt {len(dealt_cards)}/{num_cards}.")
            # Сохраняем даже пустую руку, если карт не хватило
            self.current_hands[player_idx] = dealt_cards
            logger.debug(f"Player {player_idx} received hand: {[card_to_str(c) for c in dealt_cards]}")
        except Exception as e:
            logger.error(f"Error dealing cards for Street {self.street} to Player {player_idx}: {e}", exc_info=True)
            self.current_hands[player_idx] = [] # Пустая рука при ошибке

    def _deal_fantasyland_hands(self):
        """Раздает карты игрокам в Fantasyland."""
        logger.info("Dealing Fantasyland hands...")
        for i in range(self.NUM_PLAYERS):
            if self.fantasyland_status[i]:
                num_cards = self.fantasyland_cards_to_deal[i]
                if not (14 <= num_cards <= 17):
                    logger.warning(f"Invalid FL card count {num_cards} for Player {i}. Defaulting to 14.")
                    num_cards = 14
                logger.info(f"Dealing {num_cards} Fantasyland cards to Player {i}")
                try:
                    dealt_cards = self.deck.deal(num_cards)
                    if len(dealt_cards) < num_cards:
                         logger.warning(f"Deck ran out of cards while dealing FL to Player {i}. Dealt {len(dealt_cards)}/{num_cards}.")
                    self.fantasyland_hands[i] = dealt_cards
                    logger.debug(f"Player {i} received FL hand ({len(dealt_cards)} cards)")
                except Exception as e:
                    logger.error(f"Error dealing Fantasyland hand to Player {i}: {e}", exc_info=True)
                    self.fantasyland_hands[i] = [] # Пустая рука при ошибке

    def _check_foul_and_update_fl_status(self, player_idx: int):
        """Проверяет фол и обновляет статус Fantasyland для следующего раунда."""
        board = self.boards[player_idx]
        if not board.is_complete():
            logger.warning(f"_check_foul_and_update_fl_status called for incomplete board P{player_idx}.")
            return

        is_foul = board.check_and_set_foul() # Проверяем и устанавливаем флаг фола на доске
        logger.info(f"Checked board for Player {player_idx}. Foul: {is_foul}")

        # Сбрасываем статус ФЛ по умолчанию
        self.next_fantasyland_status[player_idx] = False
        self.fantasyland_cards_to_deal[player_idx] = 0

        if not is_foul:
            # Если не фол, проверяем условия ФЛ / Re-ФЛ
            if self.fantasyland_status[player_idx]: # Если игрок БЫЛ в ФЛ
                if board.check_fantasyland_stay_conditions():
                    logger.info(f"Player {player_idx} stays in Fantasyland (Re-Fantasy).")
                    self.next_fantasyland_status[player_idx] = True
                    self.fantasyland_cards_to_deal[player_idx] = 14 # Re-Fantasy всегда 14 карт
                else:
                     logger.info(f"Player {player_idx} does not qualify for Re-Fantasy.")
            else: # Если игрок НЕ был в ФЛ
                fl_cards = board.get_fantasyland_qualification_cards()
                if fl_cards > 0:
                    logger.info(f"Player {player_idx} qualifies for Fantasyland with {fl_cards} cards.")
                    self.next_fantasyland_status[player_idx] = True
                    self.fantasyland_cards_to_deal[player_idx] = fl_cards
                else:
                     logger.info(f"Player {player_idx} does not qualify for Fantasyland.")
        else:
             logger.info(f"Player {player_idx} fouled. No Fantasyland qualification.")


    # --- Методы для MCTS и Сериализации ---

    def get_legal_actions_for_player(self, player_idx: int) -> List[Any]:
        """Возвращает список легальных действий для игрока."""
        if not (0 <= player_idx < self.NUM_PLAYERS):
            logger.error(f"Invalid player index {player_idx} requested in get_legal_actions.")
            return []
        if self._player_finished_round[player_idx]:
            return []
        # Проверяем, действительно ли ход этого игрока
        if self.get_player_to_move() != player_idx:
             # logger.debug(f"get_legal_actions called for P{player_idx}, but it's P{self.get_player_to_move()}'s turn.")
             return [] # Не ход этого игрока

        if self.is_fantasyland_round and self.fantasyland_status[player_idx]:
            hand = self.fantasyland_hands[player_idx]
            # Для MCTS возвращаем специальное действие, сигнализирующее о необходимости решить ФЛ
            # Передаем копию руки в виде кортежа для хешируемости
            return [("FANTASYLAND_INPUT", tuple(hand))] if hand else []
        else: # Обычный раунд
            hand = self.current_hands.get(player_idx)
            if not hand:
                return [] # Нет карт - нет действий

            if self.street == 1:
                return self._get_legal_actions_street1(player_idx, hand) if len(hand) == 5 else []
            elif 2 <= self.street <= 5:
                return self._get_legal_actions_pineapple(player_idx, hand) if len(hand) == 3 else []
            else:
                logger.warning(f"Requesting legal actions on invalid street {self.street}.")
                return []

    def _get_legal_actions_street1(self, player_idx: int, hand: List[int]) -> List[Tuple[Tuple[Tuple[int, str, int], ...], Tuple]]:
        """Генерирует легальные действия для улицы 1 (размещение 5 карт)."""
        board = self.boards[player_idx]
        slots = board.get_available_slots()
        if len(slots) < 5: return [] # Недостаточно слотов

        actions: List[Tuple[Tuple[Tuple[int, str, int], ...], Tuple]] = []
        MAX_ACTIONS_STREET1 = 500 # Ограничение для производительности
        count = 0

        # Генерируем комбинации слотов
        for slot_combo in combinations(slots, 5):
            if count >= MAX_ACTIONS_STREET1: break
            # Генерируем перестановки карт для этих слотов
            # Вместо всех перестановок (5! = 120), можно взять случайную или несколько
            # Для MCTS достаточно одной случайной перестановки на комбинацию слотов
            card_perm = random.sample(hand, len(hand)) # Случайная перестановка карт
            # --- ИСПРАВЛЕНО: Сортируем placement для каноничности ---
            placement = tuple(sorted([(card_perm[i], slot_combo[i][0], slot_combo[i][1]) for i in range(5)]))
            # Действие для улицы 1: (tuple_of_placements, empty_tuple_for_discard)
            action = (placement, tuple())
            actions.append(action)
            count += 1
        # Убираем дубликаты, если они возникли из-за сортировки
        return list(set(actions))

    def _get_legal_actions_pineapple(self, player_idx: int, hand: List[int]) -> List[Tuple[Tuple[int, str, int], Tuple[int, str, int], int]]:
        """Генерирует легальные действия для улиц 2-5 (размещение 2 из 3)."""
        board = self.boards[player_idx]
        slots = board.get_available_slots()
        if len(slots) < 2: return [] # Недостаточно слотов

        actions: List[Tuple[Tuple[int, str, int], Tuple[int, str, int], int]] = []
        # Перебираем карту для сброса
        for i in range(3):
            discard_card = hand[i]
            cards_to_place = [hand[j] for j in range(3) if i != j]
            c1, c2 = cards_to_place[0], cards_to_place[1]

            # Перебираем комбинации слотов для размещения
            for slot1_info, slot2_info in combinations(slots, 2):
                r1, idx1 = slot1_info
                r2, idx2 = slot2_info
                # Два варианта размещения карт в выбранные слоты
                # Действие: (placement1, placement2, discard_card)
                # --- ИСПРАВЛЕНО: Сортируем placement для каноничности ---
                # Сортируем по строке карты (не идеально, но просто)
                p1_t = (c1, r1, idx1); p2_t = (c2, r2, idx2)
                action1 = tuple(sorted((p1_t, p2_t))) + (discard_card,)
                p1_t2 = (c2, r1, idx1); p2_t2 = (c1, r2, idx2)
                action2 = tuple(sorted((p1_t2, p2_t2))) + (discard_card,)

                actions.append(action1) # type: ignore
                if action1 != action2: # Добавляем второй вариант, если он отличается
                     actions.append(action2) # type: ignore

        # Убираем дубликаты
        return list(set(actions))

    def get_state_representation(self) -> tuple:
        """Возвращает хешируемое представление состояния."""
        # Преобразуем изменяемые части в неизменяемые (кортежи)
        boards_tuple = tuple(b.get_board_state_tuple() for b in self.boards)
        # Сортируем руки и сброс для каноничности
        current_hands_tuple = tuple(
            tuple(sorted(self.current_hands.get(i, []))) if self.current_hands.get(i) is not None else None
            for i in range(self.NUM_PLAYERS)
        )
        fantasyland_hands_tuple = tuple(
            tuple(sorted(self.fantasyland_hands[i])) if self.fantasyland_hands[i] is not None else None
            for i in range(self.NUM_PLAYERS)
        )
        private_discard_tuple = tuple(tuple(sorted(p_d)) for p_d in self.private_discard)
        # --- ИСПРАВЛЕНО: Добавляем колоду в представление ---
        deck_tuple = tuple(sorted(self.deck.cards))

        return (
            boards_tuple,
            deck_tuple, # Добавлено
            private_discard_tuple,
            self.dealer_idx,
            self._internal_current_player_idx,
            self.street,
            tuple(self.fantasyland_status),
            self.is_fantasyland_round,
            tuple(self.fantasyland_cards_to_deal),
            current_hands_tuple,
            fantasyland_hands_tuple,
            tuple(self._player_acted_this_street),
            tuple(self._player_finished_round),
            self._last_player_acted
        )

    def copy(self) -> 'GameState':
        """Создает глубокую копию объекта состояния игры."""
        # --- ИСПРАВЛЕНО: Ручное глубокое копирование ---
        new_state = GameState.__new__(GameState)
        new_state.boards = [b.copy() for b in self.boards]
        new_state.deck = self.deck.copy()
        new_state.private_discard = [list(p) for p in self.private_discard]
        new_state.dealer_idx = self.dealer_idx
        new_state._internal_current_player_idx = self._internal_current_player_idx
        new_state.street = self.street
        # Копируем словарь current_hands с копированием списков внутри
        new_state.current_hands = {idx: list(h) if h is not None else None for idx, h in self.current_hands.items()}
        # Копируем список fantasyland_hands с копированием списков внутри
        new_state.fantasyland_hands = [list(h) if h is not None else None for h in self.fantasyland_hands]
        # Копируем списки bool/int
        new_state.fantasyland_status = list(self.fantasyland_status)
        new_state.next_fantasyland_status = list(self.next_fantasyland_status)
        new_state.fantasyland_cards_to_deal = list(self.fantasyland_cards_to_deal)
        new_state.is_fantasyland_round = self.is_fantasyland_round
        new_state._player_acted_this_street = list(self._player_acted_this_street)
        new_state._player_finished_round = list(self._player_finished_round)
        new_state._last_player_acted = self._last_player_acted
        return new_state

    def __hash__(self):
        """Возвращает хеш состояния."""
        return hash(self.get_state_representation())

    def __eq__(self, other):
        """Сравнивает два состояния игры."""
        if not isinstance(other, GameState):
            return NotImplemented
        return self.get_state_representation() == other.get_state_representation()

    def to_dict(self) -> Dict[str, Any]:
        """Сериализует состояние игры в словарь для JSON."""
        try:
            boards_data = []
            for b in self.boards:
                 board_dict = {
                      'rows': {r: [card_to_str(c) for c in b.rows[r]] for r in b.ROW_NAMES},
                      '_cards_placed': b._cards_placed,
                      'is_foul': b.is_foul,
                      '_is_complete': b._is_complete
                 }
                 boards_data.append(board_dict)

            return {
                "boards": boards_data,
                "private_discard": [[card_to_str(c) for c in p] for p in self.private_discard],
                "dealer_idx": self.dealer_idx,
                "_internal_current_player_idx": self._internal_current_player_idx,
                "street": self.street,
                "current_hands": {str(i): [card_to_str(c) for c in h] if h is not None else None for i, h in self.current_hands.items()},
                "fantasyland_status": self.fantasyland_status,
                "next_fantasyland_status": self.next_fantasyland_status,
                "fantasyland_cards_to_deal": self.fantasyland_cards_to_deal,
                "is_fantasyland_round": self.is_fantasyland_round,
                "fantasyland_hands": [[card_to_str(c) for c in h] if h is not None else None for h in self.fantasyland_hands],
                "_player_acted_this_street": self._player_acted_this_street,
                "_player_finished_round": self._player_finished_round,
                "_last_player_acted": self._last_player_acted,
                "deck_remaining": sorted([card_to_str(c) for c in self.deck.get_remaining_cards()])
            }
        except Exception as e:
            logger.error(f"Error during GameState serialization: {e}", exc_info=True)
            return {"error": "Serialization failed"} # Возвращаем словарь с ошибкой

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameState':
        """Десериализует состояние игры из словаря."""
        try:
            num_p = len(data.get("boards", []))
            if num_p != cls.NUM_PLAYERS and num_p != 0:
                 logger.warning(f"Deserializing GameState with {num_p} players, expected {cls.NUM_PLAYERS}. Proceeding cautiously.")
            num_p = cls.NUM_PLAYERS if num_p == 0 else num_p # Определяем кол-во игроков

            boards = []
            known_cards_int: Set[int] = set() # Собираем все известные карты (int)

            # Восстанавливаем доски
            for b_idx, b_data in enumerate(data.get("boards", [{} for _ in range(num_p)])):
                b = PlayerBoard()
                rows_raw = b_data.get('rows', {})
                cards_count = 0
                for r_name in b.ROW_NAMES:
                    row_cards_int: List[Optional[int]] = []
                    capacity = b.ROW_CAPACITY[r_name]
                    row_strs = rows_raw.get(r_name, [CARD_PLACEHOLDER] * capacity)
                    # --- ИСПРАВЛЕНО: Убедимся, что длина списка правильная ---
                    row_strs = (row_strs + [CARD_PLACEHOLDER] * capacity)[:capacity]
                    for c_str in row_strs:
                        card_int = None
                        if c_str != CARD_PLACEHOLDER:
                            try:
                                c_int_parsed = card_from_str(c_str)
                                if c_int_parsed != INVALID_CARD and c_int_parsed > 0:
                                    card_int = c_int_parsed
                                    if card_int in known_cards_int:
                                         logger.warning(f"Duplicate card {c_str} detected during board deserialization for P{b_idx}.")
                                    known_cards_int.add(card_int)
                                    cards_count += 1
                            except ValueError: pass # Ошибка парсинга -> None
                        row_cards_int.append(card_int)
                    b.rows[r_name] = row_cards_int
                # Восстанавливаем внутренние состояния доски
                b._cards_placed = b_data.get('_cards_placed', cards_count)
                b._is_complete = b_data.get('_is_complete', b._cards_placed == 13)
                b._reset_caches() # Сбрасываем кэши при загрузке
                # Пересчитываем фол на основе загруженных рядов, если доска полная
                b.is_foul = b.check_and_set_foul() if b._is_complete else False
                boards.append(b)

            # Восстанавливаем приватный сброс
            private_discard: List[List[int]] = []
            for p_idx, p_d_strs in enumerate(data.get("private_discard", [[] for _ in range(num_p)])):
                 p_d_ints: List[int] = []
                 for c_str in p_d_strs:
                      try:
                           c_int = card_from_str(c_str)
                           if c_int != INVALID_CARD and c_int > 0:
                                p_d_ints.append(c_int)
                                if c_int in known_cards_int:
                                     logger.warning(f"Duplicate card {c_str} detected between board and discard P{p_idx}.")
                                known_cards_int.add(c_int)
                      except ValueError: pass # Игнорируем невалидные карты в сбросе
                 private_discard.append(p_d_ints)

            # Восстанавливаем текущие руки
            current_hands: Dict[int, Optional[List[int]]] = {}
            raw_c_hands = data.get("current_hands", {})
            for i in range(num_p):
                 h_strs = raw_c_hands.get(str(i))
                 if h_strs is not None:
                      h_ints: List[int] = []
                      for c_str in h_strs:
                           try:
                                c_int = card_from_str(c_str)
                                if c_int != INVALID_CARD and c_int > 0:
                                     h_ints.append(c_int)
                                     if c_int in known_cards_int:
                                          logger.warning(f"Duplicate card {c_str} detected between board/discard and hand P{i}.")
                                     known_cards_int.add(c_int)
                           except ValueError: pass
                      current_hands[i] = h_ints
                 else: current_hands[i] = None

            # Восстанавливаем руки Фантазии
            fantasyland_hands: List[Optional[List[int]]] = []
            raw_fl_hands = data.get("fantasyland_hands", [None] * num_p)
            for i in range(num_p):
                 h_strs = raw_fl_hands[i] if i < len(raw_fl_hands) else None
                 if h_strs is not None:
                      h_ints: List[int] = []
                      for c_str in h_strs:
                           try:
                                c_int = card_from_str(c_str)
                                if c_int != INVALID_CARD and c_int > 0:
                                     h_ints.append(c_int)
                                     if c_int in known_cards_int:
                                          logger.warning(f"Duplicate card {c_str} detected between board/discard/hand and FL hand P{i}.")
                                     known_cards_int.add(c_int)
                           except ValueError: pass
                      fantasyland_hands.append(h_ints)
                 else: fantasyland_hands.append(None)

            # Восстанавливаем колоду
            deck_remaining_int_from_data: Set[int] = set()
            for c_str in data.get("deck_remaining", []):
                 try:
                      c_int = card_from_str(c_str)
                      if c_int != INVALID_CARD and c_int > 0: deck_remaining_int_from_data.add(c_int)
                 except ValueError: pass
            # --- ИСПРАВЛЕНО: Более надежное восстановление колоды ---
            calculated_remaining = Deck.FULL_DECK_CARDS - known_cards_int
            if deck_remaining_int_from_data != calculated_remaining:
                 missing_in_data = calculated_remaining - deck_remaining_int_from_data
                 extra_in_data = deck_remaining_int_from_data - calculated_remaining
                 logger.warning(f"Deck inconsistency detected during deserialization. "
                                f"Missing in data: {[card_to_str(c) for c in missing_in_data]}. "
                                f"Extra in data: {[card_to_str(c) for c in extra_in_data]}. "
                                f"Using calculated remaining cards based on known cards.")
                 deck = Deck(cards=calculated_remaining)
            else:
                 deck = Deck(cards=deck_remaining_int_from_data)

            # Восстанавливаем остальные атрибуты
            def_b = [False] * num_p; def_i = [0] * num_p
            d_idx = data.get("dealer_idx", 0)
            int_cp_idx = data.get("_internal_current_player_idx", (d_idx + 1) % num_p)
            st = data.get("street", 0)
            fl_stat = data.get("fantasyland_status", list(def_b))
            next_fl_stat = data.get("next_fantasyland_status", list(def_b))
            fl_cards = data.get("fantasyland_cards_to_deal", list(def_i))
            is_fl = data.get("is_fantasyland_round", any(fl_stat))
            acted = data.get("_player_acted_this_street", list(def_b))
            finished = data.get("_player_finished_round", list(def_b))
            last_acted = data.get("_last_player_acted", None)

            # Создаем экземпляр и заполняем поля
            instance = cls.__new__(cls)
            instance.boards = boards
            instance.deck = deck
            instance.private_discard = private_discard
            instance.dealer_idx = d_idx
            instance._internal_current_player_idx = int_cp_idx
            instance.street = st
            instance.current_hands = current_hands
            instance.fantasyland_hands = fantasyland_hands
            instance.fantasyland_status = fl_stat
            instance.next_fantasyland_status = next_fl_stat
            instance.fantasyland_cards_to_deal = fl_cards
            instance.is_fantasyland_round = is_fl
            instance._player_acted_this_street = acted
            instance._player_finished_round = finished
            instance._last_player_acted = last_acted

            return instance

        except Exception as e:
            logger.error(f"Error during GameState deserialization: {e}", exc_info=True)
            raise ValueError("Failed to deserialize GameState from dictionary.") from e
