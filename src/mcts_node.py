# src/mcts_node.py v1.5
"""
Представление узла дерева MCTS для OFC Pineapple.
Содержит состояние игры, статистику посещений/наград и логику MCTS (выбор, расширение, симуляция).
Использует GameState.advance_state() для симуляции.
"""

import math
import time
import random
import multiprocessing
import traceback
import sys
import logging # Используем logging
from typing import Optional, Any, List, Tuple, Set, Dict
from collections import Counter

# Импорты из src пакета
from src.game_state import GameState
from src.fantasyland_solver import FantasylandSolver
from src.card import card_to_str, Card as CardUtils, INVALID_CARD # Используем алиас для утилит
from src.scoring import (
    RANK_CLASS_QUADS, RANK_CLASS_TRIPS, get_hand_rank_safe,
    check_board_foul, get_row_royalty, RANK_CLASS_PAIR,
    RANK_CLASS_HIGH_CARD
)

# Получаем логгер
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): logger.setLevel(logging.WARNING)


# Функция-воркер для параллельного роллаута (без изменений в этой версии)
def run_parallel_rollout(node_state_dict: dict) -> Tuple[float, Set[Any]]:
    """ Выполняет один роллаут из заданного состояния в отдельном процессе. """
    try:
        # Пересоздаем зависимости внутри воркера
        from src.game_state import GameState as WorkerGameState
        from src.mcts_node import MCTSNode as WorkerMCTSNode # Импортируем здесь для доступа к политикам

        game_state = WorkerGameState.from_dict(node_state_dict)
        # Если состояние уже терминальное, сразу возвращаем счет
        if game_state.is_round_over():
            score_p0 = game_state.get_terminal_score()
            return float(score_p0), set()

        # Создаем временный узел для вызова rollout (или просто вызываем статическую/внешнюю функцию rollout)
        # Передаем game_state напрямую в функцию симуляции, чтобы избежать создания лишнего узла
        reward, sim_actions = WorkerMCTSNode.static_rollout_simulation(game_state, perspective_player=0)
        return reward, sim_actions
    except Exception as e:
        print(f"Error in parallel rollout worker: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 0.0, set()


class MCTSNode:
    """ Узел в дереве поиска Монте-Карло (MCTS). """
    # (Атрибуты и __init__ без изменений)
    def __init__(self, game_state: GameState, parent: Optional['MCTSNode'] = None, action: Optional[Any] = None):
        self.game_state: GameState = game_state
        self.parent: Optional['MCTSNode'] = parent
        self.action: Optional[Any] = action
        self.children: Dict[Any, 'MCTSNode'] = {}
        self.untried_actions: Optional[List[Any]] = None
        self.visits: int = 0
        self.total_reward: float = 0.0 # Всегда с точки зрения P0
        self.rave_visits: Dict[Any, int] = {}
        self.rave_total_reward: Dict[Any, float] = {} # Всегда с точки зрения P0

    # (Методы _get_player_to_move, is_terminal без изменений)
    def _get_player_to_move(self) -> int:
        return self.game_state.get_player_to_move()

    def is_terminal(self) -> bool:
        return self.game_state.is_round_over()

    def expand(self) -> Optional['MCTSNode']:
        """ Расширяет узел, добавляя одного потомка для неиспробованного действия. """
        player_to_move = self._get_player_to_move()
        if player_to_move == -1:
            logger.debug("Expand called on terminal or waiting node.")
            return None # Нельзя расширить терминальный или ожидающий узел

        # Инициализируем неиспробованные действия, если нужно
        if self.untried_actions is None:
            try:
                self.untried_actions = self.game_state.get_legal_actions_for_player(player_to_move)
                random.shuffle(self.untried_actions)
                # Инициализируем RAVE статистику для хешируемых действий
                for act in self.untried_actions:
                     try:
                          hash(act)
                          if act not in self.rave_visits:
                               self.rave_visits[act] = 0
                               self.rave_total_reward[act] = 0.0
                     except TypeError: pass # Игнорируем нехешируемые
            except Exception as e_leg:
                 logger.error(f"Error getting legal actions during expand for P{player_to_move}: {e_leg}", exc_info=True)
                 self.untried_actions = [] # Считаем, что действий нет при ошибке

        # Если нет неиспробованных действий, расширение невозможно
        if not self.untried_actions:
            # logger.debug(f"No untried actions left to expand for P{player_to_move}.")
            return None

        # Берем следующее неиспробованное действие
        action = self.untried_actions.pop()
        next_state: Optional[GameState] = None
        original_state_repr = self.game_state.get_state_representation() # Для проверки изменений

        try:
            # --- ИСПРАВЛЕНО: Обработка FANTASYLAND_INPUT ---
            if isinstance(action, tuple) and action[0] == "FANTASYLAND_INPUT":
                 # Не расширяем узлы Фантазии здесь, они обрабатываются в choose_action агента
                 logger.debug("Skipping expansion of FANTASYLAND_INPUT action in MCTSNode.")
                 # Пытаемся взять следующее действие рекурсивно
                 return self.expand()
            else:
                 # Применяем обычное действие
                 next_state = self.game_state.apply_action(player_to_move, action)

            # --- ИСПРАВЛЕНО: Проверка, изменилось ли состояние ---
            if next_state is None:
                 raise RuntimeError("apply_action returned None")
            if next_state.get_state_representation() == original_state_repr:
                # Это может произойти, если apply_action не смогла применить ход
                # (хотя она должна выбрасывать исключение в таких случаях)
                logger.warning(f"apply_action did not change state for P{player_to_move}. Action: {action}")
                # Пытаемся взять следующее действие рекурсивно
                return self.expand()

        except Exception as e:
            logger.error(f"Error applying action during expand for P{player_to_move}: {e}", exc_info=True)
            # Пытаемся взять следующее действие рекурсивно
            return self.expand()

        # Создаем дочерний узел
        child_node = MCTSNode(next_state, parent=self, action=action)

        # Добавляем в словарь children, только если действие хешируемое
        try:
            hash(action)
            self.children[action] = child_node
        except TypeError:
             logger.warning(f"Action {action} is not hashable, child node created but not added to children dict.")
             # Узел создан, но не будет выбран через UCT.
             # Это может быть проблемой, если такие действия важны.

        return child_node


    @staticmethod
    def static_rollout_simulation(initial_game_state: GameState, perspective_player: int = 0) -> Tuple[float, Set[Any]]:
        """
        Статический метод для выполнения симуляции (rollout) из заданного состояния.
        Может быть вызван из параллельного воркера.
        """
        # --- ИСПРАВЛЕНО: Используем копию и улучшенную логику ---
        try:
            current_rollout_state = initial_game_state.copy()
        except Exception as e_copy:
             logger.error(f"Error copying state at start of static rollout: {e_copy}", exc_info=True)
             return 0.0, set() # Не можем продолжить без копии

        simulation_actions_set: Set[Any] = set()
        MAX_ROLLOUT_STEPS = 60 # Увеличено ограничение
        steps = 0

        while not current_rollout_state.is_round_over() and steps < MAX_ROLLOUT_STEPS:
            steps += 1
            player_to_act_rollout = current_rollout_state.get_player_to_move()

            if player_to_act_rollout != -1:
                # --- Ход игрока ---
                action: Optional[Any] = None
                next_rollout_state: Optional[GameState] = None
                is_fl_turn = current_rollout_state.is_fantasyland_round and current_rollout_state.fantasyland_status[player_to_act_rollout]

                try:
                    if is_fl_turn:
                        hand = current_rollout_state.fantasyland_hands[player_to_act_rollout]
                        if hand:
                            # Используем статическую версию эвристики
                            placement, discarded = MCTSNode._static_heuristic_fantasyland_placement(hand)
                            if placement and discarded is not None:
                                next_rollout_state = current_rollout_state.apply_fantasyland_placement(player_to_act_rollout, placement, discarded)
                            else:
                                logger.debug(f"Static FL heuristic failed for P{player_to_act_rollout}, applying foul.")
                                next_rollout_state = current_rollout_state.apply_fantasyland_foul(player_to_act_rollout, hand)
                        else:
                            logger.warning(f"FL Player {player_to_act_rollout} has no hand in static rollout. Applying foul.")
                            next_rollout_state = current_rollout_state.apply_fantasyland_foul(player_to_act_rollout, [])
                    else: # Обычный ход
                        hand = current_rollout_state.current_hands.get(player_to_act_rollout)
                        if hand:
                            possible_moves = current_rollout_state.get_legal_actions_for_player(player_to_act_rollout)
                            if possible_moves:
                                # Используем статическую версию политики
                                action = MCTSNode._static_heuristic_rollout_policy(current_rollout_state, player_to_act_rollout, possible_moves)
                                if action:
                                    # Добавляем действие в набор, только если оно хешируемое
                                    try: hash(action); simulation_actions_set.add(action)
                                    except TypeError: pass

                                    next_rollout_state_candidate = current_rollout_state.apply_action(player_to_act_rollout, action)
                                    if next_rollout_state_candidate is current_rollout_state or next_rollout_state_candidate == current_rollout_state:
                                         raise RuntimeError("apply_action returned same state")
                                    next_rollout_state = next_rollout_state_candidate
                                else:
                                     logger.warning(f"Static rollout policy returned None for P{player_to_act_rollout}. Applying foul.")
                                     board = current_rollout_state.boards[player_to_act_rollout]; board.is_foul = True; current_rollout_state._player_finished_round[player_to_act_rollout] = True
                                     if current_rollout_state.current_hands.get(player_to_act_rollout): current_rollout_state.private_discard[player_to_act_rollout].extend(current_rollout_state.current_hands[player_to_act_rollout]); current_rollout_state.current_hands[player_to_act_rollout] = None
                                     next_rollout_state = current_rollout_state # Возвращаем измененное состояние с фолом
                            else:
                                 logger.warning(f"No legal actions found for P{player_to_act_rollout} in static rollout. Applying foul.")
                                 board = current_rollout_state.boards[player_to_act_rollout]; board.is_foul = True; current_rollout_state._player_finished_round[player_to_act_rollout] = True
                                 if current_rollout_state.current_hands.get(player_to_act_rollout): current_rollout_state.private_discard[player_to_act_rollout].extend(current_rollout_state.current_hands[player_to_act_rollout]); current_rollout_state.current_hands[player_to_act_rollout] = None
                                 next_rollout_state = current_rollout_state
                        else:
                             logger.warning(f"Player {player_to_act_rollout} has no hand in static rollout. Applying foul.")
                             board = current_rollout_state.boards[player_to_act_rollout]; board.is_foul = True; current_rollout_state._player_finished_round[player_to_act_rollout] = True
                             next_rollout_state = current_rollout_state

                    # --- Обновление состояния ПОСЛЕ хода ---
                    if next_rollout_state is not None:
                        current_rollout_state = next_rollout_state
                    else:
                        # Это не должно происходить, если apply_action/foul возвращают состояние
                        logger.error(f"Next state became None after action for P{player_to_act_rollout} in static rollout. Stopping.")
                        break

                    # --- Продвигаем состояние (передача хода/улицы, раздача) ПОСЛЕ хода ---
                    if not current_rollout_state.is_round_over():
                        advanced_state = current_rollout_state.advance_state()
                        current_rollout_state = advanced_state # advance_state возвращает копию или себя

                except Exception as e_action:
                     logger.error(f"Error during action processing for P{player_to_act_rollout} in static rollout: {e_action}", exc_info=True)
                     # Применяем фол в случае любой ошибки во время хода
                     try:
                          hand_to_foul = current_rollout_state.get_player_hand(player_to_act_rollout) or []
                          current_rollout_state = current_rollout_state.apply_fantasyland_foul(player_to_act_rollout, hand_to_foul) # Используем foul для универсальности
                     except Exception as e_foul:
                          logger.error(f"Error applying foul after action error for P{player_to_act_rollout}: {e_foul}", exc_info=True)
                          break # Прерываем симуляцию, если даже фол не применился
                     # Пытаемся продвинуть состояние после фола
                     if not current_rollout_state.is_round_over():
                          try: current_rollout_state = current_rollout_state.advance_state()
                          except Exception: pass # Игнорируем ошибку advance после фола

            else: # player_to_act_rollout == -1 (никто не может ходить)
                try:
                    advanced_state = current_rollout_state.advance_state()
                    if advanced_state == current_rollout_state:
                        # logger.debug("Rollout state did not change after advance_state (waiting).")
                        break # Раунд застрял или закончился
                    current_rollout_state = advanced_state
                except Exception as e_adv:
                    logger.error(f"Error advancing state (no player) in static rollout: {e_adv}", exc_info=True)
                    break

        # --- Конец симуляции ---
        if steps >= MAX_ROLLOUT_STEPS:
            logger.warning(f"Static rollout reached MAX_ROLLOUT_STEPS ({MAX_ROLLOUT_STEPS}).")

        final_score_p0 = 0.0
        if current_rollout_state.is_round_over():
            try:
                final_score_p0 = current_rollout_state.get_terminal_score()
            except Exception as e_score:
                 logger.error(f"Error getting terminal score in static rollout: {e_score}", exc_info=True)
                 final_score_p0 = 0.0 # Считаем 0 при ошибке
        else:
            logger.warning(f"Static rollout ended prematurely (Steps: {steps}, Round Over: {current_rollout_state.is_round_over()}). Returning score 0.")

        # Возвращаем награду с точки зрения указанного игрока
        reward = float(final_score_p0) if perspective_player == 0 else float(-final_score_p0)
        return reward, simulation_actions_set


    def rollout(self, perspective_player: int = 0) -> Tuple[float, Set[Any]]:
        """ Проводит симуляцию (rollout) до конца раунда (используя статический метод). """
        # Этот метод теперь просто вызывает статическую версию
        return MCTSNode.static_rollout_simulation(self.game_state, perspective_player)

    @staticmethod
    def _static_random_rollout_policy(actions: List[Any]) -> Optional[Any]:
        """ Статическая случайная политика для роллаутов. """
        return random.choice(actions) if actions else None

    @staticmethod
    def _static_heuristic_rollout_policy(state: GameState, player_idx: int, actions: List[Any]) -> Optional[Any]:
        """ Статическая эвристическая политика для роллаутов. """
        if not actions: return None
        # TODO: Реализовать более осмысленную эвристику, если требуется
        # Например, пытаться собрать пары/сеты, избегать фола и т.д.
        # Пока используем случайную политику.
        return MCTSNode._static_random_rollout_policy(actions)

    @staticmethod
    def _static_heuristic_fantasyland_placement(hand: List[int]) -> Tuple[Optional[Dict[str, List[int]]], Optional[List[int]]]:
        """ Статическая эвристика для размещения Фантазии в роллаутах. """
        try:
            n_cards = len(hand)
            n_place = 13
            if not (14 <= n_cards <= 17): return None, None

            n_discard = n_cards - n_place
            # Сбрасываем самые младшие карты
            sorted_hand = sorted(hand, key=lambda c: CardUtils.get_rank_int(c))
            discarded = sorted_hand[:n_discard]
            remaining = sorted_hand[n_discard:]

            if len(remaining) != 13: return None, discarded

            # Простое безопасное размещение
            sorted_rem = sorted(remaining, key=lambda c: CardUtils.get_rank_int(c), reverse=True)
            placement = {
                'bottom': sorted_rem[0:5],
                'middle': sorted_rem[5:10],
                'top': sorted_rem[10:13]
            }
            # Проверяем на фол
            if check_board_foul(placement['top'], placement['middle'], placement['bottom']):
                 placement_swapped = {
                      'bottom': placement['middle'],
                      'middle': placement['bottom'],
                      'top': placement['top']
                 }
                 if check_board_foul(placement_swapped['top'], placement_swapped['middle'], placement_swapped['bottom']):
                      logger.debug("Static FL heuristic: Both simple placements result in foul.")
                      return None, discarded # Обе попытки - фол
                 else:
                      logger.debug("Static FL heuristic: Swapped middle/bottom to avoid foul.")
                      return placement_swapped, discarded
            else:
                return placement, discarded
        except Exception as e:
            logger.error(f"Error in static heuristic FL placement: {e}", exc_info=True)
            default_discard = hand[13:] if len(hand) > 13 else []
            return None, default_discard

    # (Методы get_q_value, get_rave_q_value, uct_select_child, __repr__ без изменений)
    def get_q_value(self, perspective_player: int) -> float:
        """Возвращает Q-значение узла с точки зрения указанного игрока."""
        if self.visits == 0:
            return 0.0
        # Q-value всегда хранится с точки зрения P0
        raw_q = self.total_reward / self.visits
        # Определяем, чей ход привел к этому состоянию (кто ходил в родителе)
        player_who_acted = self.parent._get_player_to_move() if self.parent else -1

        # Если ход делал игрок, с чьей точки зрения смотрим, возвращаем Q как есть
        if player_who_acted == perspective_player:
            return raw_q
        # Если ход делал оппонент, инвертируем Q
        elif player_who_acted != -1:
            return -raw_q
        else:
            # Если это корневой узел (нет родителя), возвращаем Q для P0 или инвертируем для P1
            return raw_q if perspective_player == 0 else -raw_q

    def get_rave_q_value(self, action: Any, perspective_player: int) -> float:
        """Возвращает RAVE Q-значение для действия с точки зрения указанного игрока."""
        # Проверяем хешируемость действия перед доступом к словарю
        try: hash(action)
        except TypeError: return 0.0 # Нехешируемое действие не имеет RAVE

        rave_visits = self.rave_visits.get(action, 0)
        if rave_visits == 0:
            return 0.0
        # RAVE Q-value всегда хранится с точки зрения P0
        rave_reward = self.rave_total_reward.get(action, 0.0)
        raw_rave_q = rave_reward / rave_visits
        # Определяем, чей ход в ТЕКУЩЕМ узле (кто будет делать это действие)
        player_to_move_in_current_node = self._get_player_to_move()

        if player_to_move_in_current_node == -1:
            return 0.0 # Не должно происходить, если есть действия
        # Если ход делает игрок, с чьей точки зрения смотрим, возвращаем RAVE Q как есть
        return raw_rave_q if player_to_move_in_current_node == perspective_player else -raw_rave_q

    def uct_select_child(self, exploration_constant: float, rave_k: float) -> Optional['MCTSNode']:
        """Выбирает дочерний узел с использованием формулы UCT (с RAVE)."""
        best_score = -float('inf')
        best_child = None
        current_player_perspective = self._get_player_to_move()

        if current_player_perspective == -1:
            return None # Нельзя выбрать потомка из терминального/ожидающего узла

        # Используем логарифм от посещений родителя + 1 для избежания log(0)
        parent_visits_log = math.log(self.visits + 1)
        children_items = list(self.children.items())
        if not children_items:
             return None

        random.shuffle(children_items) # Для случайного выбора при равных очках

        for action, child in children_items:
            child_visits = child.visits
            rave_visits = 0
            score = -float('inf') # Начальное значение
            is_hashable = True
            try: hash(action)
            except TypeError: is_hashable = False

            if is_hashable:
                rave_visits = self.rave_visits.get(action, 0)

            # --- UCB1 + RAVE ---
            if child_visits == 0:
                # Если узел не посещался, используем RAVE для оценки или даем высокий приоритет
                if rave_visits > 0 and rave_k > 0 and is_hashable:
                    rave_q = self.get_rave_q_value(action, current_player_perspective)
                    # Используем только RAVE Q + небольшой бонус за исследование RAVE
                    score = rave_q + exploration_constant * math.sqrt(math.log(rave_visits + 1) / (rave_visits + 1e-6))
                else:
                    score = float('inf') # Не посещался ни разу - максимальный приоритет
            else:
                # Стандартный UCB1
                q_child = child.get_q_value(current_player_perspective)
                exploit_term = q_child
                explore_term = exploration_constant * math.sqrt(parent_visits_log / child_visits)
                ucb1_score = exploit_term + explore_term

                # Комбинируем с RAVE, если он доступен
                if rave_visits > 0 and rave_k > 0 and is_hashable:
                    rave_q = self.get_rave_q_value(action, current_player_perspective)
                    # Формула с параметром k для баланса RAVE и UCB1
                    beta = math.sqrt(rave_k / (3 * self.visits + rave_k)) if self.visits > 0 else 1.0
                    score = (1.0 - beta) * ucb1_score + beta * rave_q
                else:
                    score = ucb1_score # Используем только UCB1

            if score > best_score:
                best_score = score
                best_child = child
            # --- ИСПРАВЛЕНО: Случайный выбор при РАВНЫХ очках (кроме inf) ---
            elif score == best_score and score != float('inf') and score != -float('inf'):
                 if random.choice([True, False]):
                      best_child = child

        # Если все потомки имеют score = -inf (например, все проигрышные), выбираем случайно
        if best_child is None and children_items:
             best_child = random.choice([child for _, child in children_items])

        return best_child

    def _backpropagate_parallel(self, path: List['MCTSNode'], total_reward: float, num_rollouts: int, simulation_actions: Set[Any]):
        """ Обновляет статистику узлов вдоль пути с учетом RAVE. """
        if num_rollouts <= 0: return
        # Награда всегда передается с точки зрения P0
        reward_p0 = total_reward / num_rollouts

        for node in reversed(path):
            node.visits += num_rollouts
            # Обновляем total_reward (всегда с точки зрения P0)
            node.total_reward += total_reward

            # Обновляем RAVE статистику для действий, ведущих к этому узлу
            if node.parent and node.action is not None:
                 parent = node.parent
                 action_taken = node.action
                 try:
                      hash(action_taken) # Проверяем хешируемость
                      if action_taken not in parent.rave_visits:
                           parent.rave_visits[action_taken] = 0
                           parent.rave_total_reward[action_taken] = 0.0
                      parent.rave_visits[action_taken] += num_rollouts
                      # Добавляем награду с точки зрения P0
                      parent.rave_total_reward[action_taken] += total_reward
                 except TypeError: pass # Игнорируем нехешируемые

    def __repr__(self):
        """Строковое представление узла для отладки."""
        player_idx = self._get_player_to_move()
        player = f'P{player_idx}' if player_idx != -1 else 'T' # T for Terminal
        q_val_p0 = self.get_q_value(0) # Q-value с точки зрения P0
        action_repr = "Root"
        if self.action:
            try:
                # Попробуем использовать _format_action из агента, если доступно
                from src.mcts_agent import MCTSAgent # Локальный импорт
                action_str = MCTSAgent._format_action(None, self.action) # Передаем None как self
            except Exception:
                try:
                    action_str = str(self.action)
                except Exception:
                    action_str = "???"
            action_repr = (action_str[:35] + '...') if len(action_str) > 38 else action_str

        return (
            f"[{player} Act:{action_repr} V={self.visits} Q0={q_val_p0:.3f} "
            f"NChild={len(self.children)} UAct={len(self.untried_actions or [])}]"
        )
