# src/mcts_agent.py v1.1
"""
Реализация MCTS-агента для игры OFC Pineapple.
Поддерживает RAVE, параллелизацию и Progressive Fantasyland.
"""

import time
import random
import multiprocessing
import traceback
import sys
import logging # Используем logging
from typing import Optional, Any, List, Tuple, Set, Dict

# Импорты из src пакета
from src.mcts_node import MCTSNode, run_parallel_rollout # Импортируем воркер
from src.game_state import GameState
from src.fantasyland_solver import FantasylandSolver
from src.card import card_to_str, Card as CardUtils # Используем алиас для утилит

# Получаем логгер
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): logger.setLevel(logging.WARNING)


class MCTSAgent:
    """ Агент MCTS для OFC Pineapple. """
    DEFAULT_EXPLORATION: float = 1.414
    DEFAULT_RAVE_K: float = 500
    DEFAULT_TIME_LIMIT_MS: int = 5000
    DEFAULT_NUM_WORKERS: int = max(1, multiprocessing.cpu_count() - 1 if multiprocessing.cpu_count() > 1 else 1)
    DEFAULT_ROLLOUTS_PER_LEAF: int = 4

    def __init__(self,
                 exploration: Optional[float] = None,
                 rave_k: Optional[float] = None,
                 time_limit_ms: Optional[int] = None,
                 num_workers: Optional[int] = None,
                 rollouts_per_leaf: Optional[int] = None):
        """ Инициализирует MCTS-агента. """
        self.exploration: float = exploration if exploration is not None else self.DEFAULT_EXPLORATION
        self.rave_k: float = rave_k if rave_k is not None else self.DEFAULT_RAVE_K
        time_limit_val: int = time_limit_ms if time_limit_ms is not None else self.DEFAULT_TIME_LIMIT_MS
        self.time_limit: float = time_limit_val / 1000.0
        max_cpus = multiprocessing.cpu_count()
        requested_workers: int = num_workers if num_workers is not None else self.DEFAULT_NUM_WORKERS
        # Ограничиваем максимальное количество воркеров, чтобы избежать проблем
        self.num_workers: int = max(1, min(requested_workers, max_cpus, 8)) # Ограничим 8 воркерами
        self.rollouts_per_leaf: int = rollouts_per_leaf if rollouts_per_leaf is not None else self.DEFAULT_ROLLOUTS_PER_LEAF
        if self.num_workers == 1 and self.rollouts_per_leaf > 1:
            logger.warning(f"num_workers=1, reducing rollouts_per_leaf from {self.rollouts_per_leaf} to 1.")
            self.rollouts_per_leaf = 1
        self.fantasyland_solver = FantasylandSolver()
        logger.info(f"MCTS Agent initialized: TimeLimit={self.time_limit:.2f}s, Exploration={self.exploration}, "
                    f"RaveK={self.rave_k}, Workers={self.num_workers}, RolloutsPerLeaf={self.rollouts_per_leaf}")
        try:
            # Устанавливаем метод старта 'spawn' для лучшей совместимости
            current_method = multiprocessing.get_start_method(allow_none=True)
            if current_method != 'spawn':
                multiprocessing.set_start_method('spawn', force=True)
                logger.info(f"Multiprocessing start method set to 'spawn'.")
        except Exception as e:
            logger.warning(f"Could not set multiprocessing start method to 'spawn': {e}. Using default ({multiprocessing.get_start_method()}).")


    def choose_action(self, game_state: GameState) -> Optional[Any]:
        """ Выбирает лучшее действие с помощью MCTS. """
        start_time_total = time.time()
        if game_state is None:
             logger.error("AI Agent Error: choose_action called with None game_state.")
             return None

        logger.info(f"\n--- AI Agent: Choosing action (Street {game_state.street}) ---")

        player_to_act = game_state.get_player_to_move() # Используем метод GameState

        if player_to_act == -1:
            logger.error("AI Agent Error: No player can act according to game_state. Returning None.")
            return None

        logger.info(f"Player to act: {player_to_act}")

        # --- Обработка Fantasyland ---
        if game_state.is_fantasyland_round and game_state.fantasyland_status[player_to_act]:
            hand = game_state.fantasyland_hands[player_to_act]
            if hand:
                logger.info(f"Player {player_to_act} is in Fantasyland. Solving...")
                start_fl_time = time.time()
                try:
                    placement, discarded = self.fantasyland_solver.solve(hand)
                except Exception as e_fl_solve:
                     logger.error(f"Error during Fantasyland solve: {e_fl_solve}", exc_info=True)
                     placement, discarded = None, hand[13:] if len(hand) > 13 else [] # Фол при ошибке солвера

                solve_time = time.time() - start_fl_time
                logger.info(f"Fantasyland solved in {solve_time:.3f}s")
                if placement and discarded is not None:
                     logger.info("Fantasyland solver returned valid placement.")
                     return ("FANTASYLAND_PLACEMENT", placement, discarded)
                else:
                     logger.warning("Fantasyland solver failed or returned foul. Returning foul action.")
                     # Возвращаем действие фола с картами, которые нужно сбросить (если есть)
                     return ("FANTASYLAND_FOUL", hand)
            else:
                 logger.error(f"AI Agent Error: FL player {player_to_act} has no hand.")
                 return None # Не можем ходить без руки

        # --- Обычный ход (MCTS) ---
        logger.info(f"Starting MCTS for player {player_to_act}...")
        try:
            initial_actions = game_state.get_legal_actions_for_player(player_to_act)
        except Exception as e_legal:
             logger.error(f"Error getting legal actions for P{player_to_act}: {e_legal}", exc_info=True)
             return None # Не можем продолжить без действий

        logger.info(f"Found {len(initial_actions)} legal actions initially.")

        if not initial_actions:
             logger.warning(f"AI Agent Warning: No legal actions found for player {player_to_act}.")
             return None
        if len(initial_actions) == 1:
             logger.info("Only one legal action found. Returning it.")
             return initial_actions[0]

        root_node = MCTSNode(game_state)
        # Инициализируем untried_actions и RAVE статистику
        root_node.untried_actions = list(initial_actions)
        random.shuffle(root_node.untried_actions)
        for act in root_node.untried_actions:
             try:
                  hash(act) # Проверяем, можно ли хешировать действие
                  if act not in root_node.rave_visits:
                       root_node.rave_visits[act] = 0
                       root_node.rave_total_reward[act] = 0.0
             except TypeError:
                  logger.warning(f"Action {self._format_action(act)} is not hashable and cannot be used in RAVE.")


        start_mcts_time = time.time(); num_simulations = 0
        pool = None # Инициализируем pool как None
        try:
            # Создаем пул процессов только если воркеров больше 1
            if self.num_workers > 1:
                 pool = multiprocessing.Pool(processes=self.num_workers)
                 logger.debug(f"Created multiprocessing pool with {self.num_workers} workers.")

            while time.time() - start_mcts_time < self.time_limit:
                path, leaf_node = self._select(root_node)
                if leaf_node is None:
                     logger.warning("Selection phase returned None leaf node. Breaking MCTS loop.")
                     break # Прерываем, если выбор не удался

                node_to_rollout_from = leaf_node
                expanded_node = None

                # Пытаемся расширить узел, если он не терминальный
                if not leaf_node.is_terminal():
                    if leaf_node.untried_actions: # Расширяем, только если есть неиспробованные действия
                        expanded_node = leaf_node.expand()
                        if expanded_node:
                             node_to_rollout_from = expanded_node
                             path.append(expanded_node)
                        # else: logger.debug("Expansion returned None.") # Если expand не удался
                    # else: logger.debug("Leaf node has no untried actions.")

                # Выполняем роллауты
                results: List[float] = []
                simulation_actions_aggregated: Set[Any] = set()
                if not node_to_rollout_from.is_terminal():
                    try:
                        node_state_dict = node_to_rollout_from.game_state.to_dict()
                        if "error" in node_state_dict: raise RuntimeError("Serialization failed")

                        rollout_tasks = [node_state_dict] * self.rollouts_per_leaf
                        if pool: # Используем пул, если он есть
                             async_results = [pool.apply_async(run_parallel_rollout, (task,)) for task in rollout_tasks]
                             for res in async_results:
                                  try:
                                       # Увеличиваем таймаут для get, чтобы избежать ложных срабатываний
                                       timeout_get = max(10.0, self.time_limit * 0.5)
                                       reward, sim_actions = res.get(timeout=timeout_get)
                                       results.append(reward)
                                       simulation_actions_aggregated.update(sim_actions)
                                       num_simulations += 1
                                  except multiprocessing.TimeoutError: logger.warning("Rollout worker timed out.")
                                  except Exception as e_get: logger.warning(f"Error getting result from worker: {e_get}")
                        else: # Выполняем последовательно, если пул не создан
                             for task in rollout_tasks:
                                  try:
                                       reward, sim_actions = run_parallel_rollout(task)
                                       results.append(reward)
                                       simulation_actions_aggregated.update(sim_actions)
                                       num_simulations += 1
                                  except Exception as e_seq: logger.warning(f"Error during sequential rollout: {e_seq}")

                    except Exception as e_roll:
                         logger.error(f"Error during rollout phase: {e_roll}", exc_info=True)
                         continue # Пропускаем итерацию при ошибке роллаута
                else:
                     # Терминальный узел - получаем счет напрямую
                     try:
                          reward = node_to_rollout_from.game_state.get_terminal_score()
                          results.append(reward)
                          num_simulations += 1
                     except Exception as e_term:
                          logger.error(f"Error getting terminal score: {e_term}", exc_info=True)


                # Обратное распространение
                if results:
                    total_reward = sum(results)
                    num_rollouts_done = len(results)
                    # Добавляем действие, которое привело к расширенному узлу (если он был)
                    if expanded_node and expanded_node.action:
                         try: hash(expanded_node.action); simulation_actions_aggregated.add(expanded_node.action)
                         except TypeError: pass
                    # Вызываем backpropagate
                    self._backpropagate_parallel(path, total_reward, num_rollouts_done, simulation_actions_aggregated)

        except Exception as e_mcts:
            logger.error(f"Critical error during MCTS execution: {e_mcts}", exc_info=True)
            # В случае критической ошибки выбираем случайное действие
            logger.warning("Choosing random action due to MCTS error.")
            return random.choice(initial_actions) if initial_actions else None
        finally:
            # Гарантированно закрываем пул процессов
            if pool:
                 try: pool.close(); pool.join()
                 except Exception as e_pool: logger.error(f"Error closing MCTS pool: {e_pool}")

        elapsed_time = time.time() - start_mcts_time
        sims_per_sec = (num_simulations / elapsed_time) if elapsed_time > 0 else 0
        logger.info(f"MCTS finished: Ran {num_simulations} simulations in {elapsed_time:.3f}s ({sims_per_sec:.1f} sims/s).")

        best_action = self._select_best_action(root_node, initial_actions)
        total_time = time.time() - start_time_total
        logger.info(f"--- AI Agent: Action chosen in {total_time:.3f}s ---")
        return best_action

    def _select(self, node: MCTSNode) -> Tuple[List[MCTSNode], Optional[MCTSNode]]:
        """ Фаза выбора: спускаемся по дереву, выбирая лучшие узлы по UCT. """
        path = [node]
        current_node = node
        while True:
            if current_node.is_terminal():
                return path, current_node # Достигли терминального узла

            player_to_move = current_node._get_player_to_move()
            if player_to_move == -1:
                return path, current_node # Достигли узла ожидания

            # Инициализируем неиспробованные действия, если нужно
            if current_node.untried_actions is None:
                try:
                    current_node.untried_actions = current_node.game_state.get_legal_actions_for_player(player_to_move)
                    random.shuffle(current_node.untried_actions)
                    # Инициализация RAVE
                    for act in current_node.untried_actions:
                         try:
                              hash(act)
                              if act not in current_node.rave_visits:
                                   current_node.rave_visits[act] = 0
                                   current_node.rave_total_reward[act] = 0.0
                         except TypeError: pass
                except Exception as e_leg_sel:
                     logger.error(f"Error getting legal actions during select for P{player_to_move}: {e_leg_sel}", exc_info=True)
                     current_node.untried_actions = []

            # Если есть неиспробованные действия, выбираем этот узел для расширения
            if current_node.untried_actions:
                return path, current_node

            # Если нет неиспробованных действий и нет потомков, это лист
            if not current_node.children:
                # logger.debug(f"Node has no children and no untried actions: {current_node}")
                return path, current_node

            # Выбираем лучшего потомка по UCT
            selected_child = current_node.uct_select_child(self.exploration, self.rave_k)

            if selected_child is None:
                # Это может произойти, если все потомки имеют -inf score
                logger.warning(f"uct_select_child returned None for node {current_node}. Parent visits: {current_node.visits}. Children count: {len(current_node.children)}")
                # Пытаемся выбрать случайного потомка, если они есть
                if current_node.children:
                     try: selected_child = random.choice(list(current_node.children.values()))
                     except IndexError: return path, current_node # Нет потомков
                else: return path, current_node # Нет потомков

            current_node = selected_child
            path.append(current_node)


    def _backpropagate_parallel(self, path: List[MCTSNode], total_reward: float, num_rollouts: int, simulation_actions: Set[Any]):
        """ Фаза обратного распространения (с RAVE). """
        if num_rollouts <= 0: return
        # Награда всегда передается с точки зрения P0
        # total_reward - это суммарная награда за num_rollouts симуляций

        for node in reversed(path):
            node.visits += num_rollouts
            # Обновляем total_reward (всегда с точки зрения P0)
            node.total_reward += total_reward

            # Обновляем RAVE статистику для действий, которые были сыграны в симуляциях
            # и являются возможными действиями из текущего узла node
            possible_actions_from_node: Set[Any] = set()
            if node.children: possible_actions_from_node.update(node.children.keys())
            if node.untried_actions: possible_actions_from_node.update(node.untried_actions)

            # Находим пересечение сыгранных действий и возможных действий
            relevant_sim_actions = simulation_actions.intersection(possible_actions_from_node)

            for action in relevant_sim_actions:
                 try: # Проверка хешируемости
                      hash(action)
                      if action not in node.rave_visits:
                           node.rave_visits[action] = 0
                           node.rave_total_reward[action] = 0.0
                      node.rave_visits[action] += num_rollouts
                      # Добавляем награду с точки зрения P0
                      node.rave_total_reward[action] += total_reward
                 except TypeError: pass # Игнорируем нехешируемые для RAVE


    def _select_best_action(self, root_node: MCTSNode, initial_actions: List[Any]) -> Optional[Any]:
        """ Выбирает лучшее действие из корневого узла (максимум посещений). """
        if not root_node.children:
            logger.warning("No children found at root node. Choosing random action.")
            return random.choice(initial_actions) if initial_actions else None

        best_action = None
        max_visits = -1
        # Перемешиваем для случайного выбора при равных посещениях
        items = list(root_node.children.items())
        random.shuffle(items)

        logger.info(f"--- Evaluating {len(items)} child nodes ---")
        for action, child_node in items:
             q_val_p0 = child_node.get_q_value(0) # Q с точки зрения P0
             rave_q_p0 = 0.0
             rave_v = 0
             try: # Проверка хешируемости для RAVE
                  hash(action)
                  rave_v = root_node.rave_visits.get(action, 0)
                  if rave_v > 0:
                       # RAVE Q с точки зрения игрока, который будет делать ход (root_node._get_player_to_move())
                       rave_q_p0 = root_node.get_rave_q_value(action, 0) # Получаем RAVE Q для P0
             except TypeError: pass

             logger.info(f"  Action: {self._format_action(action):<45} Visits: {child_node.visits:<6} Q(P0): {q_val_p0:<8.3f} RAVE_Q(P0): {rave_q_p0:<8.3f} RAVE_V: {rave_v:<5}")

             if child_node.visits > max_visits:
                 max_visits = child_node.visits
                 best_action = action

        if best_action is None:
             logger.warning("Could not determine best action based on visits. Choosing first shuffled.")
             best_action = items[0][0] if items else (random.choice(initial_actions) if initial_actions else None)

        if best_action:
             logger.info(f"Selected action (max visits = {max_visits}): {self._format_action(best_action)}")
        else:
             logger.error("Error: Failed to select any action.")
        return best_action

    def _format_action(self, action: Any) -> str:
        """ Форматирует действие в читаемую строку для логов. """
        # --- ИСПРАВЛЕНО: Более надежное форматирование ---
        if action is None: return "None"
        try:
            # Pineapple Action: ((c1, r1, i1), (c2, r2, i2), discard)
            if isinstance(action, tuple) and len(action) == 3 \
               and isinstance(action[0], tuple) and len(action[0]) == 3 and isinstance(action[0][0], int) \
               and isinstance(action[1], tuple) and len(action[1]) == 3 and isinstance(action[1][0], int) \
               and isinstance(action[2], int):
                p1, p2, d = action
                return f"PINEAPPLE: {card_to_str(p1[0])}@{p1[1]}{p1[2]}, {card_to_str(p2[0])}@{p2[1]}{p2[2]}; DISC={card_to_str(d)}"

            # Street 1 Action: (tuple_of_placements, empty_tuple)
            elif isinstance(action, tuple) and len(action) == 2 and isinstance(action[0], tuple) and action[1] == tuple():
                 # Проверяем первый элемент placements для типа
                 if action[0] and isinstance(action[0][0], tuple) and len(action[0][0]) == 3 and isinstance(action[0][0][0], int):
                      placements_str = ", ".join([f"{card_to_str(c)}@{r}{i}" for c, r, i in action[0]])
                      return f"STREET 1: Place [{placements_str}]"
                 else:
                      # Если placements пустой или имеет неверный формат
                      return f"STREET 1: Invalid placements tuple: {action[0]}"

            # Fantasyland Actions
            elif isinstance(action, tuple) and len(action) > 0:
                 if action[0] == "FANTASYLAND_INPUT" and len(action) == 2 and isinstance(action[1], tuple):
                      return f"FANTASYLAND_INPUT ({len(action[1])} cards)"
                 elif action[0] == "FANTASYLAND_PLACEMENT" and len(action) == 3:
                      discard_count = len(action[2]) if isinstance(action[2], list) else '?'
                      return f"FANTASYLAND_PLACE (Discard {discard_count})"
                 elif action[0] == "FANTASYLAND_FOUL" and len(action) == 2:
                      discard_count = len(action[1]) if isinstance(action[1], list) else '?'
                      return f"FANTASYLAND_FOUL (Discard {discard_count})"
                 else:
                      # Общий случай для других кортежей
                      return f"Tuple Action: {action!r}" # Используем repr для кортежей
            else:
                 # Для других типов просто возвращаем строку
                 return str(action)
        except Exception as e:
             logger.warning(f"Error formatting action ({type(action)}): {e}")
             return "ErrorFormattingAction"
