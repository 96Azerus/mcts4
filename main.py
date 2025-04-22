# main.py v1.1
"""
Пример использования модулей игры OFC Pineapple для запуска
простого сценария в консоли (например, для отладки).
Использует GameState.advance_state() для управления ходом игры.
"""
import time
import random
import os
import sys
import traceback
import logging # Используем logging
from typing import List, Optional, Tuple, Any, Dict

# Импорты из src пакета
try:
    from src.card import card_from_str, card_to_str, Card as CardUtils, INVALID_CARD
    from src.game_state import GameState
    from src.mcts_agent import MCTSAgent
    from src.fantasyland_solver import FantasylandSolver
    from src.scoring import check_board_foul
    from src.board import PlayerBoard
except ImportError as e:
    print(f"Ошибка импорта в main.py: {e}", file=sys.stderr)
    print("Убедитесь, что вы запускаете скрипт из корневой директории проекта,", file=sys.stderr)
    print("или что директория src доступна в PYTHONPATH.", file=sys.stderr)
    sys.exit(1)

# Настройка логирования для main
logger = logging.getLogger('main_scenario')
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# --- Функции ввода пользователя (остаются как пример, не используются) ---
def get_human_action_street1(hand: List[int], board: 'PlayerBoard') -> Tuple[List[Tuple[int, str, int]], List[int]]:
    print("\nВаша рука (Улица 1):", ", ".join(card_to_str(c) for c in hand))
    print("Ваша доска:")
    print(board)
    placements = []
    available_slots = board.get_available_slots()
    if len(available_slots) >= 5:
         # Простое авто-размещение для примера
         for i in range(5):
              card = hand[i]
              row, idx = available_slots[i]
              placements.append((card, row, idx))
         print(f"[Пример] Авто-размещение улицы 1: {placements}")
         return tuple(sorted(placements)), tuple() # Возвращаем кортежи
    raise ValueError("Not enough slots for street 1")


def get_human_action_pineapple(hand: List[int], board: 'PlayerBoard') -> Optional[Tuple[Tuple[int, str, int], Tuple[int, str, int], int]]:
    print("\nВаша рука (Улицы 2-5):", ", ".join(f"{i+1}:{card_to_str(c)}" for i, c in enumerate(hand)))
    print("Ваша доска:")
    print(board)
    available_slots = board.get_available_slots()
    if len(available_slots) >= 2 and len(hand) == 3:
         # Простое авто-размещение для примера
         discard_card = hand[2]
         card1, card2 = hand[0], hand[1]
         slot1, slot2 = available_slots[0], available_slots[1]
         p1 = (card1, slot1[0], slot1[1])
         p2 = (card2, slot2[0], slot2[1])
         action = tuple(sorted((p1, p2))) + (discard_card,)
         print(f"[Пример] Авто-ход Pineapple: {action}")
         return action # type: ignore
    return None

def get_human_fantasyland_placement(hand: List[int]) -> Tuple[Optional[Dict[str, List[int]]], Optional[List[int]]]:
     print("\n--- FANTASYLAND ---")
     print("Ваша рука:", ", ".join(card_to_str(c) for c in hand))
     solver = FantasylandSolver()
     placement, discarded = solver.solve(hand)
     if placement:
          print("[Пример] Авто-размещение Фантазии (солвер).")
          return placement, discarded
     else:
          print("[Пример] Не удалось разместить Фантазию (фол).")
          # Возвращаем None и карты для сброса (если есть)
          return None, hand[13:] if len(hand) > 13 else []


# --- Упрощенный основной сценарий ---
def run_simple_scenario():
    """Запускает простой сценарий игры AI vs AI для демонстрации."""
    logger.info("--- Запуск упрощенного сценария (AI vs AI) ---")

    try:
        mcts_time_limit = int(os.environ.get('MCTS_TIME_LIMIT_MS', 1000))
        mcts_workers = int(os.environ.get('NUM_WORKERS', 1))
        ai_agent = MCTSAgent(time_limit_ms=mcts_time_limit, num_workers=mcts_workers)
    except Exception as e:
        logger.error(f"Ошибка инициализации AI: {e}", exc_info=True)
        return

    dealer_idx = random.choice([0, 1])
    game_state = GameState(dealer_idx=dealer_idx)
    game_state.start_new_round(dealer_idx)

    logger.info(f"Начало раунда (Дилер: {dealer_idx})")

    max_steps = 60 # Увеличим лимит шагов для полного раунда
    for step in range(max_steps):
        logger.info(f"\n--- Шаг {step+1} ---")
        logger.info(f"Улица: {game_state.street}, Дилер: {game_state.dealer_idx}")
        logger.info("Доска P0:\n%s", game_state.boards[0])
        logger.info("Доска P1:\n%s", game_state.boards[1])
        if game_state.current_hands.get(0): logger.info("Рука P0: %s", [card_to_str(c) for c in game_state.current_hands[0]])
        if game_state.current_hands.get(1): logger.info("Рука P1: %s", [card_to_str(c) for c in game_state.current_hands[1]])
        if game_state.fantasyland_hands[0]: logger.info("ФЛ Рука P0: %d карт", len(game_state.fantasyland_hands[0]))
        if game_state.fantasyland_hands[1]: logger.info("ФЛ Рука P1: %d карт", len(game_state.fantasyland_hands[1]))

        if game_state.is_round_over():
            logger.info("\n--- Раунд завершен ---")
            break

        # --- Используем get_player_to_move и advance_state ---
        player_to_act = game_state.get_player_to_move()
        logger.info(f"Ожидаемый ход: Игрок {player_to_act if player_to_act != -1 else 'Ожидание'}")

        if player_to_act != -1:
            logger.info(f"Ход Игрока {player_to_act}...")
            action: Optional[Any] = None
            next_state: Optional[GameState] = None
            try:
                action = ai_agent.choose_action(game_state)
                action_str = ai_agent._format_action(action) # Получаем строку до применения

                if action is None:
                    logger.warning(f"AI Игрок {player_to_act} не смог выбрать ход. Применяем фол.")
                    hand_to_foul = game_state.get_player_hand(player_to_act) or []
                    next_state = game_state.apply_fantasyland_foul(player_to_act, hand_to_foul) # Универсальный фол
                elif isinstance(action, tuple) and action[0] == "FANTASYLAND_PLACEMENT":
                    _, placement, discarded = action
                    next_state = game_state.apply_fantasyland_placement(player_to_act, placement, discarded)
                    logger.info(f"Игрок {player_to_act} разместил Фантазию.")
                elif isinstance(action, tuple) and action[0] == "FANTASYLAND_FOUL":
                    _, hand_to_discard = action
                    next_state = game_state.apply_fantasyland_foul(player_to_act, hand_to_discard)
                    logger.warning(f"Игрок {player_to_act} сфолил в Фантазии (по решению AI).")
                else:
                    next_state = game_state.apply_action(player_to_act, action)
                    logger.info(f"Игрок {player_to_act} применил действие: {action_str}")

                # Обновляем состояние только если ход был успешным
                game_state = next_state

            except Exception as e_act:
                 logger.error(f"Ошибка во время хода AI игрока {player_to_act}: {e_act}", exc_info=True)
                 # Пытаемся применить фол
                 try:
                      hand_to_foul = game_state.get_player_hand(player_to_act) or []
                      game_state = game_state.apply_fantasyland_foul(player_to_act, hand_to_foul)
                      logger.warning(f"Применен фол для игрока {player_to_act} из-за ошибки.")
                 except Exception as e_foul:
                      logger.error(f"Критическая ошибка: не удалось применить фол для игрока {player_to_act}: {e_foul}", exc_info=True)
                      break # Прерываем сценарий

            # Продвигаем состояние ПОСЛЕ хода (или фола)
            if not game_state.is_round_over():
                 try:
                      game_state = game_state.advance_state()
                 except Exception as e_adv:
                      logger.error(f"Ошибка при продвижении состояния после хода P{player_to_act}: {e_adv}", exc_info=True)
                      break

        else: # player_to_act == -1 (Ожидание)
             logger.info("Никто не может ходить. Продвигаем состояние для раздачи...")
             try:
                 advanced_state = game_state.advance_state()
                 if advanced_state == game_state:
                      logger.warning("Состояние не изменилось после advance_state (ожидание). Возможно, конец раунда или нет карт.")
                      break # Выходим из цикла, если продвижение не помогло
                 game_state = advanced_state
             except Exception as e_adv_wait:
                  logger.error(f"Ошибка при продвижении состояния (ожидание): {e_adv_wait}", exc_info=True)
                  break

        time.sleep(0.05) # Небольшая пауза для читаемости вывода

    # Вывод финального состояния
    logger.info("\n--- Финальные доски ---")
    logger.info("Игрок 0:\n%s", game_state.boards[0])
    logger.info("Игрок 1:\n%s", game_state.boards[1])
    if game_state.is_round_over():
        try:
            score_diff = game_state.get_terminal_score()
            logger.info(f"Счет за раунд (P0 vs P1): {score_diff}")
        except Exception as e_score:
            logger.error(f"Ошибка получения финального счета: {e_score}", exc_info=True)
    else:
        logger.warning(f"\n--- Раунд не завершен после {max_steps} шагов ---")

if __name__ == "__main__":
    # Настройка кодировки для Windows (без изменений)
    if sys.platform == "win32":
        try:
            if sys.stdout.encoding != 'utf-8': sys.stdout.reconfigure(encoding='utf-8')
            if sys.stderr.encoding != 'utf-8': sys.stderr.reconfigure(encoding='utf-8')
        except Exception as e: print(f"Warning: Could not set console encoding: {e}")

    run_simple_scenario()
