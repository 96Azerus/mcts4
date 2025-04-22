# app.py v1.1
"""
Основной файл веб-приложения Flask для игры OFC Pineapple.
Обрабатывает HTTP-запросы, управляет сессиями и взаимодействует
с игровой логикой (GameState) и AI-агентом (MCTSAgent).
Использует GameState.advance_state() для управления ходом игры.
"""

import os
import json
import random
import traceback
import sys
import logging
from typing import Optional, Dict, Any, Tuple, List

from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv

# Импорты из src пакета
try:
    from src.card import card_from_str, card_to_str, Card as CardUtils, CARD_PLACEHOLDER, INVALID_CARD
    from src.game_state import GameState
    from src.board import PlayerBoard
    from src.mcts_agent import MCTSAgent
except ImportError as e:
    print(f"FATAL ERROR: Import failed from src: {e}", file=sys.stderr)
    print("Ensure the script is run from the project root directory or PYTHONPATH is set correctly.", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
except Exception as e_global:
     print(f"FATAL ERROR during initial imports: {e_global}", file=sys.stderr)
     traceback.print_exc(file=sys.stderr)
     sys.exit(1)

# --- Константы ---
HUMAN_PLAYER_IDX: int = 0
AI_PLAYER_IDX: int = 1
FLASK_SESSION_KEY: str = 'game_state_v1.2' # Обновляем ключ при изменении структуры
MAX_AI_TURNS_IN_SEQUENCE: int = 5 # Предохранитель от зацикливания AI ходов

# --- Инициализация приложения и логирования ---
load_dotenv()
app = Flask(__name__)

# Настройка логирования (как и раньше)
# Используем логгер app.logger, который настроен Flask
app.logger.handlers.clear() # Очищаем стандартные обработчики Flask
log_level_str = os.environ.get('LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_str, logging.INFO) # Получаем уровень логирования
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s') # Добавили модуль и строку
stream_handler = logging.StreamHandler(sys.stdout) # Вывод в stdout для Render/Docker
stream_handler.setFormatter(log_formatter)
# Устанавливаем уровень для обработчика (он может быть строже, чем у логгера)
stream_handler.setLevel(log_level)
app.logger.addHandler(stream_handler)
app.logger.setLevel(log_level) # Устанавливаем уровень для самого логгера
# Уменьшаем шум от Werkzeug в логах
logging.getLogger('werkzeug').setLevel(logging.WARNING)

app.logger.info("--- Starting Flask App Initialization ---")

# Конфигурация Flask (как и раньше)
app.secret_key = os.environ.get('FLASK_SECRET_KEY')
is_production = os.environ.get('RENDER') == 'true' or os.environ.get('FLASK_ENV') == 'production'
if not app.secret_key:
    app.logger.critical("FLASK_SECRET_KEY environment variable not set.")
    if not is_production:
        app.secret_key = 'dev_secret_key_for_debug_only'
        app.logger.warning("Using temporary insecure secret key for development.")
    else:
        app.logger.critical("Exiting due to missing FLASK_SECRET_KEY in production.")
        sys.exit(1)
else:
    app.logger.info("FLASK_SECRET_KEY loaded.")
app.config['SESSION_COOKIE_SECURE'] = is_production
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Инициализация AI Агента (как и раньше)
ai_agent: Optional[MCTSAgent] = None
try:
    app.logger.info("Initializing AI Agent...")
    mcts_time_limit = int(os.environ.get('MCTS_TIME_LIMIT_MS', 5000))
    mcts_rave_k = int(os.environ.get('MCTS_RAVE_K', 500))
    default_workers = MCTSAgent.DEFAULT_NUM_WORKERS
    try:
        gunicorn_workers = int(os.environ.get('WEB_CONCURRENCY', default_workers))
    except ValueError:
        gunicorn_workers = default_workers
    mcts_workers = int(os.environ.get('NUM_WORKERS', gunicorn_workers))
    mcts_rollouts_leaf = int(os.environ.get('ROLLOUTS_PER_LEAF', MCTSAgent.DEFAULT_ROLLOUTS_PER_LEAF))
    app.logger.info(f"AI Params: TimeLimit={mcts_time_limit}ms, RaveK={mcts_rave_k}, Workers={mcts_workers}, RolloutsPerLeaf={mcts_rollouts_leaf}")
    ai_agent = MCTSAgent(time_limit_ms=mcts_time_limit, rave_k=mcts_rave_k, num_workers=mcts_workers, rollouts_per_leaf=mcts_rollouts_leaf)
    app.logger.info("AI Agent Initialized Successfully.")
except Exception as e:
    app.logger.critical(f"AI Agent initialization failed: {e}", exc_info=True)
    if is_production:
        sys.exit(1)
    ai_agent = None
if ai_agent is None:
    app.logger.warning("AI Agent is None. AI turns will result in fouls.")

# --- Функции для работы с состоянием в сессии (без изменений) ---
def save_game_state(state: Optional[GameState]):
    """Сохраняет состояние игры в сессию."""
    if state:
        try:
            session[FLASK_SESSION_KEY] = state.to_dict()
            session.modified = True # Указываем Flask, что сессия была изменена
            app.logger.debug(f"Game state saved to session (Street: {state.street}).")
        except Exception as e:
            app.logger.error(f"Error saving game state: {e}", exc_info=True)
            session.pop(FLASK_SESSION_KEY, None) # Очищаем при ошибке
    else:
        session.pop(FLASK_SESSION_KEY, None)
        app.logger.debug("Game state removed from session.")

def load_game_state() -> Optional[GameState]:
    """Загружает состояние игры из сессии."""
    state_dict = session.get(FLASK_SESSION_KEY)
    if state_dict and isinstance(state_dict, dict):
        try:
            state = GameState.from_dict(state_dict)
            app.logger.debug(f"Game state loaded from session (Street: {state.street}).")
            return state
        except Exception as e:
            app.logger.error(f"Error loading game state from dict: {e}", exc_info=True)
            session.pop(FLASK_SESSION_KEY, None) # Очищаем при ошибке
            return None
    elif state_dict:
        app.logger.warning(f"Invalid game state format in session: {type(state_dict)}. Clearing.")
        session.pop(FLASK_SESSION_KEY, None)
        return None
    app.logger.debug("No game state found in session.")
    return None

# --- Вспомогательные функции ---

def run_ai_turn(current_game_state: GameState, ai_player_index: int) -> GameState:
    """
    Выполняет ОДИН ход AI и ВОЗВРАЩАЕТ новое состояние (без вызова advance_state).
    Обрабатывает как обычные ходы, так и Фантазию.
    """
    state = current_game_state # Работаем с переданным состоянием
    if state._player_finished_round[ai_player_index]:
        app.logger.debug(f"AI Player {ai_player_index} already finished round. Skipping turn.")
        return state # Возвращаем исходное состояние, если игрок уже закончил

    action: Optional[Any] = None
    is_fl_turn = state.is_fantasyland_round and state.fantasyland_status[ai_player_index]
    ai_hand = state.get_player_hand(ai_player_index)

    if not ai_hand:
        app.logger.warning(f"AI Player {ai_player_index} has no hand. Skipping turn.")
        # Если руки нет, но игрок не закончил, это может быть ошибка состояния
        # Можно принудительно завершить раунд для AI или вернуть как есть
        return state

    if ai_agent is None:
        app.logger.error(f"AI Agent is None! AI Player {ai_player_index} will foul.")
        if is_fl_turn:
            return state.apply_fantasyland_foul(ai_player_index, ai_hand)
        else: # Обычный фол (применяем вручную для простоты)
             new_state_foul = state.copy()
             board = new_state_foul.boards[ai_player_index]
             board.is_foul = True
             new_state_foul._player_finished_round[ai_player_index] = True
             if new_state_foul.current_hands.get(ai_player_index):
                 new_state_foul.private_discard[ai_player_index].extend(new_state_foul.current_hands[ai_player_index])
                 new_state_foul.current_hands[ai_player_index] = None
             return new_state_foul

    # Получаем действие от AI
    try:
        app.logger.info(f"AI Player {ai_player_index} choosing action (FL: {is_fl_turn})...")
        action = ai_agent.choose_action(state) # Передаем текущее состояние
        action_repr = ai_agent._format_action(action) if action else "None"
        app.logger.info(f"AI Player {ai_player_index} chose action: {action_repr}")
    except Exception as e:
        app.logger.error(f"Error getting AI action: {e}", exc_info=True)
        action = None # Считаем, что AI не смог выбрать ход

    # Применяем действие
    new_state: GameState
    original_state_repr = state.get_state_representation() # Для проверки изменения

    if action is None:
        app.logger.warning(f"AI Player {ai_player_index} failed to choose action or error occurred. Applying foul.")
        if is_fl_turn:
            new_state = state.apply_fantasyland_foul(ai_player_index, ai_hand)
        else: # Обычный фол
             new_state_foul = state.copy()
             board = new_state_foul.boards[ai_player_index]
             board.is_foul = True
             new_state_foul._player_finished_round[ai_player_index] = True
             if new_state_foul.current_hands.get(ai_player_index):
                 new_state_foul.private_discard[ai_player_index].extend(new_state_foul.current_hands[ai_player_index])
                 new_state_foul.current_hands[ai_player_index] = None
             new_state = new_state_foul
    elif isinstance(action, tuple) and action[0] == "FANTASYLAND_PLACEMENT":
        _, placement, discarded = action
        app.logger.info(f"AI applying Fantasyland placement...")
        new_state = state.apply_fantasyland_placement(ai_player_index, placement, discarded)
    elif isinstance(action, tuple) and action[0] == "FANTASYLAND_FOUL":
        _, hand_to_discard = action
        app.logger.warning(f"AI applying Fantasyland foul (solver failed or chose foul).")
        new_state = state.apply_fantasyland_foul(ai_player_index, hand_to_discard)
    else: # Обычное действие
        app.logger.info(f"AI applying regular action (Street {state.street})...")
        try:
            next_state_candidate = state.apply_action(ai_player_index, action)
            # --- ИСПРАВЛЕНО: Проверка, что состояние действительно изменилось ---
            if next_state_candidate.get_state_representation() == original_state_repr:
                 app.logger.error(f"AI apply_action returned same state! Applying foul manually.")
                 raise RuntimeError("apply_action did not change state") # Вызываем ошибку для перехода к фолу
            new_state = next_state_candidate
        except Exception as e_apply:
             app.logger.error(f"Error applying AI regular action: {e_apply}. Applying foul.", exc_info=True)
             new_state_foul = state.copy()
             board = new_state_foul.boards[ai_player_index]
             board.is_foul = True
             new_state_foul._player_finished_round[ai_player_index] = True
             if new_state_foul.current_hands.get(ai_player_index):
                 new_state_foul.private_discard[ai_player_index].extend(new_state_foul.current_hands[ai_player_index])
                 new_state_foul.current_hands[ai_player_index] = None
             new_state = new_state_foul

    app.logger.info(f"AI Player {ai_player_index} action applied. Round finished: {new_state._player_finished_round[ai_player_index]}")
    return new_state


def get_state_for_frontend(state: GameState, player_idx: int) -> dict:
    """Формирует данные о состоянии игры для отправки на фронтенд."""
    # (Логика без изменений, но использует обновленный GameState)
    opponent_idx = 1 - player_idx
    boards_data: List[Dict[str, List[str]]] = []
    try:
        for i, board in enumerate(state.boards):
            board_data = {r_name: [card_to_str(c) for c in board.rows[r_name]] for r_name in PlayerBoard.ROW_NAMES}
            boards_data.append(board_data)
    except Exception as e:
         app.logger.error(f"Error processing board data for frontend: {e}", exc_info=True)
         empty_row = [CARD_PLACEHOLDER] * 5
         empty_top = [CARD_PLACEHOLDER] * 3
         boards_data = [{'top': empty_top, 'middle': empty_row, 'bottom': empty_row}] * GameState.NUM_PLAYERS

    player_hand_list = state.get_player_hand(player_idx)
    current_hand = [card_to_str(c) for c in player_hand_list] if player_hand_list else []
    is_player_fl = state.is_fantasyland_round and state.fantasyland_status[player_idx]
    fantasyland_hand = current_hand if is_player_fl else []
    regular_hand = current_hand if not is_player_fl else []

    message = ""
    is_waiting = False
    can_act_now = False
    player_finished = state._player_finished_round[player_idx]
    player_can_move = state.get_player_to_move() == player_idx # Проверяем, может ли игрок ходить СЕЙЧАС

    if state.is_round_over():
        message = "Раунд завершен! Нажмите 'Начать Раунд'."
        try:
            score_p0 = state.get_terminal_score()
            score_display = score_p0 if player_idx == 0 else -score_p0
            message += f" Счет за раунд: {score_display}"
        except Exception as e:
            app.logger.error(f"Error calculating terminal score: {e}", exc_info=True)
            message += " (Ошибка подсчета очков)"
    elif player_finished:
        message = "Вы завершили раунд. Ожидание AI..."
        is_waiting = True
    else:
         if player_can_move: # Если игрок может ходить сейчас
              can_act_now = True
              is_waiting = False
              if is_player_fl:
                   message = f"Ваш ход: Разместите руку Фантазии ({len(fantasyland_hand)} карт)."
              else:
                   message = f"Ваш ход (Улица {state.street}). Разместите карты."
         else: # Игрок не может ходить сейчас (ожидает)
              is_waiting = True
              if state.get_player_hand(player_idx) is None:
                   message = f"Ожидание раздачи (Улица {state.street})..."
              else:
                   message = "Ожидание хода AI..." # Если карты есть, но ходить не может - ждет AI

    player_discard_count = len(state.private_discard[player_idx])
    frontend_state = {
        "playerBoard": boards_data[player_idx],
        "opponentBoard": boards_data[opponent_idx],
        "humanPlayerIndex": player_idx,
        "street": state.street,
        "hand": regular_hand,
        "fantasylandHand": fantasyland_hand,
        "isFantasylandRound": state.is_fantasyland_round,
        "playerFantasylandStatus": state.fantasyland_status[player_idx],
        "isGameOver": state.is_round_over(),
        "playerDiscardCount": player_discard_count,
        "message": message,
        "isWaiting": is_waiting,
        "playerFinishedRound": player_finished,
        "canActNow": can_act_now,
        "error_message": None # Очищаем поле ошибки по умолчанию
    }
    return frontend_state


# --- Маршруты Flask ---
app.logger.info("Defining Flask routes...")

@app.route('/')
def index():
    """Отдает главную HTML страницу."""
    return render_template('index.html')

@app.route('/api/game_state', methods=['GET'])
def get_game_state_api():
    """Возвращает текущее состояние игры для фронтенда."""
    game_state = load_game_state()
    is_initial_request = False
    if game_state is None:
        app.logger.info("No game state in session, creating initial empty state.")
        # Создаем пустое состояние, готовое к началу раунда
        game_state = GameState(dealer_idx=random.choice([0, 1]))
        game_state.street = 0
        game_state._player_finished_round = [True, True] # Считаем, что раунд "завершен" до старта
        save_game_state(game_state)
        is_initial_request = True
        app.logger.info("Initial empty state created and saved.")

    try:
        frontend_state = get_state_for_frontend(game_state, HUMAN_PLAYER_IDX)
        # Если это первый запрос или раунд завершен, показываем сообщение о старте
        if is_initial_request or game_state.is_round_over():
             frontend_state["message"] = "Нажмите 'Начать Раунд' для старта новой игры."
             frontend_state["isGameOver"] = True
             frontend_state["isWaiting"] = False
             frontend_state["canActNow"] = False
             frontend_state["hand"] = []
             frontend_state["fantasylandHand"] = []
        return jsonify(frontend_state)
    except Exception as e:
        app.logger.error(f"Error preparing state for frontend API: {e}", exc_info=True)
        # Возвращаем базовое состояние ошибки
        empty_board_dict = {r: [CARD_PLACEHOLDER] * PlayerBoard.ROW_CAPACITY[r] for r in PlayerBoard.ROW_NAMES}
        error_response = {
            "error_message": "Ошибка загрузки состояния игры.", "isGameOver": True,
            "message": "Ошибка сервера. Обновите страницу.", "humanPlayerIndex": HUMAN_PLAYER_IDX,
            "playerBoard": empty_board_dict, "opponentBoard": empty_board_dict,
            "street": 0, "hand": [], "fantasylandHand": [], "isFantasylandRound": False,
            "playerFantasylandStatus": False, "playerDiscardCount": 0,
            "isWaiting": False, "playerFinishedRound": True, "canActNow": False,
        }
        return jsonify(error_response), 500


@app.route('/start', methods=['POST'])
def start_game():
    """Начинает новый раунд игры."""
    app.logger.info("Route /start called")
    old_state = load_game_state()
    fl_status_carryover = [False, False]
    fl_cards_carryover = [0, 0]
    last_dealer = -1

    if old_state and old_state.street > 0: # Переносим статус только если был предыдущий раунд
        fl_status_carryover = old_state.next_fantasyland_status
        fl_cards_carryover = old_state.fantasyland_cards_to_deal
        last_dealer = old_state.dealer_idx
        app.logger.info(f"Carrying over FL: {fl_status_carryover}, Cards: {fl_cards_carryover} from previous round.")
    else:
        app.logger.info("Starting first round or after error. FL status reset.")

    # Определяем нового дилера
    dealer_idx = (last_dealer + 1) % GameState.NUM_PLAYERS if last_dealer != -1 else random.choice([0, 1])

    try:
        # Создаем новое состояние с учетом ФЛ
        game_state = GameState(dealer_idx=dealer_idx,
                              fantasyland_status=fl_status_carryover,
                              fantasyland_cards_to_deal=fl_cards_carryover)
        game_state.start_new_round(dealer_idx) # Начинает раунд и раздает начальные карты
        app.logger.info(f"New round started. Dealer: {dealer_idx}. FL: {game_state.fantasyland_status}. Street: {game_state.street}")
    except Exception as e:
        app.logger.error(f"Error starting new round: {e}", exc_info=True)
        return jsonify({"error": "Ошибка инициализации раунда."}), 500

    # --- Цикл обработки ходов AI сразу после старта ---
    current_state = game_state
    ai_turn_counter = 0
    while not current_state.is_round_over() and ai_turn_counter < MAX_AI_TURNS_IN_SEQUENCE:
         player_to_move = current_state.get_player_to_move()
         app.logger.debug(f"Start loop: Player to move: {player_to_move}")

         if player_to_move == AI_PLAYER_IDX:
              app.logger.info(f"AI's turn immediately after start (Turn {ai_turn_counter + 1}).")
              try:
                   # Выполняем ход AI
                   current_state = run_ai_turn(current_state, AI_PLAYER_IDX)
                   # Продвигаем состояние ПОСЛЕ хода AI
                   if not current_state.is_round_over():
                        current_state = current_state.advance_state()
              except Exception as e_ai:
                   app.logger.error(f"Error during AI turn after start: {e_ai}", exc_info=True)
                   break # Прерываем цикл при ошибке AI
              ai_turn_counter += 1
         elif player_to_move == HUMAN_PLAYER_IDX:
              app.logger.info("Human's turn after start.")
              break # Ход человека, выходим из цикла
         else: # player_to_move == -1 (Ожидание)
              app.logger.info("No player can move after start (waiting or round over).")
              # Пытаемся продвинуть состояние, чтобы раздать карты, если нужно
              advanced_state = current_state.advance_state()
              if advanced_state == current_state: # Если advance_state не помог
                   break # Выходим из цикла
              current_state = advanced_state # Продолжаем цикл с новым состоянием

    if ai_turn_counter >= MAX_AI_TURNS_IN_SEQUENCE:
         app.logger.warning("Max AI turns limit reached after start. Breaking loop.")

    app.logger.info("Saving state after /start processing.")
    save_game_state(current_state)
    frontend_state = get_state_for_frontend(current_state, HUMAN_PLAYER_IDX)
    app.logger.info("Returning state after /start.")
    return jsonify(frontend_state)


@app.route('/move', methods=['POST'])
def handle_move():
    """Обрабатывает ход человека и запускает последующие ходы AI."""
    app.logger.info("Route /move called")
    game_state = load_game_state()

    # Проверки состояния
    if game_state is None:
        app.logger.warning("Move attempt with no game state.")
        return jsonify({"error": "Игра не найдена. Начните новый раунд."}), 400
    if game_state.is_round_over():
        app.logger.warning("Move attempt after round over.")
        return jsonify({"error": "Раунд завершен. Начните новый раунд."}), 400
    if game_state._player_finished_round[HUMAN_PLAYER_IDX]:
        app.logger.warning("Move attempt player finished.")
        return jsonify({"error": "Вы уже завершили раунд."}), 400

    # Проверяем, действительно ли сейчас ход человека
    if game_state.get_player_to_move() != HUMAN_PLAYER_IDX:
         app.logger.warning(f"Move attempt by human when it's not their turn (player to move: {game_state.get_player_to_move()}).")
         frontend_state = get_state_for_frontend(game_state, HUMAN_PLAYER_IDX)
         frontend_state["error_message"] = "Сейчас не ваш ход."
         return jsonify(frontend_state), 400 # Возвращаем ошибку 400 Bad Request

    move_data = request.json
    if not move_data:
        app.logger.warning("Move attempt with no move data.")
        return jsonify({"error": "Нет данных хода."}), 400
    app.logger.debug(f"Received move data: {move_data}")

    current_state = game_state # Начинаем с текущего состояния

    try:
        # --- Применение хода человека ---
        is_player_in_fl = current_state.is_fantasyland_round and current_state.fantasyland_status[HUMAN_PLAYER_IDX]
        original_state_repr = current_state.get_state_representation() # Для проверки изменения

        if is_player_in_fl:
            # Обработка хода Фантазии
            placement_raw = move_data.get('placement')
            discarded_raw = move_data.get('discarded')
            if not isinstance(placement_raw, dict) or not isinstance(discarded_raw, list):
                raise ValueError("Invalid Fantasyland data format received from frontend.")

            placement_dict: Dict[str, List[int]] = {}
            discarded_cards: List[int] = []
            try:
                # Парсим размещение
                for row, card_strs in placement_raw.items():
                    if row not in PlayerBoard.ROW_NAMES: raise ValueError(f"Unknown row: {row}")
                    placement_dict[row] = [card_from_str(s) for s in card_strs if s != CARD_PLACEHOLDER]
                # Парсим сброс
                discarded_cards = [card_from_str(s) for s in discarded_raw]
            except ValueError as e_conv:
                raise ValueError(f"Card conversion error in FL data: {e_conv}")

            app.logger.info("Applying human Fantasyland placement...")
            current_state = current_state.apply_fantasyland_placement(HUMAN_PLAYER_IDX, placement_dict, discarded_cards)
        else: # Обычный ход
            action: Optional[Any] = None
            current_hand = current_state.get_player_hand(HUMAN_PLAYER_IDX)
            if not current_hand: raise ValueError("Internal error: no hand for regular move.")

            if current_state.street == 1: # Улица 1
                if len(current_hand) != 5: raise ValueError(f"Invalid hand size {len(current_hand)} for street 1.")
                placements_raw = move_data.get('placements')
                if not isinstance(placements_raw, list) or len(placements_raw) != 5: raise ValueError("Street 1 needs 5 placements.")
                placements: List[Tuple[int, str, int]] = []
                placed_card_strs = set()
                try:
                    for p in placements_raw:
                        card_str = p['card']; row = p['row']; index = int(p['index'])
                        if card_str in placed_card_strs: raise ValueError(f"Duplicate card in placement: {card_str}")
                        if row not in PlayerBoard.ROW_NAMES: raise ValueError(f"Invalid row: {row}")
                        placed_card_strs.add(card_str); card_int = card_from_str(card_str); placements.append((card_int, row, index))
                except (ValueError, KeyError, TypeError) as e_parse: raise ValueError(f"Parse error street 1: {e_parse}")
                # --- ИСПРАВЛЕНО: Сортируем для каноничности ---
                action = (tuple(sorted(placements)), tuple())
            elif 2 <= current_state.street <= 5: # Улицы 2-5
                if len(current_hand) != 3: raise ValueError(f"Invalid hand size {len(current_hand)} for street {current_state.street}.")
                placements_raw = move_data.get('placements'); discard_str = move_data.get('discard')
                if not isinstance(placements_raw, list) or len(placements_raw) != 2 or not isinstance(discard_str, str): raise ValueError("Invalid Pineapple data format.")
                try:
                    p1_raw, p2_raw = placements_raw[0], placements_raw[1]
                    card1 = card_from_str(p1_raw['card']); row1, idx1 = p1_raw['row'], int(p1_raw['index'])
                    card2 = card_from_str(p2_raw['card']); row2, idx2 = p2_raw['row'], int(p2_raw['index'])
                    discarded_card = card_from_str(discard_str)
                    action_cards_set = {card1, card2, discarded_card}; hand_set = set(current_hand)
                    if len(action_cards_set) != 3: raise ValueError("Duplicate cards in action/discard.")
                    if action_cards_set != hand_set: raise ValueError("Action cards mismatch hand.")
                    if row1 not in PlayerBoard.ROW_NAMES or row2 not in PlayerBoard.ROW_NAMES: raise ValueError("Invalid row.")
                    # --- ИСПРАВЛЕНО: Сортируем для каноничности ---
                    p1_t = (card1, row1, idx1); p2_t = (card2, row2, idx2)
                    action = tuple(sorted((p1_t, p2_t))) + (discarded_card,)
                except (ValueError, KeyError, TypeError, IndexError) as e_parse: raise ValueError(f"Parse error Pineapple: {e_parse}")
            else:
                raise ValueError(f"Cannot apply action on invalid street {current_state.street}.")

            # Применяем обычное действие
            if action:
                app.logger.info(f"Applying human regular action (Street {current_state.street})...")
                current_state = current_state.apply_action(HUMAN_PLAYER_IDX, action)
            else:
                raise RuntimeError("Internal error: Failed to create action from move data.")

        # Проверяем, изменилось ли состояние после хода человека
        if current_state.get_state_representation() == original_state_repr:
             # Если apply_action/apply_fantasyland_placement вернули исходное состояние из-за ошибки
             raise RuntimeError("Failed to apply human move (state did not change). Check logs for errors in GameState.")

        app.logger.info(f"Human action applied. Human finished: {current_state._player_finished_round[HUMAN_PLAYER_IDX]}")

        # --- Цикл продвижения состояния и ходов AI ---
        ai_turn_counter = 0
        while not current_state.is_round_over() and ai_turn_counter < MAX_AI_TURNS_IN_SEQUENCE:
             # 1. Продвигаем состояние (переход хода/улицы, раздача)
             advanced_state = current_state.advance_state()
             if advanced_state == current_state and not current_state.is_round_over():
                  # Если advance_state не изменил состояние (например, оба ждут карт)
                  app.logger.debug("advance_state did not change state. Waiting.")
                  current_state = advanced_state # Сохраняем на всякий случай
                  break # Выходим из цикла, ждем следующего действия игрока/события
             current_state = advanced_state

             # 2. Определяем, кто ходит
             player_to_move = current_state.get_player_to_move()
             app.logger.debug(f"State advanced after human/AI turn. Player to move: {player_to_move}")

             # 3. Если ход AI, выполняем его
             if player_to_move == AI_PLAYER_IDX:
                  app.logger.info(f"AI's turn (Turn {ai_turn_counter + 1}).")
                  try:
                       current_state = run_ai_turn(current_state, AI_PLAYER_IDX) # AI ходит
                       ai_turn_counter += 1
                       # Не вызываем advance_state здесь, он будет вызван в начале следующей итерации
                  except Exception as e_ai:
                       app.logger.error(f"Error during AI turn in sequence: {e_ai}", exc_info=True)
                       break # Прерываем цикл при ошибке AI
             # 4. Если ход человека или никто не ходит, выходим из цикла
             else:
                  app.logger.info(f"Not AI's turn (Player: {player_to_move}). Exiting AI loop.")
                  break

        if ai_turn_counter >= MAX_AI_TURNS_IN_SEQUENCE:
             app.logger.warning("Max AI turns limit reached after human move. Breaking loop.")

        # --- Сохранение и ответ ---
        app.logger.info("Saving final state after /move processing.")
        save_game_state(current_state)
        frontend_state = get_state_for_frontend(current_state, HUMAN_PLAYER_IDX)
        app.logger.info("Returning final state after /move.")
        return jsonify(frontend_state)

    # Обработка ожидаемых ошибок (ValueError, TypeError и т.д.)
    except (ValueError, TypeError, RuntimeError, IndexError) as e:
        app.logger.warning(f"Move Error ({type(e).__name__}): {e}", exc_info=True)
        # Возвращаем исходное состояние с сообщением об ошибке
        original_state = load_game_state()
        fe_state = {}
        if original_state:
            fe_state = get_state_for_frontend(original_state, HUMAN_PLAYER_IDX)
        else: # Если не удалось загрузить состояние
             empty_board = {r: [CARD_PLACEHOLDER]*PlayerBoard.ROW_CAPACITY[r] for r in PlayerBoard.ROW_NAMES}
             fe_state = {"playerBoard": empty_board, "opponentBoard": empty_board, "message": "Ошибка состояния.", "isGameOver": True}
        fe_state["error_message"] = f"Неверный ход: {e}"
        return jsonify(fe_state), 400 # Bad Request
    # Обработка неожиданных ошибок сервера
    except Exception as e:
        app.logger.error(f"Unexpected Error during /move: {e}", exc_info=True)
        return jsonify({"error": "Произошла неожиданная ошибка сервера."}), 500


# --- Запуск приложения (без изменений) ---
if __name__ == '__main__':
    app.logger.info("--- Starting Main Execution ---")
    port = int(os.environ.get('PORT', 8080))
    # Включаем debug mode только если не в продакшене и FLASK_DEBUG установлен
    debug_mode = os.environ.get('FLASK_DEBUG', '0').lower() in ['true', '1', 'yes'] and not is_production
    if debug_mode:
        app.logger.setLevel(logging.DEBUG) # Устанавливаем DEBUG уровень для логгера Flask
        app.logger.info("Flask debug mode is ON.")
    else:
        app.logger.info("Flask debug mode is OFF.")
    app.logger.info(f"Starting Flask app server on host 0.0.0.0, port {port}")
    # use_reloader=False рекомендуется при запуске через Gunicorn или другие WSGI серверы
    app.run(host='0.0.0.0', port=port, debug=debug_mode, use_reloader=debug_mode)
    app.logger.info("--- Flask App Exiting ---")
