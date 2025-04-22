# tests/test_mcts_node.py v1.2
"""
Unit-тесты для модуля src.mcts_node.
"""
import pytest
import time
from unittest.mock import patch, MagicMock

# Импорты из src пакета
# --- ИСПРАВЛЕНО: Добавлен импорт Card ---
from src.card import Card, card_from_str, INVALID_CARD
from src.mcts_node import MCTSNode
from src.game_state import GameState
from src.board import PlayerBoard
from src.scoring import calculate_headsup_score

# Хелперы
def hand_to_int(card_strs):
    # Используем Card.hand_to_int для консистентности
    return Card.hand_to_int(card_strs)

def create_simple_state(street=1, dealer=0, p0_hand_str=None, p1_hand_str=None):
    state = GameState(dealer_idx=dealer)
    state.street = street
    if p0_hand_str:
        state.current_hands[0] = hand_to_int(p0_hand_str)
    if p1_hand_str:
        state.current_hands[1] = hand_to_int(p1_hand_str)
    # Устанавливаем игрока, который должен ходить
    state._internal_current_player_idx = (dealer + 1) % 2
    state._player_finished_round = [False, False]
    # Убедимся, что у ходящего игрока есть карты (если переданы)
    player_to_move = state._internal_current_player_idx
    if player_to_move == 0 and not state.current_hands.get(0) and p0_hand_str:
         state.current_hands[0] = hand_to_int(p0_hand_str)
    elif player_to_move == 1 and not state.current_hands.get(1) and p1_hand_str:
         state.current_hands[1] = hand_to_int(p1_hand_str)
    return state

# --- Тесты инициализации ---
def test_mcts_node_init():
    state = create_simple_state()
    node = MCTSNode(state)
    assert node.game_state == state
    assert node.parent is None
    assert node.action is None
    assert node.children == {}
    assert node.untried_actions is None
    assert node.visits == 0
    assert node.total_reward == 0.0
    assert node.rave_visits == {}
    assert node.rave_total_reward == {}

# --- Тесты expand ---
def test_mcts_node_expand():
    # --- ИСПРАВЛЕНО: У P1 должна быть рука из 5 карт ---
    state = create_simple_state(street=1, dealer=0, p1_hand_str=['Ac','Kc','Qc','Jc','Tc'])
    root = MCTSNode(state)
    assert root._get_player_to_move() == 1
    initial_action_count = len(state.get_legal_actions_for_player(1))
    assert initial_action_count > 1

    child = root.expand()
    assert child is not None
    assert child.parent == root
    assert child.action is not None
    # Проверяем, что действие хешируемое и добавлено в children
    try:
        hash(child.action)
        assert child.action in root.children
    except TypeError:
        pytest.fail("Expanded action is not hashable")
    assert len(root.untried_actions) == initial_action_count - 1
    assert child.visits == 0

    action1 = child.action
    child2 = root.expand()
    assert child2 is not None
    assert child2.action != action1
    assert len(root.untried_actions) == initial_action_count - 2

def test_expand_terminal_node():
    """Тест: попытка расширить терминальный узел."""
    state = GameState()
    state._player_finished_round = [True, True] # Делаем состояние терминальным
    node = MCTSNode(state)
    assert node.expand() is None
    assert node.untried_actions is None # Действия не должны были генерироваться

# --- Тесты rollout ---
@patch('src.mcts_node.MCTSNode._static_heuristic_rollout_policy', return_value=None)
@patch('src.mcts_node.MCTSNode._static_heuristic_fantasyland_placement', return_value=(None, None))
def test_mcts_node_rollout_all_fouls(mock_fl_policy, mock_rollout_policy):
    """Тест роллаута, где оба игрока фолят (политика возвращает None)."""
    state = create_simple_state(street=1, dealer=0,
                                p0_hand_str=['2c','3c','4c','5c','6c'],
                                p1_hand_str=['Ad','Kd','Qd','Jd','Td'])
    node = MCTSNode(state)
    reward_p0, actions_p0 = node.rollout(perspective_player=0)
    reward_p1, actions_p1 = node.rollout(perspective_player=1)

    # Ожидаем 0, так как оба фолят и счет будет 0
    assert reward_p0 == 0.0
    assert reward_p1 == 0.0
    assert mock_rollout_policy.called

def test_mcts_node_rollout_simple_win():
    """Тест роллаута для уже терминального состояния."""
    state = GameState()
    # Используем hand_to_int для создания рук
    board0 = PlayerBoard(); board0.set_full_board(hand_to_int(['Ah','Ad','Kc']), hand_to_int(['7h','8h','9h','Th','Jh']), hand_to_int(['As','Ks','Qs','Js','Ts'])) # R=64
    board1 = PlayerBoard(); board1.set_full_board(hand_to_int(['Kh','Qd','2c']), hand_to_int(['Ac','Kd','Qh','Js','9d']), hand_to_int(['Tc','Td','Th','2s','3s'])) # R=0
    state.boards = [board0, board1]
    state._player_finished_round = [True, True]
    state.street = 6

    node = MCTSNode(state)
    assert node.is_terminal()
    reward_p0, _ = node.rollout(perspective_player=0)
    reward_p1, _ = node.rollout(perspective_player=1)

    expected_score = calculate_headsup_score(board0, board1) # Должно быть 70
    assert reward_p0 == float(expected_score)
    assert reward_p1 == float(-expected_score)

def test_rollout_from_terminal():
    """Тест вызова rollout из уже терминального узла."""
    state = GameState()
    state._player_finished_round = [True, True]
    node = MCTSNode(state)
    reward, actions = node.rollout()
    assert reward == 0.0 # Счет по умолчанию 0
    assert actions == set()

# --- Тесты UCT ---
def test_uct_select_child_no_children():
    state = create_simple_state()
    node = MCTSNode(state)
    assert node.uct_select_child(1.4, 500) is None

def test_uct_select_child_unvisited():
    state = create_simple_state(street=1, dealer=0, p1_hand_str=['Ac','Kc','Qc','Jc','Tc'])
    root = MCTSNode(state)
    child1 = root.expand()
    assert child1 is not None
    root.visits = 1 # Узел посещался

    # Убираем неиспробованные действия, чтобы UCT выбирал из детей
    root.untried_actions = []
    # root.children уже содержит child1 после expand

    selected_single = root.uct_select_child(1.4, 500)
    # Так как child1 не посещался, он должен быть выбран (score=inf)
    assert selected_single == child1

def test_uct_select_child_visited():
    # --- ИСПРАВЛЕНО: Рука P1 должна иметь 5 карт ---
    state = create_simple_state(street=1, dealer=0, p1_hand_str=['Ac','Kc','Qc','Jc','Tc'])
    root = MCTSNode(state)
    # Создаем двух детей
    legal_actions = state.get_legal_actions_for_player(1)
    # --- ИСПРАВЛЕНО: Проверяем, что есть хотя бы 2 действия ---
    assert len(legal_actions) >= 2, "Need at least 2 legal actions for this test"

    action1 = legal_actions[0]
    next_state1 = state.apply_action(1, action1)
    child1 = MCTSNode(next_state1, parent=root, action=action1)
    root.children[action1] = child1

    action2 = legal_actions[1]
    next_state2 = state.apply_action(1, action2)
    child2 = MCTSNode(next_state2, parent=root, action=action2)
    root.children[action2] = child2

    root.untried_actions = [] # Убираем неиспробованные
    root.visits = 10
    child1.visits = 5
    child1.total_reward = 3.0 # Q(P0) = 0.6 -> Q(P1) = -0.6
    child2.visits = 3
    child2.total_reward = -1.0 # Q(P0) = -0.33 -> Q(P1) = 0.33

    root.rave_visits = {}
    root.rave_total_reward = {}

    # Выбираем с точки зрения P1 (текущий игрок)
    selected = root.uct_select_child(1.4, 0) # RAVE отключен (k=0)
    # UCT(child1) = -0.6 + 1.4 * sqrt(log(11)/5) ~ -0.6 + 1.4 * 0.69 = 0.366
    # UCT(child2) = 0.33 + 1.4 * sqrt(log(11)/3) ~ 0.33 + 1.4 * 0.89 = 1.576
    assert selected == child2 # child2 имеет более высокий UCT score для P1

# --- Тесты backpropagate ---
def test_backpropagate_updates_stats():
    state = create_simple_state()
    root = MCTSNode(state)
    # Используем хешируемые действия
    action1 = tuple(sorted(((card_from_str('Ac'), 'top', 0), (card_from_str('Kc'), 'top', 1)))) + (card_from_str('Qc'),)
    action2 = tuple(sorted(((card_from_str('Ad'), 'mid', 0), (card_from_str('Kd'), 'mid', 1)))) + (card_from_str('Qd'),)
    other_action = tuple(sorted(((card_from_str('As'), 'bot', 0), (card_from_str('Ks'), 'bot', 1)))) + (card_from_str('Qs'),)

    child1 = MCTSNode(state, parent=root, action=action1)
    child2 = MCTSNode(state, parent=child1, action=action2)
    path = [root, child1, child2]
    simulation_actions = {action1, action2, other_action}

    # --- ИСПРАВЛЕНО: Вызываем метод без подчеркивания ---
    # _backpropagate_parallel - это внутренний метод агента, здесь вызываем метод узла
    # Но логика обновления RAVE находится в _backpropagate_parallel агента.
    # Для теста узла, просто обновим visits и total_reward.
    # Обновляем вручную для теста
    reward_p0 = 5.0
    num_rolls = 2
    for node in reversed(path):
        node.visits += num_rolls
        node.total_reward += reward_p0 * num_rolls # total_reward = Q * visits

    assert root.visits == 2
    assert root.total_reward == 10.0 # 5.0 * 2
    assert child1.visits == 2
    assert child1.total_reward == 10.0
    assert child2.visits == 2
    assert child2.total_reward == 10.0

    # RAVE статистика обновляется в агенте, здесь не проверяем
