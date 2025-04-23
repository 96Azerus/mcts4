# tests/test_game_state.py v1.2
"""
Unit-тесты для модуля src.game_state.
"""

import pytest
import random
import copy

# Импорты из src пакета
from src.game_state import GameState
from src.card import card_from_str, card_to_str, INVALID_CARD, CARD_PLACEHOLDER
from src.deck import Deck
from src.board import PlayerBoard
from src.scoring import calculate_headsup_score # Для проверки get_terminal_score

# Хелперы
def hand(card_strs):
    return [card_from_str(s) if isinstance(s, str) and s and s != CARD_PLACEHOLDER else None for s in card_strs]

def create_deck_with_known_cards(top_cards_strs: list[str]) -> Deck:
    """Создает колоду, где верхние карты известны (для deal)."""
    deck = Deck()
    top_cards_ints = [card_from_str(s) for s in top_cards_strs]
    deck.remove(top_cards_ints)
    remaining_cards = deck.get_remaining_cards()
    random.shuffle(remaining_cards)
    final_card_list = top_cards_ints + remaining_cards
    test_deck = Deck(cards=set(final_card_list))
    return test_deck

# --- Тесты инициализации и старта раунда ---

def test_gamestate_init_default():
    """Тест инициализации по умолчанию."""
    state = GameState()
    assert state.dealer_idx == 0
    assert state._internal_current_player_idx == 1
    assert state.street == 0
    assert len(state.boards) == 2
    assert len(state.deck) == 52
    assert not state.is_fantasyland_round
    assert state.fantasyland_status == [False, False]
    assert state.next_fantasyland_status == [False, False]
    assert state.fantasyland_cards_to_deal == [0, 0]
    assert state._player_finished_round == [False, False]

def test_gamestate_start_new_round_normal():
    """Тест старта обычного раунда."""
    state = GameState(dealer_idx=1)
    state.start_new_round(dealer_button_idx=1)

    assert state.street == 1
    assert state.dealer_idx == 1
    assert state._internal_current_player_idx == 0
    assert not state.is_fantasyland_round
    assert state.current_hands.get(0) is not None
    assert len(state.current_hands[0]) == 5
    assert state.current_hands.get(1) is None
    assert len(state.deck) == 52 - 5
    assert state._player_acted_this_street == [False, False]
    assert state._player_finished_round == [False, False]

def test_gamestate_start_new_round_fantasyland():
    """Тест старта раунда Фантазии."""
    state = GameState(next_fantasyland_status=[True, False], fantasyland_cards_to_deal=[15, 0])
    state.start_new_round(dealer_button_idx=0)

    assert state.street == 1
    assert state.dealer_idx == 0
    assert state._internal_current_player_idx == 1
    assert state.is_fantasyland_round
    assert state.fantasyland_status == [True, False]
    assert state.fantasyland_hands[0] is not None
    assert len(state.fantasyland_hands[0]) == 15
    assert state.fantasyland_hands[1] is None
    assert state.current_hands.get(0) is None
    assert state.current_hands.get(1) is not None
    assert len(state.current_hands[1]) == 5
    assert len(state.deck) == 52 - 15 - 5

def test_gamestate_start_new_round_fl_carryover():
    """Тест переноса статуса ФЛ между раундами."""
    state = GameState()
    state.next_fantasyland_status = [False, True]
    state.fantasyland_cards_to_deal = [0, 14]
    state.start_new_round(dealer_button_idx=1)

    assert state.is_fantasyland_round
    assert state.fantasyland_status == [False, True]
    assert state.fantasyland_hands[1] is not None
    assert len(state.fantasyland_hands[1]) == 14
    assert state.current_hands.get(0) is not None
    assert len(state.current_hands[0]) == 5

# --- Тесты применения действий ---

def test_gamestate_apply_action_street1():
    """Тест применения действия на улице 1."""
    state = GameState(dealer_idx=1)
    state.start_new_round(1)
    hand_p0 = state.current_hands[0]
    assert hand_p0 is not None and len(hand_p0) == 5

    placements = [
        (hand_p0[0], 'bottom', 0), (hand_p0[1], 'bottom', 1),
        (hand_p0[2], 'middle', 0), (hand_p0[3], 'middle', 1),
        (hand_p0[4], 'top', 0)
    ]
    action = (tuple(sorted(placements)), tuple())
    next_state = state.apply_action(0, action)

    assert next_state is not state
    assert next_state.current_hands.get(0) is None
    assert next_state.boards[0].get_total_cards() == 5
    assert hand_p0[0] in next_state.boards[0].get_row_cards('bottom')
    assert hand_p0[4] in next_state.boards[0].get_row_cards('top')
    assert next_state._player_acted_this_street[0] is True
    assert next_state._last_player_acted == 0

def test_gamestate_apply_action_pineapple():
    """Тест применения действия Pineapple (улицы 2-5)."""
    state = GameState(dealer_idx=0)
    state.start_new_round(0)
    state.street = 2
    state._internal_current_player_idx = 0
    hand_p0 = hand(['As', 'Ks', 'Qs'])
    state.current_hands[0] = hand_p0
    state.current_hands[1] = None
    state._player_acted_this_street = [False, False]

    p1 = (hand_p0[0], 'top', 0)
    p2 = (hand_p0[1], 'middle', 0)
    discard = hand_p0[2]
    # --- ИСПРАВЛЕНО: Формат действия Pineapple ---
    action = (tuple(sorted((p1, p2))), discard)
    next_state = state.apply_action(0, action)

    assert next_state is not state
    assert next_state.current_hands.get(0) is None
    assert next_state.boards[0].rows['top'][0] == hand_p0[0]
    assert next_state.boards[0].rows['middle'][0] == hand_p0[1]
    assert hand_p0[2] in next_state.private_discard[0]
    assert next_state._player_acted_this_street[0] is True
    assert next_state._last_player_acted == 0

def test_gamestate_apply_action_completes_board():
    """Тест, что apply_action корректно обрабатывает завершение доски."""
    state = GameState()
    board = state.boards[0]
    # --- ИСПРАВЛЕНО: Добавляем 11 карт ---
    cards_on_board = hand(['2c','3c','4c','5c','6c','7c','8c','9c','Tc','Jc','Qc'])
    idx = 0
    for r in ['bottom', 'middle']:
        for i in range(5): board.add_card(cards_on_board[idx], r, i); idx+=1
    board.add_card(cards_on_board[idx], 'top', 0); idx+=1 # 11 карт
    assert board.get_total_cards() == 11

    state.street = 5
    state._internal_current_player_idx = 0
    hand_p0 = hand(['Ac', 'Ad', 'Ah']) # Рука для завершения (AA на топ)
    state.current_hands[0] = hand_p0

    p1 = (hand_p0[0], 'top', 1) # Ac
    p2 = (hand_p0[1], 'top', 2) # Ad
    discard = hand_p0[2]       # Ah
    # --- ИСПРАВЛЕНО: Формат действия Pineapple ---
    action = (tuple(sorted((p1, p2))), discard)
    next_state = state.apply_action(0, action)

    assert next_state.boards[0].is_complete()
    assert next_state._player_finished_round[0] is True
    # --- ИСПРАВЛЕНО: Ожидаем True для ФЛ ---
    assert next_state.next_fantasyland_status[0] is True # AA -> 16 карт
    assert next_state.fantasyland_cards_to_deal[0] == 16

def test_gamestate_apply_fantasyland_placement():
    """Тест применения размещения Фантазии."""
    state = GameState(fantasyland_status=[True, False], fantasyland_cards_to_deal=[14, 0])
    state.start_new_round(0)
    hand_fl = state.fantasyland_hands[0]
    assert hand_fl is not None and len(hand_fl) == 14

    placement_dict = {
        'bottom': hand_fl[0:5],
        'middle': hand_fl[5:10],
        'top': hand_fl[10:13]
    }
    discarded = [hand_fl[13]]

    next_state = state.apply_fantasyland_placement(0, placement_dict, discarded)

    assert next_state is not state
    assert next_state.fantasyland_hands[0] is None
    assert next_state.boards[0].is_complete()
    assert next_state._player_finished_round[0] is True
    assert discarded[0] in next_state.private_discard[0]
    assert next_state._last_player_acted == 0

def test_gamestate_apply_fantasyland_foul():
    """Тест применения фола в Фантазии."""
    state = GameState(fantasyland_status=[True, False], fantasyland_cards_to_deal=[14, 0])
    state.start_new_round(0)
    hand_fl = state.fantasyland_hands[0]
    assert hand_fl is not None

    next_state = state.apply_fantasyland_foul(0, hand_fl)

    assert next_state is not state
    assert next_state.fantasyland_hands[0] is None
    assert next_state.boards[0].is_foul is True
    assert next_state.boards[0].get_total_cards() == 0
    assert next_state._player_finished_round[0] is True
    assert set(hand_fl).issubset(set(next_state.private_discard[0]))
    assert next_state.next_fantasyland_status[0] is False
    assert next_state._last_player_acted == 0

# --- Тесты продвижения состояния ---

def test_gamestate_get_player_to_move():
    """Тестирует определение игрока для хода."""
    state = GameState(dealer_idx=0); state.start_new_round(0)
    assert state.get_player_to_move() == 1

    state._player_acted_this_street[1] = True; state.current_hands[1] = None; state.current_hands[0] = None
    state._last_player_acted = 1
    state._internal_current_player_idx = 0
    assert state.get_player_to_move() == -1

    state.current_hands[0] = hand(['Ac','Kc','Qc','Jc','Tc'])
    assert state.get_player_to_move() == 0

    state._player_acted_this_street[0] = True; state.current_hands[0] = None; state.current_hands[1] = None
    state._last_player_acted = 0
    state._internal_current_player_idx = 1
    assert state.get_player_to_move() == -1

    state_fl = GameState(fantasyland_status=[True, False], fantasyland_cards_to_deal=[14,0])
    state_fl.start_new_round(0)
    state_fl.current_hands[1] = None
    assert state_fl.get_player_to_move() == 0

    state_fl.fantasyland_hands[0] = None
    state_fl.current_hands[1] = hand(['Ac','Kc','Qc','Jc','Tc'])
    assert state_fl.get_player_to_move() == 1

    state._player_finished_round = [True, True]
    assert state.is_round_over()
    assert state.get_player_to_move() == -1


def test_gamestate_advance_state_normal_round():
    """Тестирует advance_state в обычном раунде."""
    state = GameState(dealer_idx=1); state.start_new_round(1)
    hand_p0 = state.current_hands[0]
    action_p0 = (tuple(sorted([(hand_p0[i], 'bottom', i) for i in range(5)])), tuple())
    state_after_p0 = state.apply_action(0, action_p0)
    assert state_after_p0._last_player_acted == 0
    assert state_after_p0._internal_current_player_idx == 0

    state_after_advance1 = state_after_p0.advance_state()
    assert state_after_advance1._internal_current_player_idx == 1
    assert state_after_advance1.current_hands.get(1) is not None
    assert len(state_after_advance1.current_hands[1]) == 5
    assert state_after_advance1.get_player_to_move() == 1

    hand_p1 = state_after_advance1.current_hands[1]
    action_p1 = (tuple(sorted([(hand_p1[i], 'middle', i) for i in range(5)])), tuple())
    state_after_p1 = state_after_advance1.apply_action(1, action_p1)
    assert state_after_p1._last_player_acted == 1
    assert state_after_p1._internal_current_player_idx == 1

    state_after_advance2 = state_after_p1.advance_state()
    assert state_after_advance2.street == 2
    assert state_after_advance2._player_acted_this_street == [False, False]
    assert state_after_advance2._internal_current_player_idx == 0
    assert state_after_advance2.current_hands.get(0) is not None
    assert len(state_after_advance2.current_hands[0]) == 3
    assert state_after_advance2.current_hands.get(1) is not None
    assert len(state_after_advance2.current_hands[1]) == 3
    assert state_after_advance2.get_player_to_move() == 0

def test_gamestate_advance_state_street_change():
    """Тестирует смену улицы после ходов обоих игроков."""
    state = GameState(dealer_idx=0)
    state.start_new_round(0) # Ход P1
    hand_p1 = state.current_hands[1]
    action_p1 = (tuple(sorted([(hand_p1[i], 'bottom', i) for i in range(5)])), tuple())
    state = state.apply_action(1, action_p1)
    state = state.advance_state() # Переход хода к P0, раздача P0
    assert state.street == 1
    assert state.get_player_to_move() == 0
    hand_p0 = state.current_hands[0]
    action_p0 = (tuple(sorted([(hand_p0[i], 'middle', i) for i in range(5)])), tuple())
    state = state.apply_action(0, action_p0)
    state = state.advance_state() # Оба сходили -> смена улицы
    assert state.street == 2
    assert state._internal_current_player_idx == 1
    assert state._player_acted_this_street == [False, False]
    assert len(state.current_hands.get(0)) == 3
    assert len(state.current_hands.get(1)) == 3

def test_gamestate_advance_state_fantasyland():
    """Тестирует advance_state в раунде Фантазии."""
    state = GameState(fantasyland_status=[True, False], fantasyland_cards_to_deal=[14, 0])
    state.start_new_round(0)
    assert state.get_player_to_move() == 1

    hand_p1_s1 = state.current_hands[1]
    action_p1_s1 = (tuple(sorted([(hand_p1_s1[i], 'bottom', i) for i in range(5)])), tuple())
    state_after_p1_s1 = state.apply_action(1, action_p1_s1)
    assert state_after_p1_s1._last_player_acted == 1

    state_after_advance1 = state_after_p1_s1.advance_state()
    # --- ИСПРАВЛЕНО: Улица не должна меняться ---
    assert state_after_advance1.street == 1
    assert state_after_advance1.get_player_to_move() == 0

    hand_p0_fl = state_after_advance1.fantasyland_hands[0]
    placement_p0 = {'bottom': hand_p0_fl[0:5], 'middle': hand_p0_fl[5:10], 'top': hand_p0_fl[10:13]}
    discarded_p0 = [hand_p0_fl[13]]
    state_after_p0_fl = state_after_advance1.apply_fantasyland_placement(0, placement_p0, discarded_p0)
    assert state_after_p0_fl._player_finished_round[0] is True
    assert state_after_p0_fl._last_player_acted == 0

    state_after_advance2 = state_after_p0_fl.advance_state()
    # --- ИСПРАВЛЕНО: Улица должна смениться для P1 ---
    assert state_after_advance2.street == 2
    assert state_after_advance2.current_hands.get(1) is not None
    assert len(state_after_advance2.current_hands[1]) == 3
    assert state_after_advance2.get_player_to_move() == 1

def test_gamestate_advance_state_fl_non_fl_player_moves():
    """Тест: P0 в ФЛ, P1 (не ФЛ) ходит на улице 2."""
    state = GameState(fantasyland_status=[True, False], fantasyland_cards_to_deal=[14, 0])
    state.start_new_round(0)
    state.street = 2
    state._internal_current_player_idx = 1
    state.current_hands[1] = hand(['Ac','Kc','Qc'])
    state.fantasyland_hands[0] = hand(['2s']*14)
    state._player_finished_round = [False, False]
    state._player_acted_this_street = [False, False]

    assert state.get_player_to_move() == 1
    hand_p1 = state.current_hands[1]
    # --- ИСПРАВЛЕНО: Формат действия Pineapple ---
    action_p1 = (tuple(sorted(((hand_p1[0], 'top', 0), (hand_p1[1], 'middle', 0)))), hand_p1[2])
    state = state.apply_action(1, action_p1)
    assert state._last_player_acted == 1

    state = state.advance_state()
    assert state.street == 2
    assert state.get_player_to_move() == 0

# --- Тесты конца раунда и счета ---

def test_gamestate_end_of_round_and_score():
    """Тестирует is_round_over и get_terminal_score."""
    state = GameState()
    board1 = PlayerBoard(); board1.set_full_board(hand(['Ah','Ad','Kc']), hand(['7h','8h','9h','Th','Jh']), hand(['As','Ks','Qs','Js','Ts'])) # R=64
    board2 = PlayerBoard(); board2.set_full_board(hand(['Kh','Qd','2c']), hand(['Ac','Kd','Qh','Js','9d']), hand(['Tc','Td','Th','2s','3s'])) # R=0
    state.boards = [board1, board2]
    state._player_finished_round = [True, True]
    state.street = 6

    assert state.is_round_over()
    assert state.get_terminal_score() == 70

    # Случай с фолом P1
    board1_foul = PlayerBoard(); board1_foul.set_full_board(hand(['Ah','Ad','Ac']), hand(['Ks','Kd','Qc','Qd','2s']), hand(['As','Kh','Qs','Js','Ts'])) # Foul, R=0
    state.boards = [board1_foul, board2]
    assert state.is_round_over()
    # --- ИСПРАВЛЕНО: Ожидаем -6 ---
    assert state.get_terminal_score() == -6 # P1 foul, P2 R=0 -> P0 получает -6

# --- Тесты логики Fantasyland ---

def test_gamestate_fantasyland_entry_and_stay():
    """Тестирует обновление next_fantasyland_status."""
    state = GameState()
    state.street = 5
    state._internal_current_player_idx = 0
    hand_entry = hand(['Ac', 'Ad', 'Kc'])
    state.current_hands[0] = hand_entry
    board = state.boards[0]
    # --- ИСПРАВЛЕНО: Добавляем 11 карт, чтобы осталось 2 слота ---
    cards_on_board = hand(['2c','3c','4c','5c','6c','7c','8c','9c','Tc','Jc','Qc'])
    idx = 0
    for r in ['bottom', 'middle']:
        for i in range(5): board.add_card(cards_on_board[idx], r, i); idx+=1
    board.add_card(cards_on_board[idx], 'top', 0); idx+=1 # 11 карт
    assert board.get_total_cards() == 11

    # Действие: Положить Ac, Ad в top[1], top[2], сбросить Kc
    p1 = (hand_entry[0], 'top', 1) # Ac
    p2 = (hand_entry[1], 'top', 2) # Ad
    discard = hand_entry[2]        # Kc
    # --- ИСПРАВЛЕНО: Формат действия Pineapple ---
    action_entry = (tuple(sorted((p1, p2))), discard)
    state_after_entry = state.apply_action(0, action_entry)

    # --- ИСПРАВЛЕНО: Доска должна быть завершена ---
    assert state_after_entry.boards[0].is_complete()
    assert state_after_entry._player_finished_round[0] is True
    assert state_after_entry.next_fantasyland_status[0] is True
    assert state_after_entry.fantasyland_cards_to_deal[0] == 16

    # Симулируем раунд ФЛ и проверяем Re-Fantasy
    state_fl = GameState(fantasyland_status=[True, False], fantasyland_cards_to_deal=[16, 0])
    state_fl.start_new_round(0)
    fl_hand_stay = hand(['Ah','Ad','Ac','Ks','Kd','Qs','Qd','Js','Jd','Ts','Td','2s','2d','2h','3c','3d']) # 16 карт
    state_fl.fantasyland_hands[0] = fl_hand_stay
    placement_stay = {'top': fl_hand_stay[0:3], 'middle': fl_hand_stay[3:8], 'bottom': fl_hand_stay[8:13]}
    discarded_stay = fl_hand_stay[13:] # Сбрасываем 3 карты

    state_after_fl_stay = state_fl.apply_fantasyland_placement(0, placement_stay, discarded_stay)
    assert state_after_fl_stay.boards[0].is_complete()
    assert state_after_fl_stay._player_finished_round[0] is True
    assert state_after_fl_stay.next_fantasyland_status[0] is True
    assert state_after_fl_stay.fantasyland_cards_to_deal[0] == 14

# --- Тесты сериализации и копирования ---

def test_gamestate_serialization():
    """Тестирует to_dict и from_dict."""
    state = GameState(dealer_idx=1, street=3)
    state.start_new_round(1)
    hand0 = state.current_hands[0]
    action0 = (tuple(sorted([(hand0[i], 'bottom', i) for i in range(5)])), tuple())
    state = state.apply_action(0, action0)
    state = state.advance_state()
    hand1 = state.current_hands[1]
    action1 = (tuple(sorted([(hand1[i], 'middle', i) for i in range(5)])), tuple())
    state = state.apply_action(1, action1)
    state = state.advance_state()

    state_dict = state.to_dict()
    assert isinstance(state_dict, dict)
    assert isinstance(state_dict['boards'], list)
    assert isinstance(state_dict['fantasyland_status'], list)
    assert isinstance(state_dict['street'], int)
    assert 'deck_remaining' in state_dict

    restored_state = GameState.from_dict(state_dict)

    assert state.get_state_representation() == restored_state.get_state_representation()
    assert state.street == restored_state.street
    assert state.dealer_idx == restored_state.dealer_idx
    assert state.boards[0].get_total_cards() == restored_state.boards[0].get_total_cards()
    assert len(state.deck) == len(restored_state.deck)
    assert state.deck.cards == restored_state.deck.cards

def test_gamestate_serialization_consistency():
    """Проверяет консистентность данных при сериализации/десериализации."""
    state = GameState(dealer_idx=1)
    state.start_new_round(1)
    hand0 = state.current_hands[0]
    action0 = (tuple(sorted([(hand0[i], 'bottom', i) for i in range(5)])), tuple())
    state = state.apply_action(0, action0)
    state = state.advance_state()

    state_dict = state.to_dict()
    restored_state = GameState.from_dict(state_dict)

    all_restored_cards = set()
    for b in restored_state.boards:
        for r in b.rows.values():
            all_restored_cards.update(c for c in r if c is not None)
    for h in restored_state.current_hands.values():
        if h: all_restored_cards.update(h)
    for h in restored_state.fantasyland_hands:
        if h: all_restored_cards.update(h)
    for d in restored_state.private_discard:
        all_restored_cards.update(d)
    all_restored_cards.update(restored_state.deck.cards)

    assert all_restored_cards == Deck.FULL_DECK_CARDS
    assert len(all_restored_cards) == 52


def test_gamestate_copy():
    """Тестирует метод copy()."""
    state1 = GameState(dealer_idx=1)
    state1.start_new_round(1)
    hand0 = state1.current_hands[0]
    action0 = (tuple(sorted([(hand0[i], 'bottom', i) for i in range(5)])), tuple())
    state1 = state1.apply_action(0, action0)

    state2 = state1.copy()

    assert state1 is not state2
    assert state1.boards is not state2.boards
    assert state1.boards[0] is not state2.boards[0]
    assert state1.deck is not state2.deck
    assert state1.current_hands is not state2.current_hands
    assert state1.private_discard is not state2.private_discard
    assert state1.get_state_representation() == state2.get_state_representation()

    state2_advanced = state2.advance_state()
    hand_p1 = state2_advanced.current_hands[1]
    action_p1 = (tuple(sorted([(hand_p1[i], 'middle', i) for i in range(5)])), tuple())
    state2_final = state2_advanced.apply_action(1, action_p1)

    assert state1.street == 1
    assert state1.boards[1].get_total_cards() == 0
    assert state1.current_hands.get(1) is None
    assert len(state1.private_discard[1]) == 0

    assert state2_final.street == 1
    assert state2_final.boards[1].get_total_cards() == 5
    assert state2_final.current_hands.get(1) is None
    assert len(state2_final.private_discard[1]) == 0

# --- Тесты get_legal_actions ---
def test_gamestate_get_legal_actions_various_states():
    """Тестирует get_legal_actions_for_player в разных состояниях."""
    # Улица 1, ход P1
    state1 = GameState(dealer_idx=0)
    state1.start_new_round(0)
    assert state1.get_player_to_move() == 1
    actions1 = state1.get_legal_actions_for_player(1)
    assert len(actions1) > 0
    assert isinstance(actions1[0], tuple)
    assert len(actions1[0]) == 2
    assert isinstance(actions1[0][0], tuple)
    assert len(actions1[0][0]) == 5
    assert actions1[0][1] == tuple()

    # Улица 2, ход P0
    state2 = copy.deepcopy(state1)
    state2.street = 2
    state2._internal_current_player_idx = 0
    state2.current_hands[0] = hand(['Ac','Kc','Qc'])
    state2.current_hands[1] = None
    state2._player_acted_this_street = [False, False]
    assert state2.get_player_to_move() == 0
    actions2 = state2.get_legal_actions_for_player(0)
    assert len(actions2) > 0
    assert isinstance(actions2[0], tuple)
    # --- ИСПРАВЛЕНО: Проверка формата Pineapple ---
    assert len(actions2[0]) == 2 # (placements_tuple, discard_card)
    assert isinstance(actions2[0][0], tuple) # placements_tuple
    assert len(actions2[0][0]) == 2 # 2 placements
    assert isinstance(actions2[0][0][0], tuple) # placement1
    assert isinstance(actions2[0][0][1], tuple) # placement2
    assert isinstance(actions2[0][1], int) # discard_card

    # Фантазия, ход P0
    state3 = GameState(fantasyland_status=[True, False], fantasyland_cards_to_deal=[14,0])
    state3.start_new_round(1)
    assert state3.get_player_to_move() == 0
    actions3 = state3.get_legal_actions_for_player(0)
    assert len(actions3) == 1
    assert actions3[0][0] == "FANTASYLAND_INPUT"
    assert isinstance(actions3[0][1], tuple)
    assert len(actions3[0][1]) == 14

    # Игрок закончил
    state4 = copy.deepcopy(state1)
    state4._player_finished_round[1] = True
    assert state4.get_player_to_move() == -1
    assert state4.get_legal_actions_for_player(1) == []

    # Не ход игрока
    assert state1.get_legal_actions_for_player(0) == []
