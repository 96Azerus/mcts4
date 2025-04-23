# tests/test_scoring.py v1.6
"""
Unit-тесты для модуля src.scoring.
"""

import pytest

# Импорты из src пакета
from src.card import (
    Card as CardUtils, card_from_str, RANK_MAP, STR_RANKS,
    INVALID_CARD, INT_RANK_TO_CHAR, CARD_PLACEHOLDER
)
from src.scoring import (
    get_hand_rank_safe, get_row_royalty, check_board_foul,
    get_fantasyland_entry_cards, check_fantasyland_stay,
    calculate_headsup_score,
    RANK_CLASS_HIGH_CARD, RANK_CLASS_PAIR, RANK_CLASS_TWO_PAIR,
    RANK_CLASS_TRIPS, RANK_CLASS_STRAIGHT, RANK_CLASS_FLUSH,
    RANK_CLASS_FULL_HOUSE, RANK_CLASS_QUADS, RANK_CLASS_STRAIGHT_FLUSH,
    ROYALTY_TOP_PAIRS, ROYALTY_TOP_TRIPS,
    ROYALTY_MIDDLE_POINTS, ROYALTY_BOTTOM_POINTS,
    ROYALTY_MIDDLE_POINTS_RF, ROYALTY_BOTTOM_POINTS_RF,
    WORST_RANK # Импортируем WORST_RANK
)
from src.board import PlayerBoard # Импорт для хелпера create_board

# Хелпер для создания рук
def hand(card_strs):
    return [card_from_str(s) if isinstance(s, str) and s and s != CARD_PLACEHOLDER else None for s in card_strs]

# --- Тесты get_hand_rank_safe ---
@pytest.mark.parametrize("cards_str, expected_len, is_3card", [
    (['As', 'Ks', 'Qs'], 3, True),
    (['2d', '3c', '4h'], 3, True),
    (['7h', '7d', 'Ac'], 3, True),
    (['As', 'Ks', 'Qs', 'Js', 'Ts'], 5, False), # RF
    (['7h', '7d', '7c', '7s', 'Ad'], 5, False), # Quads
    (['6h', '6d', '6c', 'Ks', 'Kd'], 5, False), # Full House
    (['As', 'Qs', '8s', '5s', '3s'], 5, False), # Flush
    (['5d', '4c', '3h', '2s', 'Ad'], 5, False), # Straight (Wheel)
    (['Ac', 'Ad', 'Ah', 'Ks', 'Qd'], 5, False), # Trips
    (['Ac', 'Ad', 'Kc', 'Kd', '2s'], 5, False), # Two Pair
    (['Ac', 'Ad', 'Ks', 'Qd', 'Jc'], 5, False), # Pair
    (['Ac', 'Kc', 'Qs', 'Js', '9d'], 5, False), # High Card
])
def test_get_hand_rank_safe_valid(cards_str, expected_len, is_3card):
    cards = hand(cards_str)
    rank = get_hand_rank_safe(cards)
    assert isinstance(rank, int)
    if is_3card:
        assert 1 <= rank <= 455
    else:
        assert 1 <= rank <= RANK_CLASS_HIGH_CARD
    assert rank > 0

@pytest.mark.parametrize("cards_str, expected_rank", [
    (['As', 'Ks', None], WORST_RANK),
    ([None, '2c', '3d'], WORST_RANK),
    (['As', 'Ks', 'Qs', 'Js', None], WORST_RANK),
    (['As', None, 'Qs', None, 'Ts'], WORST_RANK),
    ([None, None, None, None, None], WORST_RANK),
    ([None, None, None], WORST_RANK),
])
def test_get_hand_rank_safe_incomplete(cards_str, expected_rank):
    cards = hand(cards_str)
    rank = get_hand_rank_safe(cards)
    assert rank == expected_rank

def test_get_hand_rank_safe_invalid_input():
    """Тестирует get_hand_rank_safe с невалидным вводом (неверная длина)."""
    assert get_hand_rank_safe(hand(['As', 'Ks'])) == WORST_RANK
    assert get_hand_rank_safe(hand(['As', 'Ks', 'Qs', 'Js'])) == WORST_RANK
    assert get_hand_rank_safe([]) == WORST_RANK
    assert get_hand_rank_safe(hand(['As'])) == WORST_RANK
    assert get_hand_rank_safe(None) == WORST_RANK # type: ignore
    assert get_hand_rank_safe("not a list") == WORST_RANK # type: ignore

def test_get_hand_rank_safe_duplicates():
    """Тестирует get_hand_rank_safe с дубликатами."""
    assert get_hand_rank_safe(hand(['As', 'As', 'Ks'])) == WORST_RANK
    assert get_hand_rank_safe(hand(['As', 'Ks', 'Qs', 'Js', 'As'])) == WORST_RANK

# --- Тесты get_row_royalty ---
# Top Row Royalty
@pytest.mark.parametrize("cards_str, expected_royalty", [
    (['Ah', 'Ad', 'Ac'], ROYALTY_TOP_TRIPS[RANK_MAP['A']]), # 22
    (['Kh', 'Kd', 'Kc'], ROYALTY_TOP_TRIPS[RANK_MAP['K']]), # 21
    (['6h', '6d', '6c'], ROYALTY_TOP_TRIPS[RANK_MAP['6']]), # 14
    (['2h', '2d', '2c'], ROYALTY_TOP_TRIPS[RANK_MAP['2']]), # 10
    (['Ah', 'Ad', 'Kc'], ROYALTY_TOP_PAIRS[RANK_MAP['A']]), # 9
    (['Qh', 'Qd', '2c'], ROYALTY_TOP_PAIRS[RANK_MAP['Q']]), # 7
    # --- ИСПРАВЛЕНО: Ожидаемые значения ---
    (['6h', '6d', 'Ac'], ROYALTY_TOP_PAIRS[RANK_MAP['6']]), # 1
    (['5h', '5d', 'Ac'], 0), # Пара ниже 66
    (['Ah', 'Kc', 'Qd'], 0), # Старшая карта
    (['Ah', 'Ad', None], 0), # Неполная рука
    (['Ah', 'Ah', 'Kc'], 0), # Дубликат
])
def test_get_row_royalty_top(cards_str, expected_royalty):
    cards_int = hand(cards_str)
    assert get_row_royalty(cards_int, "top") == expected_royalty

# Middle Row Royalty
@pytest.mark.parametrize("cards_str, expected_royalty", [
    (['As', 'Ks', 'Qs', 'Js', 'Ts'], ROYALTY_MIDDLE_POINTS_RF), # 50
    (['9d', '8d', '7d', '6d', '5d'], ROYALTY_MIDDLE_POINTS["Straight Flush"]), # 30
    (['Ac', 'Ad', 'Ah', 'As', '2c'], ROYALTY_MIDDLE_POINTS["Four of a Kind"]), # 20
    (['Kc', 'Kd', 'Kh', 'Qc', 'Qs'], ROYALTY_MIDDLE_POINTS["Full House"]), # 12
    (['As', 'Qs', '8s', '5s', '3s'], ROYALTY_MIDDLE_POINTS["Flush"]), # 8
    (['Ad', 'Kc', 'Qh', 'Js', 'Td'], ROYALTY_MIDDLE_POINTS["Straight"]), # 4
    (['Ac', 'Ad', 'Ah', 'Ks', 'Qd'], ROYALTY_MIDDLE_POINTS["Three of a Kind"]), # 2
    (['Ac', 'Ad', 'Kc', 'Kd', '2s'], 0), # Two Pair
    (['Ac', 'Ad', 'Ks', 'Qd', 'Jc'], 0), # Pair
    (['Ac', 'Kc', 'Qs', 'Js', '9d'], 0), # High Card
    (['Ac', 'Ad', 'Ah', 'Ks', None], 0), # Incomplete
    (['Ac', 'Ac', 'Ah', 'Ks', 'Qd'], 0), # Duplicate
])
def test_get_row_royalty_middle(cards_str, expected_royalty):
    cards_int = hand(cards_str)
    assert get_row_royalty(cards_int, "middle") == expected_royalty

# Bottom Row Royalty
@pytest.mark.parametrize("cards_str, expected_royalty", [
    (['As', 'Ks', 'Qs', 'Js', 'Ts'], ROYALTY_BOTTOM_POINTS_RF), # 25
    (['9d', '8d', '7d', '6d', '5d'], ROYALTY_BOTTOM_POINTS["Straight Flush"]), # 15
    (['Ac', 'Ad', 'Ah', 'As', '2c'], ROYALTY_BOTTOM_POINTS["Four of a Kind"]), # 10
    (['Kc', 'Kd', 'Kh', 'Qc', 'Qs'], ROYALTY_BOTTOM_POINTS["Full House"]), # 6
    (['As', 'Qs', '8s', '5s', '3s'], ROYALTY_BOTTOM_POINTS["Flush"]), # 4
    (['Ad', 'Kc', 'Qh', 'Js', 'Td'], ROYALTY_BOTTOM_POINTS["Straight"]), # 2
    (['Ac', 'Ad', 'Ah', 'Ks', 'Qd'], 0), # Trips - no royalty on bottom
    (['Ac', 'Ad', 'Kc', 'Kd', '2s'], 0), # Two Pair
    (['Ac', 'Ad', 'Ks', 'Qd', 'Jc'], 0), # Pair
    (['Ac', 'Kc', 'Qs', 'Js', '9d'], 0), # High Card
    (['Ac', 'Ad', 'Ah', 'Ks', None], 0), # Incomplete
    (['Ac', 'Ac', 'Ah', 'Ks', 'Qd'], 0), # Duplicate
])
def test_get_row_royalty_bottom(cards_str, expected_royalty):
    cards_int = hand(cards_str)
    assert get_row_royalty(cards_int, "bottom") == expected_royalty

# --- Тесты check_board_foul ---
def test_check_board_foul_valid():
    top = hand(['Ah', 'Kc', 'Qd']) # High Card
    middle = hand(['2s', '2d', '3c', '4h', '5s']) # Pair
    bottom = hand(['7h', '7d', '7c', 'As', 'Ks']) # Trips
    # --- ИСПРАВЛЕНО: Ожидаем False (не фол) ---
    assert not check_board_foul(top, middle, bottom)

def test_check_board_foul_invalid():
    # Middle > Top
    top = hand(['2h', '3c', '4d']) # High Card
    middle = hand(['As', 'Ad', 'Ac', 'Ks', 'Kd']) # Trips
    bottom = hand(['Qs', 'Qd', 'Jc', 'Jh', '2s']) # Two Pair
    assert check_board_foul(top, middle, bottom)
    # Bottom > Middle
    top = hand(['Ah', 'Ad', 'Ac']) # Trips
    middle = hand(['Ks', 'Kd', 'Qc', 'Qd', '2s']) # Two Pair
    # --- ИСПРАВЛЕНО: Убран дубликат As, Ks ---
    bottom = hand(['Th', 'Jh', 'Qh', 'Kh', 'Ah']) # Royal Flush
    # --- ИСПРАВЛЕНО: Ожидаем True (фол) ---
    assert check_board_foul(top, middle, bottom)

def test_check_board_foul_incomplete():
    top = hand(['Ah', 'Ad', 'Ac'])
    middle = hand(['Ks', 'Kd', 'Qc', 'Qd', None]) # Неполный ряд
    bottom = hand(['As', 'Ks', 'Qs', 'Js', 'Ts'])
    assert not check_board_foul(top, middle, bottom) # Неполная доска не фол

def test_check_board_foul_duplicates():
    """Тест фола с дубликатами (должен вернуть False, т.к. доска невалидна)."""
    top = hand(['Ah', 'Ad', 'Ac'])
    middle = hand(['Ks', 'Kd', 'Qc', 'Qd', 'Ah']) # Дубликат
    bottom = hand(['As', 'Ks', 'Qs', 'Js', 'Ts'])
    assert not check_board_foul(top, middle, bottom)

# --- Тесты get_fantasyland_entry_cards (Progressive) ---
@pytest.mark.parametrize("top_hand_str, expected_cards", [
    (['Ah', 'Ad', 'Ac'], 17), # Trips A -> 17
    (['2h', '2d', '2c'], 17), # Trips 2 -> 17
    (['Ah', 'Ad', 'Kc'], 16), # Pair A -> 16
    # --- ИСПРАВЛЕНО: Ожидаемые значения ---
    (['Kh', 'Kd', 'Ac'], 15), # Pair K -> 15
    (['Qh', 'Qd', 'Ac'], 14), # Pair Q -> 14
    (['Jh', 'Jd', 'Ac'], 0),  # Pair J -> 0
    (['6h', '6d', 'Ac'], 0),  # Pair 6 -> 0
    (['Ah', 'Kc', 'Qd'], 0),  # High Card -> 0
    (['Ah', 'Ad', None], 0),  # Incomplete -> 0
    (['Ah', 'Ah', 'Kc'], 0),  # Duplicate -> 0
])
def test_get_fantasyland_entry_cards(top_hand_str, expected_cards):
    top_hand_int = hand(top_hand_str)
    assert get_fantasyland_entry_cards(top_hand_int) == expected_cards

# --- Тесты check_fantasyland_stay ---
@pytest.mark.parametrize("top_str, middle_str, bottom_str, expected_stay", [
    # Stay: Trips on Top
    (['Ah', 'Ad', 'Ac'], ['Ks', 'Kd', 'Qc', 'Qd', '2s'], ['As', 'Kh', 'Qs', 'Js', 'Ts'], True),
    # Stay: Quads on Bottom
    (['Ah', 'Kc', 'Qd'], ['2s', '2d', '3c', '4h', '5s'], ['7h', '7d', '7c', '7s', 'Ad'], True),
    # Stay: Straight Flush on Bottom
    (['Ah', 'Kc', 'Qd'], ['2s', '2d', '3c', '4h', '5s'], ['9d', '8d', '7d', '6d', '5d'], True),
    # No Stay: High card top, Straight bottom
    (['Ah', 'Kc', 'Qd'], ['2s', '2d', '3c', '4h', '5s'], ['As', 'Ks', 'Qs', 'Js', '9d'], False),
    # No Stay: Foul (Middle > Top) - Эта доска теперь не фол, но и не stay
    (['Ah', 'Ad', '2c'], ['As', 'Ks', 'Qs', 'Js', 'Ts'], ['Ks', 'Kd', 'Qc', 'Qd', '2s'], False),
    # No Stay: Foul (Bottom > Middle) - Эта доска теперь фол
    (['Ah', 'Kc', 'Qd'], ['As', 'Ks', 'Qs', 'Js', 'Ts'], ['7h', '7d', '7c', '7s', 'Ad'], False),
    # No Stay: Incomplete board
    (['Ah', 'Ad', 'Ac'], ['Ks', 'Kd', 'Qc', 'Qd', None], ['As', 'Ks', 'Qs', 'Js', 'Ts'], False),
    # No Stay: Duplicate cards
    (['Ah', 'Ad', 'Ac'], ['Ks', 'Kd', 'Qc', 'Qd', 'Ah'], ['As', 'Ks', 'Qs', 'Js', 'Ts'], False),
])
def test_check_fantasyland_stay(top_str, middle_str, bottom_str, expected_stay):
    top = hand(top_str)
    middle = hand(middle_str)
    bottom = hand(bottom_str)
    assert check_fantasyland_stay(top, middle, bottom) == expected_stay

# --- Тесты calculate_headsup_score ---
# Хелпер create_board
def create_board(top_s, mid_s, bot_s):
    board = PlayerBoard()
    try:
        board.set_full_board(hand(top_s), hand(mid_s), hand(bot_s))
    except ValueError as e:
        print(f"Warning: set_full_board failed in test setup: {e}. Creating board manually.")
        board.rows['top'] = (hand(top_s) + [None]*3)[:3]
        board.rows['middle'] = (hand(mid_s) + [None]*5)[:5]
        board.rows['bottom'] = (hand(bot_s) + [None]*5)[:5]
        board._cards_placed = sum(1 for r in board.rows.values() for c in r if c is not None and c != INVALID_CARD)
        board._is_complete = board._cards_placed == 13
        if board._is_complete:
             board.check_and_set_foul()
    return board

# Данные для тестов calculate_headsup_score
board_p1_scoop = create_board(['Ah', 'Ad', 'Kc'], ['7h', '8h', '9h', 'Th', 'Jh'], ['As', 'Ks', 'Qs', 'Js', 'Ts']) # R=64
board_p2_basic = create_board(['Kh', 'Qd', '2c'], ['Ac', 'Kd', 'Qh', 'Js', '9d'], ['Tc', 'Td', 'Th', '2s', '3s']) # R=0
test_score_p1_scoop_data = (board_p1_scoop, board_p2_basic, 70) # P1 wins 3 lines (+3), scoop (+3), R diff (+64) = 70

board_p1_basic = create_board(['Kh', 'Qd', '2c'], ['Ac', 'Kd', 'Qh', 'Js', '9d'], ['Tc', 'Td', 'Th', '2s', '3s']) # R=0
board_p2_scoop = create_board(['Qh', 'Qd', 'Ac'], ['Kc', 'Kd', 'Kh', '2c', '2s'], ['Ad', 'Ac', 'Ah', 'As', '3c']) # R=29
test_score_p2_scoop_data = (board_p1_basic, board_p2_scoop, -35) # P2 wins 3 lines (-3), scoop (-3), R diff (-29) = -35

board_p1_mix = create_board(['Ah', 'Ad', 'Kc'], ['2h', '3h', '4h', '5h', '7h'], ['6c', '6d', '6h', 'Ks', 'Kd']) # R=23
board_p2_mix = create_board(['Kh', 'Qd', '2c'], ['Ac', 'Kd', 'Qh', 'Js', '9d'], ['7c', '7d', '7h', '7s', 'Ad']) # R=10
# P1 vs P2: Top(AA>K, +1), Mid(Flush>A, +1), Bot(FH<Quads, -1). Line score = +1. Total = +1 + (23-10) = 14.
test_score_mix_data = (board_p1_mix, board_p2_mix, 14) # --- ИСПРАВЛЕНО ОЖИДАНИЕ ---

board_p1_foul = create_board(['Ah', 'Ad', 'Ac'], ['Ks', 'Kd', 'Qc', 'Qd', '2s'], ['As', 'Kh', 'Qs', 'Js', 'Ts']) # Foul, R=0
board_p2_ok = create_board(['Kh', 'Qd', '2c'], ['Ac', 'Kd', 'Qh', 'Js', '9d'], ['Tc', 'Td', 'Th', '2s', '3s']) # R=0
test_score_p1_foul_data = (board_p1_foul, board_p2_ok, -6) # P1 foul, P2 R=0 -> P0 получает -6

board_p1_ok = create_board(['Ah', 'Ad', 'Kc'], ['7h', '8h', '9h', 'Th', 'Jh'], ['As', 'Ks', 'Qs', 'Js', 'Ts']) # R=64
board_p2_foul = create_board(['2h', '2d', '2c'], ['3s', '3d', '4c', '4d', '5s'], ['Ah', 'Kh', 'Qh', 'Jh', '9h']) # Foul, R=0
test_score_p2_foul_data = (board_p1_ok, board_p2_foul, 70) # P2 foul, P1 R=64 -> P0 получает 6 + 64 = 70

board_p1_foul_too = create_board(['Ah', 'Ad', 'Ac'], ['Ks','Kd','Qc','Qd','2s'], ['As','Kh','Qs','Js','Ts']) # Foul
board_p2_foul_too = create_board(['2h', '2d', '2c'], ['3s','3d','4c','4d','5s'], ['Ah','Kh','Qh','Jh','9h']) # Foul
test_score_both_foul_data = (board_p1_foul_too, board_p2_foul_too, 0) # Both foul -> 0

@pytest.mark.parametrize("board1, board2, expected_score", [
    test_score_p1_scoop_data,
    test_score_p2_scoop_data,
    test_score_mix_data,
    test_score_p1_foul_data,
    test_score_p2_foul_data,
    test_score_both_foul_data,
])
def test_calculate_headsup_score(board1, board2, expected_score):
    """Тестирует расчет итогового счета между двумя игроками."""
    assert calculate_headsup_score(board1, board2) == expected_score
    # Проверяем симметрию (смена мест игроков меняет знак счета)
    assert calculate_headsup_score(board2, board1) == -expected_score
