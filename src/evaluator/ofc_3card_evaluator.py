# src/evaluator/ofc_3card_evaluator.py v1.1
# -*- coding: utf-8 -*-
"""
Функция для оценки 3-карточной руки OFC (верхний бокс).
Использует предрасчитанную таблицу поиска.
"""

import traceback
from typing import Tuple, List

# --- ИСПРАВЛЕНО: Импортируем таблицу и Card из правильных мест ---
from .ofc_3card_lookup import three_card_lookup
# Используем прямой импорт из src пакета
from src.card import Card as RootCardUtils, INVALID_CARD, card_to_str


def evaluate_3_card_ofc(card1: int, card2: int, card3: int) -> Tuple[int, str, str]:
    """
    Оценивает 3-карточную руку по правилам OFC, используя предрасчитанную таблицу.
    Карты должны быть целочисленными представлениями из src.card.

    Args:
        card1 (int): Первая карта.
        card2 (int): Вторая карта.
        card3 (int): Третья карта.

    Returns:
        Tuple[int, str, str]: Кортеж (rank, type_string, rank_string).
                              Меньший ранг соответствует более сильной руке.
                              rank: 1 (AAA) - 455 (432 разномастные).
                              type_string: 'Trips', 'Pair', 'High Card'.
                              rank_string: Строка рангов карт (отсортированная, например, 'AAK', 'KQJ').

    Raises:
        ValueError: Если переданы некорректные карты (None, INVALID_CARD, дубликаты)
                    или их количество не равно 3.
        TypeError: Если переданные значения не являются целыми числами.
    """
    ranks: List[int] = []
    input_cards: List[int] = [card1, card2, card3]
    valid_cards: List[int] = [] # Собираем валидные карты для проверки дубликатов

    # Проверка типов и валидности карт
    for i, card_int in enumerate(input_cards):
        card_idx = i + 1
        if not isinstance(card_int, int):
            raise TypeError(f"Card {card_idx} is not an integer (received type {type(card_int)}). Hand: {input_cards}")
        if card_int == INVALID_CARD:
             raise ValueError(f"Card {card_idx} is invalid (INVALID_CARD). Hand: {input_cards}")
        if card_int <= 0: # Дополнительная проверка на отрицательные и ноль
             raise ValueError(f"Card {card_idx} has invalid value {card_int}. Hand: {input_cards}")

        try:
            rank_int = RootCardUtils.get_rank_int(card_int)
            # Проверяем, что ранг находится в допустимом диапазоне 0-12
            if 0 <= rank_int <= 12:
                ranks.append(rank_int)
                valid_cards.append(card_int) # Добавляем валидную карту
            else:
                # Эта ошибка маловероятна, если card_int валиден, но оставляем проверку
                raise ValueError(f"Invalid rank {rank_int} extracted from card int: {card_int} ({card_to_str(card_int)})")
        except Exception as e:
            # Перехватываем другие возможные ошибки при обработке карты
            raise ValueError(f"Error processing card int {card_int} ({card_to_str(card_int)}): {e}")

    # Проверка на дубликаты карт
    if len(valid_cards) != len(set(valid_cards)):
        raise ValueError(f"Duplicate cards found in 3-card hand: {[card_to_str(c) for c in valid_cards]}")

    # Сортируем ранги по убыванию для ключа поиска
    # Кортеж используется как ключ словаря, так как он неизменяемый
    lookup_key = tuple(sorted(ranks, reverse=True))

    # Ищем результат в таблице
    result = three_card_lookup.get(lookup_key)

    if result is None:
        # Эта ситуация не должна возникать, если таблица полная и карты валидные
        raise ValueError(f"Combination not found for rank key: {lookup_key} (original ranks: {ranks}, cards: {[card_to_str(c) for c in valid_cards]})")

    # Возвращаем найденный кортеж (rank, type_string, rank_string)
    return result

# Пример использования внутри модуля (для тестирования)
if __name__ == '__main__':
    # Тесты используют RootCardUtils, который теперь импортируется из src.card
    print("--- Тестирование evaluate_3_card_ofc ---")
    test_hands_str = [
        ('Ah', 'Ad', 'As'), ('Qh', 'Qs', '2d'), ('Kd', '5s', '2h'),
        ('6c', '6d', 'Ts'), ('2h', '2d', '2s'), ('Jd', 'Th', '9s'),
        ('5h', '3d', '2c'), ('Ac', 'Kc', 'Qc') # Добавим старшую карту
    ]
    results = []
    expected_ranks = [1, 40, 287, 114, 13, 336, 455, 170] # Ожидаемые ранги для тестов

    for i, hand_str in enumerate(test_hands_str):
        try:
            # Преобразуем строки в int
            hand_int = tuple(RootCardUtils.from_str(c) for c in hand_str)
            rank, type_str, rank_str = evaluate_3_card_ofc(*hand_int)
            print(f"Тест {i+1}: {hand_str} -> Ранг: {rank}, Тип: {type_str}, Строка: {rank_str}")
            results.append(rank)
        except Exception as e:
            print(f"Ошибка при тесте {i+1} {hand_str}: {e}")
            traceback.print_exc()

    # Тест на дубликат
    print("\n--- Тест на дубликат ---")
    try:
        hand_dup_int = (RootCardUtils.from_str('As'), RootCardUtils.from_str('As'), RootCardUtils.from_str('Ks'))
        evaluate_3_card_ofc(*hand_dup_int)
        print("ОШИБКА: Дубликат не вызвал ValueError!")
    except ValueError as e:
        print(f"Успех: Дубликат вызвал ошибку: {e}")
    except Exception as e:
        print(f"ОШИБКА: Дубликат вызвал неожиданную ошибку: {e}")

    # Тест на невалидную карту
    print("\n--- Тест на невалидную карту ---")
    try:
        hand_inv_int = (RootCardUtils.from_str('As'), INVALID_CARD, RootCardUtils.from_str('Ks'))
        evaluate_3_card_ofc(*hand_inv_int)
        print("ОШИБКА: INVALID_CARD не вызвал ValueError!")
    except ValueError as e:
        print(f"Успех: INVALID_CARD вызвал ошибку: {e}")
    except Exception as e:
        print(f"ОШИБКА: INVALID_CARD вызвал неожиданную ошибку: {e}")


    if results:
        print("\n--- Проверка результатов ---")
        print(f"Полученные ранги: {results}")
        print(f"Ожидаемые ранги: {expected_ranks}")
        correct_count = sum(1 for i, r in enumerate(results) if i < len(expected_ranks) and r == expected_ranks[i])
        print(f"Совпадений рангов: {correct_count} из {len(expected_ranks)}")

        if correct_count == len(expected_ranks):
             print("\nФункция оценки 3-карточных рук работает корректно.")
        else:
             print("\nВНИМАНИЕ: Обнаружены несоответствия в рангах!")
