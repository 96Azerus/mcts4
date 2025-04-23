# src/evaluator/ofc_5card_lookup.py v1.2
# -*- coding: utf-8 -*-
"""
Генерация таблиц поиска для 5-карточного эвалуатора OFC.
Использует вариант алгоритма Cactus Kev с простыми числами.
"""

import itertools
import traceback
import sys
import logging # Используем logging
from typing import Dict, List, Generator

# Импорты из src пакета
try:
    from src.card import Card as CardUtils, INT_RANKS, PRIMES
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import from src.card in ofc_5card_lookup.py: {e}", file=sys.stderr)
    # Заглушка
    class CardUtils:
        PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
        INT_RANKS = range(13)
        @staticmethod
        def prime_product_from_rankbits(rankbits): return 1
        @staticmethod
        def prime_product_from_hand(card_ints): return 1
    INT_RANKS = range(13)
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
    raise ImportError("Failed to import Card utilities, cannot initialize 5-card lookup table.") from e

# Получаем логгер
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): logger.setLevel(logging.WARNING)


class LookupTable:
    """
    Создает и хранит таблицы поиска для быстрой оценки 5-карточных рук.

    Таблицы:
    - flush_lookup: Отображает произведение простых чисел для рангов флеша
                    в ранг руки [1 (Royal Flush) .. 1599].
    - unsuited_lookup: Отображает произведение простых чисел для неулучшенных рук
                       (не флеш) в ранг руки [11 (Four Aces) .. 7462 (7-5-4-3-2)].

    Ранги: Меньшее значение соответствует более сильной руке.
    """
    # Константы рангов (без изменений)
    MAX_STRAIGHT_FLUSH: int = 10
    MAX_FOUR_OF_A_KIND: int = 166
    MAX_FULL_HOUSE: int = 322
    MAX_FLUSH: int = 1599
    MAX_STRAIGHT: int = 1609
    MAX_THREE_OF_A_KIND: int = 2467
    MAX_TWO_PAIR: int = 3325
    MAX_PAIR: int = 6185
    MAX_HIGH_CARD: int = 7462

    MAX_TO_RANK_CLASS: Dict[int, int] = {
        MAX_STRAIGHT_FLUSH: 1, MAX_FOUR_OF_A_KIND: 2, MAX_FULL_HOUSE: 3,
        MAX_FLUSH: 4, MAX_STRAIGHT: 5, MAX_THREE_OF_A_KIND: 6,
        MAX_TWO_PAIR: 7, MAX_PAIR: 8, MAX_HIGH_CARD: 9
    }
    RANK_CLASS_TO_STRING: Dict[int, str] = {
        1: "Straight Flush", 2: "Four of a Kind", 3: "Full House", 4: "Flush",
        5: "Straight", 6: "Three of a Kind", 7: "Two Pair", 8: "Pair", 9: "High Card"
    }

    def __init__(self):
        """ Инициализирует и вычисляет таблицы поиска. """
        self.flush_lookup: Dict[int, int] = {}
        self.unsuited_lookup: Dict[int, int] = {}
        logger.info("Initializing 5-card lookup tables...")
        try:
            self._calculate_flushes()
            self._calculate_multiples()
            logger.info(f"Lookup tables initialized. Flush: {len(self.flush_lookup)}, Unsuited: {len(self.unsuited_lookup)}")
        except Exception as e:
             logger.critical(f"Error during lookup table calculation: {e}", exc_info=True)
             raise # Перевыбрасываем исключение, чтобы остановить приложение

    def _calculate_flushes(self):
        """ Вычисляет ранги для стрит-флешей и обычных флешей. """
        # --- ИСПРАВЛЕНО: Определяем битовые маски стрит-флешей в правильном порядке старшинства ---
        straight_flushes_rank_bits: List[int] = [
            0b1111100000000, # A, K, Q, J, T (Royal Flush) - Rank 1
            0b0111110000000, # K, Q, J, T, 9              - Rank 2
            0b0011111000000, # Q, J, T, 9, 8              - Rank 3
            0b0001111100000, # J, T, 9, 8, 7              - Rank 4
            0b0000111110000, # T, 9, 8, 7, 6              - Rank 5
            0b0000011111000, # 9, 8, 7, 6, 5              - Rank 6
            0b0000001111100, # 8, 7, 6, 5, 4              - Rank 7
            0b0000000111110, # 7, 6, 5, 4, 3              - Rank 8
            0b0000000011111, # 6, 5, 4, 3, 2              - Rank 9
            0b1000000001111, # A, 5, 4, 3, 2 (Wheel)      - Rank 10
        ]
        # Генерируем все возможные 5-карточные комбинации рангов (для флешей)
        all_flush_rank_bits: List[int] = []
        start_bits = (1 << 5) - 1 # 0b11111 (2,3,4,5,6)
        all_flush_rank_bits.append(start_bits)
        gen = self._get_lexographically_next_bit_sequence(start_bits)
        try:
            while True:
                rank_bits = next(gen)
                all_flush_rank_bits.append(rank_bits)
        except StopIteration: pass
        except Exception as e:
            logger.error(f"Error during bit sequence generation: {e}", exc_info=True)
            # Продолжаем с тем, что есть, но таблица будет неполной

        expected_combinations = 1287 # C(13, 5)
        if len(all_flush_rank_bits) != expected_combinations:
            logger.critical(f"Generated {len(all_flush_rank_bits)} flush rank combinations, expected {expected_combinations}. Lookup table will be incomplete!")
            # raise RuntimeError("Failed to generate the correct number of flush combinations.")

        # Разделяем стрит-флеши и обычные флеши
        straight_flush_set = set(straight_flushes_rank_bits)
        # Сортируем обычные флеши по убыванию (для присвоения рангов от лучшего к худшему)
        normal_flush_rank_bits = sorted([rb for rb in all_flush_rank_bits if rb not in straight_flush_set], reverse=True)

        # Присваиваем ранги стрит-флешам (1-10)
        rank = 1
        # --- ИСПРАВЛЕНО: Используем предопределенный порядок straight_flushes_rank_bits ---
        for sf_bits in straight_flushes_rank_bits:
            prime_product = CardUtils.prime_product_from_rankbits(sf_bits)
            self.flush_lookup[prime_product] = rank
            rank += 1
        # Убедимся, что присвоили 10 рангов
        if rank != 11:
             logger.error(f"Assigned {rank-1} straight flush ranks, expected 10.")

        # Присваиваем ранги обычным флешам (начиная с MAX_FULL_HOUSE + 1)
        rank = LookupTable.MAX_FULL_HOUSE + 1
        for f_bits in normal_flush_rank_bits:
            prime_product = CardUtils.prime_product_from_rankbits(f_bits)
            self.flush_lookup[prime_product] = rank
            rank += 1
        # Проверяем, что последний ранг флеша соответствует константе
        if rank -1 != LookupTable.MAX_FLUSH:
             logger.warning(f"Last flush rank assigned is {rank-1}, expected {LookupTable.MAX_FLUSH}.")

        # Вычисляем ранги для стритов и старших карт (используя те же битовые маски)
        self._calculate_straights_and_highcards(straight_flushes_rank_bits, normal_flush_rank_bits)

    def _calculate_straights_and_highcards(self, straights_rank_bits: List[int], highcards_rank_bits: List[int]):
        """ Вычисляет ранги для стритов и старших карт (не флеш). """
        # Присваиваем ранги стритам (начиная с MAX_FLUSH + 1)
        rank = LookupTable.MAX_FLUSH + 1
        # --- ИСПРАВЛЕНО: Используем правильный порядок straight_flushes_rank_bits ---
        for s_bits in straights_rank_bits: # Этот список уже отсортирован по покерному старшинству
            prime_product = CardUtils.prime_product_from_rankbits(s_bits)
            self.unsuited_lookup[prime_product] = rank
            rank += 1
        if rank -1 != LookupTable.MAX_STRAIGHT:
             logger.warning(f"Last straight rank assigned is {rank-1}, expected {LookupTable.MAX_STRAIGHT}.")

        # Присваиваем ранги старшим картам (начиная с MAX_PAIR + 1)
        rank = LookupTable.MAX_PAIR + 1
        # --- ИСПРАВЛЕНО: Используем normal_flush_rank_bits, который отсортирован по убыванию ---
        for h_bits in highcards_rank_bits: # Этот список уже отсортирован по убыванию
            prime_product = CardUtils.prime_product_from_rankbits(h_bits)
            self.unsuited_lookup[prime_product] = rank
            rank += 1
        if rank -1 != LookupTable.MAX_HIGH_CARD:
             logger.warning(f"Last high card rank assigned is {rank-1}, expected {LookupTable.MAX_HIGH_CARD}.")


    def _calculate_multiples(self):
        """ Вычисляет ранги для Каре, Фулл-хаусов, Сетов, Двух пар и Пар. """
        backwards_ranks = range(len(INT_RANKS) - 1, -1, -1) # 12 down to 0 (A to 2)

        # 1) Каре (Four of a Kind)
        rank = LookupTable.MAX_STRAIGHT_FLUSH + 1 # Начинаем с 11
        for quad_rank_idx in backwards_ranks: # От AAAA down to 2222
            kickers_indices = list(backwards_ranks)
            kickers_indices.remove(quad_rank_idx)
            for kicker_idx in kickers_indices: # От A down to 2 (кроме quad_rank)
                product = PRIMES[quad_rank_idx]**4 * PRIMES[kicker_idx]
                self.unsuited_lookup[product] = rank
                rank += 1
        if rank -1 != LookupTable.MAX_FOUR_OF_A_KIND:
             logger.warning(f"Last 4-of-a-kind rank assigned is {rank-1}, expected {LookupTable.MAX_FOUR_OF_A_KIND}.")

        # 2) Фулл-хаус (Full House)
        rank = LookupTable.MAX_FOUR_OF_A_KIND + 1 # Начинаем с 167
        for trip_rank_idx in backwards_ranks: # От AAAxx down to 222xx
            pair_rank_indices = list(backwards_ranks)
            pair_rank_indices.remove(trip_rank_idx)
            for pair_rank_idx in pair_rank_indices: # От AKK down to A22, ..., 233
                product = PRIMES[trip_rank_idx]**3 * PRIMES[pair_rank_idx]**2
                self.unsuited_lookup[product] = rank
                rank += 1
        if rank -1 != LookupTable.MAX_FULL_HOUSE:
             logger.warning(f"Last full house rank assigned is {rank-1}, expected {LookupTable.MAX_FULL_HOUSE}.")

        # 3) Сет (Three of a Kind)
        rank = LookupTable.MAX_STRAIGHT + 1 # Начинаем с 1610
        for trip_rank_idx in backwards_ranks: # От AAA down to 222
            kickers_indices = list(backwards_ranks)
            kickers_indices.remove(trip_rank_idx)
            # Комбинации кикеров должны быть отсортированы (старшие два кикера)
            kicker_combos = sorted(itertools.combinations(kickers_indices, 2), reverse=True)
            for k1_idx, k2_idx in kicker_combos: # k1 > k2
                product = PRIMES[trip_rank_idx]**3 * PRIMES[k1_idx] * PRIMES[k2_idx]
                self.unsuited_lookup[product] = rank
                rank += 1
        if rank -1 != LookupTable.MAX_THREE_OF_A_KIND:
             logger.warning(f"Last 3-of-a-kind rank assigned is {rank-1}, expected {LookupTable.MAX_THREE_OF_A_KIND}.")

        # 4) Две пары (Two Pair)
        rank = LookupTable.MAX_THREE_OF_A_KIND + 1 # Начинаем с 2468
        # Комбинации пар, отсортированные по старшей паре, затем по младшей
        two_pair_combos = sorted(itertools.combinations(backwards_ranks, 2), reverse=True)
        for p1_idx, p2_idx in two_pair_combos: # p1 > p2
            kickers_indices = list(backwards_ranks)
            kickers_indices.remove(p1_idx)
            kickers_indices.remove(p2_idx)
            # Кикер должен быть отсортирован (старший)
            for kicker_idx in sorted(kickers_indices, reverse=True):
                product = PRIMES[p1_idx]**2 * PRIMES[p2_idx]**2 * PRIMES[kicker_idx]
                self.unsuited_lookup[product] = rank
                rank += 1
        if rank -1 != LookupTable.MAX_TWO_PAIR:
             logger.warning(f"Last two pair rank assigned is {rank-1}, expected {LookupTable.MAX_TWO_PAIR}.")

        # 5) Пара (Pair)
        rank = LookupTable.MAX_TWO_PAIR + 1 # Начинаем с 3326
        for pair_rank_idx in backwards_ranks: # От AA down to 22
            kickers_indices = list(backwards_ranks)
            kickers_indices.remove(pair_rank_idx)
            # Комбинации кикеров должны быть отсортированы (старшие три)
            kicker_combos = sorted(itertools.combinations(kickers_indices, 3), reverse=True)
            for k1_idx, k2_idx, k3_idx in kicker_combos: # k1 > k2 > k3
                product = PRIMES[pair_rank_idx]**2 * PRIMES[k1_idx] * PRIMES[k2_idx] * PRIMES[k3_idx]
                self.unsuited_lookup[product] = rank
                rank += 1
        if rank -1 != LookupTable.MAX_PAIR:
             logger.warning(f"Last pair rank assigned is {rank-1}, expected {LookupTable.MAX_PAIR}.")


    def _get_lexographically_next_bit_sequence(self, bits: int) -> Generator[int, None, None]:
        """ Генератор, возвращающий следующую лексикографическую перестановку битов. """
        # Алгоритм Gosper's Hack
        next_val = bits
        while True:
            # t = (v | (v - 1)) + 1; next = t | ((((t & -t) // (v & -v)) >> 1) - 1)
            if next_val == 0: break # Защита от бесконечного цикла, если bits=0
            try:
                rightmost_one = next_val & -next_val # Изолируем самый правый установленный бит
                if rightmost_one == 0: break # Не должно происходить для > 0
                next_higher_one_bit = next_val + rightmost_one # Устанавливаем бит слева от блока единиц
                rightmost_block_of_ones = next_val ^ next_higher_one_bit # Изолируем блок единиц справа
                # Сдвигаем блок вправо и убираем лишние биты
                rightmost_block_shifted = (rightmost_block_of_ones // rightmost_one) >> 2
                next_val = next_higher_one_bit | rightmost_block_shifted
            except Exception as e_gosper:
                 logger.error(f"Error in Gosper's Hack for bits={bin(bits)}: {e_gosper}", exc_info=True)
                 break # Прерываем генерацию при ошибке

            # Ограничиваем 13 битами (максимальная маска для 5 из 13)
            if next_val >= (1 << 13): break
            yield next_val

# (Блок if __name__ == '__main__' без изменений)
if __name__ == '__main__':
    print("Generating 5-card lookup table...")
    lookup = LookupTable()
    print("Table generated.")
    prime_4a_k = CardUtils.PRIMES[12]**4 * CardUtils.PRIMES[11]
    print(f"Rank for AAAA K (prime {prime_4a_k}): {lookup.unsuited_lookup.get(prime_4a_k)}")
    prime_rf = CardUtils.prime_product_from_rankbits(0b1111100000000)
    print(f"Rank for Royal Flush (prime {prime_rf}): {lookup.flush_lookup.get(prime_rf)}")
    prime_wheel_sf = CardUtils.prime_product_from_rankbits(0b1000000001111)
    print(f"Rank for Wheel SF (prime {prime_wheel_sf}): {lookup.flush_lookup.get(prime_wheel_sf)}")
    prime_75432 = CardUtils.PRIMES[5] * CardUtils.PRIMES[3] * CardUtils.PRIMES[2] * CardUtils.PRIMES[1] * CardUtils.PRIMES[0]
    print(f"Rank for 75432 unsuited (prime {prime_75432}): {lookup.unsuited_lookup.get(prime_75432)}")
