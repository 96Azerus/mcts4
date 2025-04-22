# src/fantasyland_solver.py v1.1
"""
Эвристический солвер для размещения 13 из N (14-17) карт в Progressive Fantasyland.
Приоритеты: 1. Не фол. 2. Удержание ФЛ (Re-Fantasy). 3. Максимизация роялти.
"""
import random
import time
import logging # Используем logging
from typing import List, Tuple, Dict, Optional
from itertools import combinations, permutations
from collections import Counter

# --- ИСПРАВЛЕНО: Импорты из src пакета ---
from src.card import Card as CardUtils, card_to_str, RANK_MAP, STR_RANKS, INVALID_CARD
from src.scoring import (
    check_fantasyland_stay, get_row_royalty, check_board_foul,
    get_hand_rank_safe, RANK_CLASS_QUADS, RANK_CLASS_TRIPS,
    RANK_CLASS_HIGH_CARD
)

# Импортируем PlayerBoard только для аннотации типов
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.board import PlayerBoard

# Получаем логгер
logger = logging.getLogger(__name__)
# Устанавливаем уровень по умолчанию, если он не настроен глобально
if not logger.hasHandlers(): logger.setLevel(logging.WARNING)


class FantasylandSolver:
    """
    Решает задачу размещения руки Fantasyland (14-17 карт).
    Использует эвристики для поиска хорошего, но не обязательно оптимального решения.
    """
    # Максимальное количество комбинаций сброса для перебора (для производительности)
    MAX_DISCARD_COMBINATIONS = 100 # Увеличено для большего охвата
    # Максимальное количество размещений для оценки на каждую комбинацию сброса
    MAX_PLACEMENTS_PER_DISCARD = 50 # Можно настроить

    def solve(self, hand: List[int]) -> Tuple[Optional[Dict[str, List[int]]], Optional[List[int]]]:
        """
        Находит эвристически лучшее размещение 13 карт из N (14-17) карт руки Fantasyland.

        Args:
            hand (List[int]): Список из N (14-17) карт (int) в руке Фантазии.

        Returns:
            Tuple[Optional[Dict[str, List[int]]], Optional[List[int]]]:
            Кортеж:
            - Словарь с размещением {'top': [...], 'middle': [...], 'bottom': [...]} (карты int)
              или None, если валидное размещение не найдено (сигнал фола).
            - Список сброшенных карт (int) или None (если фол).
        """
        start_time = time.time()
        n_cards = len(hand)
        n_place = 13

        # --- Улучшено: Валидация входных данных ---
        if not isinstance(hand, list):
             logger.error(f"FL Solver Error: Input hand is not a list (type: {type(hand)}).")
             return None, None
        if not (14 <= n_cards <= 17):
            logger.error(f"FL Solver Error: Invalid hand size {n_cards}. Expected 14-17.")
            return None, None

        valid_hand: List[int] = []
        for i, c in enumerate(hand):
             if not isinstance(c, int) or c == INVALID_CARD or c <= 0:
                  logger.error(f"FL Solver Error: Invalid card value '{c}' at index {i} in hand: {[card_to_str(card) for card in hand]}")
                  return None, None # Невалидная карта - фол
             valid_hand.append(c)

        if len(valid_hand) != len(set(valid_hand)):
             counts = Counter(valid_hand)
             duplicates = {card_to_str(card): count for card, count in counts.items() if count > 1}
             logger.error(f"FL Solver Error: Duplicate cards found in hand: {duplicates}")
             return None, None # Дубликаты - фол

        n_discard = n_cards - n_place
        if not (1 <= n_discard <= 4): # Должно быть от 1 до 4 карт для сброса
             logger.error(f"FL Solver Error: Calculated discard count {n_discard} is invalid for hand size {n_cards}.")
             return None, None

        best_overall_placement: Optional[Dict[str, List[int]]] = None
        best_overall_discarded: Optional[List[int]] = None
        # Оценка: -1 = фол, 0 = не удерживает ФЛ, 1 = удерживает ФЛ
        best_overall_score: int = -2 # Начальное значение хуже фола
        best_overall_royalty: int = -1

        # 1. Генерируем комбинации карт для сброса
        try:
            discard_combinations_iter = combinations(valid_hand, n_discard)
            # Ограничиваем количество комбинаций для анализа
            # Берем с запасом, чтобы потом выбрать случайные + умные
            discard_combinations_list = list(itertools.islice(discard_combinations_iter, self.MAX_DISCARD_COMBINATIONS * 3))

            if len(discard_combinations_list) > self.MAX_DISCARD_COMBINATIONS:
                 # Добавляем "умный" сброс (самые младшие карты)
                 smart_discards = []
                 try:
                      sorted_hand = sorted(valid_hand, key=lambda c: CardUtils.get_rank_int(c))
                      smart_discards = [tuple(sorted_hand[:n_discard])]
                 except Exception as e_sort:
                      logger.warning(f"Warning: Error sorting hand for smart discard in FL solver: {e_sort}")

                 # Выбираем случайные комбинации + умный сброс
                 num_random_needed = max(0, self.MAX_DISCARD_COMBINATIONS - len(smart_discards))
                 if num_random_needed > 0 and len(discard_combinations_list) > num_random_needed:
                      # Убедимся, что не выбираем умный сброс случайно
                      non_smart_combos = [c for c in discard_combinations_list if c not in smart_discards]
                      random_discards = random.sample(non_smart_combos, min(num_random_needed, len(non_smart_combos)))
                      combinations_to_check = smart_discards + random_discards
                 else:
                      combinations_to_check = smart_discards[:self.MAX_DISCARD_COMBINATIONS]
            else:
                 combinations_to_check = discard_combinations_list

        except Exception as e_comb:
            logger.error(f"Error generating discard combinations: {e_comb}", exc_info=True)
            return None, None # Не можем продолжить без комбинаций

        if not combinations_to_check:
             logger.error("FL Solver Error: No discard combinations generated.")
             return None, None

        logger.debug(f"FL Solver: Checking {len(combinations_to_check)} discard combinations...")

        # 2. Перебираем варианты сброса
        for discarded_tuple in combinations_to_check:
            discarded_list = list(discarded_tuple)
            # Формируем набор из 13 карт для размещения
            remaining_cards = [c for c in valid_hand if c not in discarded_list]
            if len(remaining_cards) != 13:
                logger.warning(f"Debug: Skipping discard combo due to wrong remaining count {len(remaining_cards)}")
                continue # Пропускаем, если что-то пошло не так

            current_best_placement_for_discard = None
            current_best_score_for_discard = -1 # -1 фол, 0 не ФЛ, 1 ФЛ
            current_max_royalty_for_discard = -1

            # 3. Генерируем и оцениваем эвристические размещения для текущих 13 карт
            placements_to_evaluate = self._generate_heuristic_placements(remaining_cards)

            for placement in placements_to_evaluate:
                if not placement: continue # Пропускаем None

                # Оцениваем размещение (фол, удержание ФЛ, роялти)
                score, royalty = self._evaluate_placement(placement)

                # Выбираем лучшее размещение для ДАННОГО набора 13 карт
                # Приоритет: не фол -> удержание ФЛ -> макс роялти
                if score > current_best_score_for_discard or \
                   (score == current_best_score_for_discard and royalty > current_max_royalty_for_discard):
                    current_best_score_for_discard = score
                    current_max_royalty_for_discard = royalty
                    current_best_placement_for_discard = placement

            # 4. Обновляем лучшее ОБЩЕЕ размещение (среди всех вариантов сброса)
            if current_best_placement_for_discard:
                if current_best_score_for_discard > best_overall_score or \
                   (current_best_score_for_discard == best_overall_score and current_max_royalty_for_discard > best_overall_royalty):
                    best_overall_score = current_best_score_for_discard
                    best_overall_royalty = current_max_royalty_for_discard
                    best_overall_placement = current_best_placement_for_discard
                    best_overall_discarded = discarded_list
                    logger.debug(f"FL Solver: New best found! Score: {best_overall_score}, Royalty: {best_overall_royalty}, Discard: {[card_to_str(c) for c in best_overall_discarded]}")

        solve_duration = time.time() - start_time
        logger.info(f"FL Solver: Checked {len(combinations_to_check)} discard combinations in {solve_duration:.3f}s. Best Score: {best_overall_score}, Royalty: {best_overall_royalty}")

        # 5. Обработка случая, если не найдено ни одного валидного размещения
        if best_overall_placement is None:
            logger.warning("FL Solver Warning: No valid non-foul placement found for any discard combination.")
            # Пытаемся сделать простое безопасное размещение для первого варианта сброса
            if combinations_to_check:
                first_discard = list(combinations_to_check[0])
                first_remaining = [c for c in valid_hand if c not in first_discard]
                if len(first_remaining) == 13:
                    fallback_placement = self._generate_simple_safe_placement(first_remaining)
                    if fallback_placement:
                        logger.info("FL Solver: Falling back to simple non-foul placement.")
                        return fallback_placement, first_discard
            # Если и это не помогло, возвращаем None, None (сигнал фола для агента)
            logger.error("FL Solver Error: Could not generate any valid placement.")
            # Возвращаем None для размещения и карты для сброса (если есть)
            return None, valid_hand[13:] if len(valid_hand) > 13 else []

        # Убедимся, что возвращаем список для сброса
        return best_overall_placement, best_overall_discarded if best_overall_discarded is not None else []

    def _generate_heuristic_placements(self, cards: List[int]) -> List[Optional[Dict[str, List[int]]]]:
        """Генерирует несколько кандидатов на размещение 13 карт."""
        if len(cards) != 13: return []

        placements = []
        # Эвристика 1: Пытаемся собрать Каре+ на боттоме
        placements.append(self._try_build_strong_bottom(cards))
        # Эвристика 2: Пытаемся собрать Сет на топе
        placements.append(self._try_build_set_top(cards))
        # Эвристика 3: Пытаемся максимизировать роялти (общее размещение)
        placements.append(self._try_maximize_royalty_heuristic(cards))
        # Эвристика 4: Простое безопасное размещение (на всякий случай)
        placements.append(self._generate_simple_safe_placement(cards))

        # Убираем None и дубликаты (хотя дубликаты маловероятны)
        unique_placements = []
        seen_placements = set()
        for p in placements:
             if p:
                  # Создаем неизменяемое представление для проверки дубликатов
                  try:
                       p_tuple = tuple(tuple(sorted(p[row])) for row in ['top', 'middle', 'bottom'])
                       if p_tuple not in seen_placements:
                            unique_placements.append(p)
                            seen_placements.add(p_tuple)
                  except Exception as e_tup:
                       logger.warning(f"Error creating tuple for placement check: {e_tup}")
                       unique_placements.append(p) # Добавляем без проверки, если ошибка

        # Ограничиваем количество для оценки
        if len(unique_placements) > self.MAX_PLACEMENTS_PER_DISCARD:
             return random.sample(unique_placements, self.MAX_PLACEMENTS_PER_DISCARD)
        else:
             return unique_placements


    def _evaluate_placement(self, placement: Dict[str, List[int]]) -> Tuple[int, int]:
        """
        Оценивает готовое размещение 13 карт.

        Args:
            placement: Словарь с размещением {'top': [...], 'middle': [...], 'bottom': [...]}.

        Returns:
            Tuple[int, int]: Кортеж (score, total_royalty).
                             score: -1 (фол), 0 (не удерживает ФЛ), 1 (удерживает ФЛ).
                             total_royalty: Сумма роялти (или -1 при фоле).
        """
        # Проверка корректности структуры placement
        if not placement or \
           len(placement.get('top', [])) != 3 or \
           len(placement.get('middle', [])) != 5 or \
           len(placement.get('bottom', [])) != 5:
             logger.warning("Warning: Invalid placement structure in _evaluate_placement.")
             return -1, -1 # Некорректное размещение считаем фолом

        top, middle, bottom = placement['top'], placement['middle'], placement['bottom']

        # Проверяем на фол
        if check_board_foul(top, middle, bottom):
            return -1, -1 # Фол

        # Проверяем удержание ФЛ
        stays_in_fl = check_fantasyland_stay(top, middle, bottom)

        # Считаем роялти
        try:
            total_royalty = (get_row_royalty(top, 'top') +
                             get_row_royalty(middle, 'middle') +
                             get_row_royalty(bottom, 'bottom'))
        except Exception as e_royalty:
             logger.error(f"Error calculating royalty during FL placement evaluation: {e_royalty}", exc_info=True)
             total_royalty = 0 # Считаем 0 роялти при ошибке

        score = 1 if stays_in_fl else 0
        return score, total_royalty

    def _find_best_hand(self, cards: List[int], n: int) -> Optional[List[int]]:
        """Находит лучшую n-карточную комбинацию (по рангу) из списка карт."""
        if len(cards) < n: return None

        best_hand_combo: Optional[tuple] = None
        # Инициализируем худшим возможным рангом + запас
        best_rank: int = RANK_CLASS_HIGH_CARD + 1000

        # Ограничиваем количество комбинаций для производительности
        MAX_COMBOS_TO_CHECK = 500
        combos_checked = 0

        try:
            # Перебираем комбинации n карт
            for combo in combinations(cards, n):
                if combos_checked >= MAX_COMBOS_TO_CHECK: break
                combos_checked += 1
                combo_list = list(combo)
                # Используем get_hand_rank_safe, т.к. комбинация всегда будет полной (n карт)
                rank = get_hand_rank_safe(combo_list)
                # Меньший ранг лучше
                if rank < best_rank:
                    best_rank = rank
                    best_hand_combo = combo # Сохраняем кортеж
        except Exception as e:
             logger.error(f"Error finding best {n}-card hand: {e}", exc_info=True)
             return None

        # Возвращаем список или None
        return list(best_hand_combo) if best_hand_combo is not None else None

    def _generate_simple_safe_placement(self, cards: List[int]) -> Optional[Dict[str, List[int]]]:
         """Создает простое размещение, сортируя карты и раскладывая по рядам, проверяя на фол."""
         if len(cards) != 13: return None
         try:
              # Сортируем карты от старшей к младшей
              sorted_cards = sorted(cards, key=lambda c: CardUtils.get_rank_int(c), reverse=True)
              # Простое размещение
              placement = {
                   'bottom': sorted_cards[0:5],
                   'middle': sorted_cards[5:10],
                   'top': sorted_cards[10:13]
              }
              # Проверяем на фол
              if not check_board_foul(placement['top'], placement['middle'], placement['bottom']):
                   return placement
              else:
                   # Если фол, пытаемся поменять местами средний и нижний ряд (частая причина фола)
                   placement_swapped = {
                        'bottom': sorted_cards[5:10],
                        'middle': sorted_cards[0:5],
                        'top': sorted_cards[10:13]
                   }
                   if not check_board_foul(placement_swapped['top'], placement_swapped['middle'], placement_swapped['bottom']):
                        return placement_swapped
                   else:
                        # Если и так фол, возвращаем None
                        return None
         except Exception as e:
              logger.error(f"Error creating simple safe placement: {e}", exc_info=True)
              return None

    # --- Эвристики для генерации кандидатов ---
    # (Эти методы пытаются построить сильные комбинации в нужных рядах)

    def _try_build_strong_bottom(self, cards: List[int]) -> Optional[Dict[str, List[int]]]:
        """Пытается собрать Каре+ на боттоме, остальное эвристически."""
        if len(cards) != 13: return None

        best_placement_candidate = None
        max_royalty = -1 # Отслеживаем роялти для выбора лучшего варианта
        best_score = -1 # Отслеживаем score (-1 фол, 0 не ФЛ, 1 ФЛ)

        # Ограниченный перебор комбинаций для боттома
        MAX_BOTTOM_COMBOS = 250 # Увеличено
        combos_checked = 0

        try:
            for bottom_tuple in combinations(cards, 5):
                if combos_checked >= MAX_BOTTOM_COMBOS: break
                combos_checked += 1
                bottom_list = list(bottom_tuple)
                rank_b = get_hand_rank_safe(bottom_list)

                # Проверяем, является ли боттом Каре или лучше
                if rank_b <= RANK_CLASS_QUADS: # Каре = 166
                    remaining8 = [c for c in cards if c not in bottom_list]
                    if len(remaining8) != 8: continue

                    # Находим лучшую среднюю руку из оставшихся 8
                    middle_list = self._find_best_hand(remaining8, 5)
                    if middle_list:
                        top_list = [c for c in remaining8 if c not in middle_list]
                        if len(top_list) == 3:
                            placement = {'top': top_list, 'middle': middle_list, 'bottom': bottom_list}
                            score, royalty = self._evaluate_placement(placement)

                            # Обновляем лучший, если лучше по score или по royalty при равном score
                            if score > best_score or (score == best_score and royalty > max_royalty):
                                best_score = score
                                max_royalty = royalty
                                best_placement_candidate = placement
        except Exception as e:
             logger.error(f"Error trying to build strong bottom: {e}", exc_info=True)

        return best_placement_candidate

    def _try_build_set_top(self, cards: List[int]) -> Optional[Dict[str, List[int]]]:
        """Пытается собрать Сет на топе, остальное эвристически."""
        if len(cards) != 13: return None

        best_placement_candidate = None
        max_royalty = -1
        best_score = -1

        try:
            rank_counts = Counter(CardUtils.get_rank_int(c) for c in cards)
            possible_set_ranks = [rank for rank, count in rank_counts.items() if count >= 3]
        except Exception as e:
            logger.error(f"Error counting ranks for set top in FL solver: {e}", exc_info=True)
            return None

        for set_rank_index in possible_set_ranks:
            try:
                # Собираем карты для сета
                set_cards = [c for c in cards if CardUtils.get_rank_int(c) == set_rank_index][:3]
                if len(set_cards) != 3: continue

                remaining10 = [c for c in cards if c not in set_cards]
                if len(remaining10) != 10: continue

                # Находим лучшую нижнюю руку из оставшихся 10
                bottom_list = self._find_best_hand(remaining10, 5)
                if bottom_list:
                     middle_list = [c for c in remaining10 if c not in bottom_list]
                     if len(middle_list) == 5:
                         placement = {'top': set_cards, 'middle': middle_list, 'bottom': bottom_list}
                         score, royalty = self._evaluate_placement(placement)

                         if score > best_score or (score == best_score and royalty > max_royalty):
                             best_score = score
                             max_royalty = royalty
                             best_placement_candidate = placement
            except Exception as e_set:
                 logger.error(f"Error trying to build set top for rank {set_rank_index}: {e_set}", exc_info=True)
                 continue # Пробуем следующий сет

        return best_placement_candidate

    def _try_maximize_royalty_heuristic(self, cards: List[int]) -> Optional[Dict[str, List[int]]]:
        """Простая эвристика: размещаем лучшие возможные руки на боттом/мидл/топ без фола."""
        if len(cards) != 13: return None

        best_placement_candidate = None
        max_royalty = -1
        best_score = -1

        # Ограниченный перебор комбинаций для боттома
        MAX_BOTTOM_COMBOS = 200 # Увеличено
        combos_checked = 0

        try:
            for bottom_tuple in combinations(cards, 5):
                if combos_checked >= MAX_BOTTOM_COMBOS: break
                combos_checked += 1
                bottom_list = list(bottom_tuple)
                remaining8 = [c for c in cards if c not in bottom_list]
                if len(remaining8) != 8: continue

                # Находим лучшую среднюю руку
                middle_list = self._find_best_hand(remaining8, 5)
                if middle_list:
                     top_list = [c for c in remaining8 if c not in middle_list]
                     if len(top_list) == 3:
                         placement = {'top': top_list, 'middle': middle_list, 'bottom': bottom_list}
                         score, royalty = self._evaluate_placement(placement)

                         if score > best_score or (score == best_score and royalty > max_royalty):
                             best_score = score
                             max_royalty = royalty
                             best_placement_candidate = placement
        except Exception as e:
             logger.error(f"Error trying to maximize royalty heuristic: {e}", exc_info=True)


        # Если перебор не дал результата или роялти низкие, возвращаем простое безопасное
        if best_score < 0 or (best_score == 0 and max_royalty < 5): # Порог можно настроить
             simple_safe = self._generate_simple_safe_placement(cards)
             if simple_safe:
                  simple_score, simple_royalty = self._evaluate_placement(simple_safe)
                  # Выбираем простое безопасное, только если оно лучше текущего лучшего
                  if simple_score > best_score or (simple_score == best_score and simple_royalty > max_royalty):
                       return simple_safe

        return best_placement_candidate
