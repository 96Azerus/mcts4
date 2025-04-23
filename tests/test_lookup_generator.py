# tests/test_lookup_generator.py v1.2
"""
Unit-тесты для генератора таблиц поиска 5-карточных рук.
"""
import pytest

# Импортируем CardUtils из src.card
from src.card import Card as CardUtils
from src.evaluator.ofc_5card_lookup import LookupTable

def test_lexographical_generator():
    """Тестирует генератор лексикографических битовых последовательностей."""
    lookup = LookupTable() # Создаем экземпляр для доступа к генератору
    start_bits = (1 << 5) - 1 # 0b11111 (5 бит)
    gen = lookup._get_lexographically_next_bit_sequence(start_bits)

    count = 0
    generated_sequences = set()
    generated_sequences.add(start_bits) # Добавляем начальное значение

    try:
        while True:
            bits = next(gen)
            assert bits < (1 << 13)
            assert bin(bits).count('1') == 5
            assert bits not in generated_sequences
            generated_sequences.add(bits)
            count += 1
            if count > 2000:
                pytest.fail("Generator produced too many sequences (> 2000)")
    except StopIteration:
        pass

    assert count == 1286, f"Generator produced {count} sequences, expected 1286"
    assert len(generated_sequences) == 1287, f"Total unique sequences {len(generated_sequences)}, expected 1287"

def test_lookup_table_generation():
    """Тестирует полноту сгенерированных таблиц."""
    lookup = LookupTable()
    assert len(lookup.flush_lookup) == 1287, f"Flush lookup size: {len(lookup.flush_lookup)}"
    assert len(lookup.unsuited_lookup) == 6175, f"Unsuited lookup size: {len(lookup.unsuited_lookup)}"

    # Royal Flush
    rf_bits = 0b1111100000000
    rf_prime = CardUtils.prime_product_from_rankbits(rf_bits)
    assert lookup.flush_lookup.get(rf_prime) == 1, "Royal Flush rank mismatch"

    # Straight Flush (Wheel)
    wheel_sf_bits = 0b1000000001111
    wheel_sf_prime = CardUtils.prime_product_from_rankbits(wheel_sf_bits)
    # --- ИСПРАВЛЕНО: Ожидаем ранг 10 ---
    assert lookup.flush_lookup.get(wheel_sf_prime) == 10, "Wheel SF rank mismatch"

    # Four Aces (AAAA K)
    four_aces_k_prime = CardUtils.PRIMES[12]**4 * CardUtils.PRIMES[11]
    assert lookup.unsuited_lookup.get(four_aces_k_prime) == 11, "Four Aces K rank mismatch"

    # Worst High Card (75432)
    worst_hc_prime = CardUtils.PRIMES[5] * CardUtils.PRIMES[3] * CardUtils.PRIMES[2] * CardUtils.PRIMES[1] * CardUtils.PRIMES[0]
    assert lookup.unsuited_lookup.get(worst_hc_prime) == 7462, "Worst High Card rank mismatch"
