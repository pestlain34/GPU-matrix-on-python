import numpy as np
import random
import time

# Константы
N_MIN = 58
N_MAX = 58
W_FALLBACK = 1000_000_000
M_BITS_FALLBACK = 14
MAX_BUCKET_SIZE_FALLBACK = 50
FALLBACK_TIME_LIMIT_MS = 24 * 60 * 60 * 1000
PSD_FILTER_ENABLED = False
PSD_THRESHOLD_ADDITIVE = 2.0
OUTPUT_FILENAME = "balonin_" + str(N_MIN) + "_fast.js"
MAX_CHECKS_PER_BUCKET_PAIR = 8000

fft_cache = {}

def calculate_k1(v):
    return v // 2

def negate_seq(seq):
    if seq is None:
        return None
    return [-x for x in seq]

class SequencePair:
    def __init__(self, seqA, seqB, method):
        self.method = method

        currentA = seqA.copy()
        currentB = seqB.copy()

        if currentA > currentB:
            currentA, currentB = currentB, currentA

        negatedA = negate_seq(currentA)
        if currentA > negatedA:
            self.a = negatedA
            self.b = negate_seq(currentB)
        else:
            self.a = currentA
            self.b = currentB

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b

    def __hash__(self):
        return hash((tuple(self.a), tuple(self.b)))

    def __str__(self):
        return f"Pair(method={self.method}, a={self.a}, b={self.b})"

def write_array_to_file_js(writer, arr, symbol):
    if arr is None:
        writer.write(f"            {symbol} = null;\n")
        return
    writer.write(f"            {symbol} = [{','.join(map(str, arr))}];\n")

def find_all_pairs_dispatcher(n):
    print(f"Ищем уникальные пары для n = {n}")
    start_time = time.time()

    unique_pairs_for_n = find_pairs_by_randomized_bucketing(n)

    end_time = time.time()
    if unique_pairs_for_n:
        print(f"+++ Найдена {len(unique_pairs_for_n)} уникальная пара для n = {n} за {end_time - start_time:.3f} сек +++")
    else:
        print(f"--- Решения для n={n} не найдены за отведенное время/попытки ---")
    return unique_pairs_for_n

def find_pairs_by_randomized_bucketing(n):
    found_pairs_set = set()
    v = n // 2
    if v <= 0:
        print(f"Некорректное значение v={v} для n={n}")
        return []
    k1_target = calculate_k1(v)
    print(f"Используем k1 = {k1_target} для генерации последовательностей длины v = {v}")

    total_attempts = 0
    total_comparisons = 0
    start_time = time.time()
    deadline = start_time + FALLBACK_TIME_LIMIT_MS / 1000
    candidates_to_gen = 0


    # Основной цикл без многозадачности
    while candidates_to_gen < W_FALLBACK and len(found_pairs_set) < 1:  # Ищем только первую пару
        buckets_a_local = {}
        buckets_b_local = {}

        for _ in range(10000):
            candidates_to_gen += 1
            if len(found_pairs_set) >= 1:
                break
            seq = generate_sequence_with_k1(v, k1_target, random.Random())
            paf = cyclic_autocorrelation(seq)

            if paf is not None:
                data = (seq, paf)
                store_sequence_local(buckets_a_local, calculate_hash_n1(paf), data)
                store_sequence_local(buckets_b_local, calculate_hash_n2(paf), data)
            total_attempts += 1

        if len(found_pairs_set) >= 1:
            break

        common_keys = set(buckets_a_local.keys()).intersection(buckets_b_local.keys())

        for key in common_keys:
            if len(found_pairs_set) >= 1:
                break

            list_a = buckets_a_local.get(key)
            list_b = buckets_b_local.get(key)

            if not list_a or not list_b:
                continue

            checks_in_bucket_pair = 0

            for data_a in list_a:
                for data_b in list_b:
                    total_comparisons += 1

                    # Убедимся, что последовательности a и b не одинаковые
                    if np.array_equal(data_a[0], data_b[0]):
                        continue  # Пропускаем, если последовательности одинаковые

                    if check_euler_pair_condition(data_a[1], data_b[1], v):
                        found_pair = SequencePair(data_a[0], data_b[0], "BucketSearch")
                        found_pairs_set.add(found_pair)
                        print(f"\n[n={v}] +++ Найдена пара! Всего найдено: {len(found_pairs_set)}. Попыток: {total_attempts / 1_000_000:.2f}M, Сравнений: {total_comparisons / 1_000_000:.2f}M.")
                        return list(found_pairs_set)
                    checks_in_bucket_pair += 1
                    if checks_in_bucket_pair >= MAX_CHECKS_PER_BUCKET_PAIR:
                        break
                if checks_in_bucket_pair >= MAX_CHECKS_PER_BUCKET_PAIR:
                    break

    duration = time.time() - start_time
    print(f"Попыток генерации: {total_attempts / 1_000_000:.2f}M. Сравнений PAF: {total_comparisons / 1_000_000:.2f}M. Время: {duration:.2f}s.")
    return list(found_pairs_set)

def cyclic_autocorrelation(seq):
    v_len = len(seq)
    if v_len == 0:
        return None

    try:
        fft = fft_cache.get(v_len, None)
        if fft is None:
            fft = np.fft.fft
            fft_cache[v_len] = fft

        data = np.zeros(2 * v_len)
        for i in range(v_len):
            data[2 * i] = seq[i]

        fft_data = fft(data)
        data[:v_len] = np.real(fft_data[:v_len])**2 + np.imag(fft_data[:v_len])**2

        if PSD_FILTER_ENABLED:
            psd_threshold = v_len + PSD_THRESHOLD_ADDITIVE
            if any(d > psd_threshold for d in data[1:v_len]):
                return None

        result = np.fft.ifft(data[:v_len]).real
        return np.round(result).astype(int)
    except Exception as e:
        print(f"Ошибка при вычислении АКФ: {e}")
        return None

def store_sequence_local(buckets, hash, data):
    if hash not in buckets:
        buckets[hash] = []
    if MAX_BUCKET_SIZE_FALLBACK == 0 or len(buckets[hash]) < MAX_BUCKET_SIZE_FALLBACK:
        buckets[hash].append(data)

def check_euler_pair_condition(paf_a, paf_b, v):
    if paf_a is None or paf_b is None or len(paf_a) != v or len(paf_b) != v:
        return False

    if paf_a[0] != v or paf_b[0] != v:
        return False

    for k in range(1, v):
        if paf_a[k] + paf_b[k] != -2:
            return False
    return True

def generate_sequence_with_k1(length, k1, rand):
    if k1 < 0 or k1 > length:
        raise ValueError(f"k1 должно быть в диапазоне [0, {length}]")
    seq = [1] * length

    if k1 == length:
        seq = [-1] * length
        return seq
    if k1 == 0:
        return seq

    indices = list(range(length))
    rand.shuffle(indices)

    for i in indices[:k1]:
        seq[i] = -1
    return seq

def calculate_hash_n1(paf):
    if paf is None:
        return 0
    return sum(1 << (i - 1) if paf[i] > 0 else 0 for i in range(1, min(M_BITS_FALLBACK + 1, len(paf))))

def calculate_hash_n2(paf):
    if paf is None:
        return 0
    return sum(1 << (i - 1) if paf[i] < -2 else 0 for i in range(1, min(M_BITS_FALLBACK + 1, len(paf))))

if __name__ == '__main__':
    print(f"Цель: Найти первую пару для n = {N_MIN} и завершить программу.")
    print(f"Результат будет записан в файл: {OUTPUT_FILENAME}")
    print("--------------------------------------------------")

    with open(OUTPUT_FILENAME, "w") as writer:
        writer.write(f"// VERSION: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        writer.write(f"// Поиск для n={N_MIN}\n")
        writer.write(f"\nn={N_MIN}; example(n, 0);\n")
        writer.write("\nputs(\"a=[\"+a+\"];\"); puts(\"b=[\"+b+\"];\");\n")
        writer.write("H=twocircul(a, b); {{I=H'*H}} putm(I);\n")
        writer.write("plotm(H,'XR',140,20);\n\n")

        first_n_processed = False
        for n_current in range(N_MIN, N_MAX + 1, 4):
            unique_pairs_for_n = find_all_pairs_dispatcher(n_current)

            if unique_pairs_for_n:
                if first_n_processed:
                    writer.write("else ")
                writer.write(f"    if (n == {n_current}) {{\n")

                for k_idx, pair in enumerate(unique_pairs_for_n):
                    writer.write(f"        if (k == {k_idx}) {{\n")
                    write_array_to_file_js(writer, pair.a, "a")
                    write_array_to_file_js(writer, pair.b, "b")
                    writer.write(f"        }}\n")

                writer.write(f"    }}\n")
                first_n_processed = True
                break

    print("Поиск завершен. Результаты записаны в " + OUTPUT_FILENAME)
