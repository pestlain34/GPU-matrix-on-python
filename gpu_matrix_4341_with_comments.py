import time
import math
import cupy as cp
# Константы
SEQ = 200_000 # размер пакета генерации последовательностей за одну итерацию.
N = 70  # Порядок
v = N // 2
TARGET_CHUNK_MEMORY_FFT_GB = 0.5  # Целевой объем памяти (в ГБ) для обработки одного чанка в FFT
ABSOLUTE_MAX_CHUNK_SIZE_PAIRS = 2 ** 21 # Максимальное количество пар в одном чанке
MAX_BUCKET_SIZE_FALLBACK = 50 # Максимальное количество последовательностей в одном бакете
M_BITS_FALLBACK = 14  # Количество битов, используемых для хеширования (число бакетов: 2^M)
MAX_SEARCH_DURATION_SECONDS = float('inf')
MAX_TOTAL_GENERATED_SEQUENCES = float('inf')

# Вычисление значения k1 (это то сколько -1-ц будет в последовательности)
def calculate_k1(v):
    return v // 2

# Класс для представления пары последовательностей
class SequencePair:
    def __init__(self, seqA, seqB, method):
        self.method = method

        # Преобразуем последовательности к формату CuPy-массивов
        currentA_cp = cp.asarray(seqA)
        currentB_cp = cp.asarray(seqB)
        self.a = currentA_cp
        self.b = currentB_cp

    def __eq__(self, other):
        if not isinstance(other, SequencePair):
            return NotImplemented
        # Сравнение массивов на GPU
        return cp.array_equal(cp.asarray(self.a), cp.asarray(other.a)) and \
            cp.array_equal(cp.asarray(self.b), cp.asarray(other.b))

    def __hash__(self):
        # Хешируем пары
        a_tuple = tuple(cp.asnumpy(self.a))
        b_tuple = tuple(cp.asnumpy(self.b))
        return hash((a_tuple, b_tuple))

    def __str__(self):
        return f"Pair(method={self.method}, a={cp.asnumpy(self.a)}, b={cp.asnumpy(self.b)})"


def write_array_to_file_js_single_line(arr, symbol):
    # Преобразует массив в формат JavaScript строки для записи в файл.
    if arr is None:
        return f"            {symbol} = null;\n"
    np_arr = cp.asnumpy(arr) if isinstance(arr, cp.ndarray) else arr
    return f"            {symbol} = [{','.join(map(str, np_arr))}];\n"


# Возвращаем сформированную строку кода
def format_pair_for_js_worker(args):
    k_idx, pair_a_cpu, pair_b_cpu, n_val = args
    output_str = f"        if (k == {k_idx}) {{\n"
    output_str += write_array_to_file_js_single_line(pair_a_cpu, "a")
    output_str += write_array_to_file_js_single_line(pair_b_cpu, "b")
    output_str += f"        }}\n"
    return output_str


def check_collisions(buckets_a, buckets_b, counts_a, counts_b, v_val):
    if v_val <= 0: return []

    # Находим непустые бакеты в A и B
    non_empty_a = cp.where(counts_a > 0)[0]
    non_empty_b = cp.where(counts_b > 0)[0]
    common_hashes_idx = cp.intersect1d(non_empty_a, non_empty_b, assume_unique=True)

    # Находим пересечение непустых бакетов
    if common_hashes_idx.size == 0: return []
    del non_empty_a, non_empty_b

    # Количество элементов в общих бакетах
    current_a_counts = counts_a[common_hashes_idx]
    current_b_counts = counts_b[common_hashes_idx]

    # Общее количество возможных пар
    num_pairs_per_hash = current_a_counts * current_b_counts
    sum_total_pairs_val = cp.sum(num_pairs_per_hash).item()
    if sum_total_pairs_val == 0: return []
    total_generated_pairs = int(sum_total_pairs_val)

    # Создаем индексы для всех возможных пар
    _flat_all_pairs_indices = cp.arange(total_generated_pairs, dtype=cp.int32)
    _end_indices_cumsum = cp.cumsum(current_a_counts * current_b_counts)

    # Определяем хэш для каждой пары
    _target_hash_group_indices = cp.searchsorted(_end_indices_cumsum, _flat_all_pairs_indices, side='right')
    h_flat_idx = common_hashes_idx[_target_hash_group_indices]

    # Вычисляем позиции внутри группы
    _start_indices_per_group_contribution = cp.concatenate(
        (cp.array([0], dtype=_end_indices_cumsum.dtype), _end_indices_cumsum[:-1]))
    _local_pair_indices_in_group_meshgrid = _flat_all_pairs_indices - _start_indices_per_group_contribution[
        _target_hash_group_indices]

    _a_sizes_for_pairs_group = current_a_counts[_target_hash_group_indices]

    a_local_flat_idx = _local_pair_indices_in_group_meshgrid % _a_sizes_for_pairs_group
    b_local_flat_idx = _local_pair_indices_in_group_meshgrid // _a_sizes_for_pairs_group

    # Очистка временных переменных
    del _flat_all_pairs_indices, _target_hash_group_indices, _end_indices_cumsum
    del _start_indices_per_group_contribution, _local_pair_indices_in_group_meshgrid, _a_sizes_for_pairs_group

    cp.get_default_memory_pool().free_all_blocks()

    # Расчет размера порции для обработки с учетом памяти
    bytes_per_pair_for_fft_intermediate = 16 * v_val
    if bytes_per_pair_for_fft_intermediate == 0:
        dynamic_chunk_size = ABSOLUTE_MAX_CHUNK_SIZE_PAIRS
    else:
        dynamic_chunk_size = math.floor(
            (TARGET_CHUNK_MEMORY_FFT_GB * (1024 ** 3)) / bytes_per_pair_for_fft_intermediate)

    chunk_size = max(1, int(dynamic_chunk_size))
    chunk_size = min(chunk_size, ABSOLUTE_MAX_CHUNK_SIZE_PAIRS)
    chunk_size = min(chunk_size, total_generated_pairs)
    if chunk_size == 0 and total_generated_pairs > 0:
        chunk_size = 1
    elif total_generated_pairs == 0:
        return []  # Нет пар для обработки

    found_pair_object_list = []

    # Обработка пар порциями
    for i in range(0, total_generated_pairs, chunk_size):
        if found_pair_object_list: break

        chunk_start = i
        chunk_end = min(i + chunk_size, total_generated_pairs)

        # Извлекаем последовательности для текущей порции
        current_chunk_h_idx = h_flat_idx[chunk_start:chunk_end]
        current_chunk_a_local_idx = a_local_flat_idx[chunk_start:chunk_end]
        current_chunk_b_local_idx = b_local_flat_idx[chunk_start:chunk_end]

        a_seqs_chunk = buckets_a[current_chunk_h_idx, current_chunk_a_local_idx, :]
        b_seqs_chunk = buckets_b[current_chunk_h_idx, current_chunk_b_local_idx, :]
        del current_chunk_h_idx, current_chunk_a_local_idx, current_chunk_b_local_idx

        # Вычисление автокорреляций
        paf_a_chunk = cyclic_autocorrelation_batch(a_seqs_chunk)
        del a_seqs_chunk
        paf_b_chunk = cyclic_autocorrelation_batch(b_seqs_chunk)
        del b_seqs_chunk

        # Проверка условий Эйлера
        # проверяем, что первый элемент автокорреляций (paf_a_chunk[:, 0] и paf_b_chunk[:, 0]) для каждой из последовательностей равен v.
        cond1_chunk = (paf_a_chunk[:, 0] == v_val) & (paf_b_chunk[:, 0] == v_val)
        if v_val > 1:
            #проверяем, что сумма всех элементов автокорреляций, начиная с индекса 1 (то есть, исключая первый элемент), для каждой пары последовательностей (paf_a_chunk и paf_b_chunk) равна -2.
            cond2_chunk = cp.all(paf_a_chunk[:, 1:] + paf_b_chunk[:, 1:] == -2, axis=1)
        else:  # Для v_val == 1 paf_a_chunk[:, 1:] пуст
            cond2_chunk = cp.ones(paf_a_chunk.shape[0], dtype=cp.bool_)
        del paf_a_chunk, paf_b_chunk

        # Поиск пар, удовлетворяющих обоим условиям
        valid_euler_pairs_mask_chunk = cond1_chunk & cond2_chunk
        del cond1_chunk, cond2_chunk

        if cp.any(valid_euler_pairs_mask_chunk):
            valid_indices_in_chunk_gpu = cp.flatnonzero(valid_euler_pairs_mask_chunk)
            del valid_euler_pairs_mask_chunk

            original_flat_indices_for_valid_in_chunk = valid_indices_in_chunk_gpu + chunk_start
            del valid_indices_in_chunk_gpu

            # Извлечение валидных последовательностей
            h_coords_valid = h_flat_idx[original_flat_indices_for_valid_in_chunk]
            a_l_coords_valid = a_local_flat_idx[original_flat_indices_for_valid_in_chunk]
            b_l_coords_valid = b_local_flat_idx[original_flat_indices_for_valid_in_chunk]
            del original_flat_indices_for_valid_in_chunk

            a_seqs_validated = buckets_a[h_coords_valid, a_l_coords_valid, :]
            b_seqs_validated = buckets_b[h_coords_valid, b_l_coords_valid, :]
            del h_coords_valid, a_l_coords_valid, b_l_coords_valid

            num_validated_in_chunk = a_seqs_validated.shape[0]

            # Добавление найденных пар в результат
            for k_valid in range(num_validated_in_chunk):
                a_seq = a_seqs_validated[k_valid]
                b_seq = b_seqs_validated[k_valid]

                if not cp.array_equal(a_seq, b_seq):
                    pair = SequencePair(a_seq, b_seq, "randomized_bucketing_batched")
                    found_pair_object_list.append(pair)
                    del a_seq, b_seq
                    break

            del a_seqs_validated, b_seqs_validated
            if found_pair_object_list: break

    # Освобождение памяти и возврат результата
    del h_flat_idx, a_local_flat_idx, b_local_flat_idx
    del common_hashes_idx, current_a_counts, current_b_counts
    cp.get_default_memory_pool().free_all_blocks()

    return found_pair_object_list


# Генерирует случайные последовательности на GPU
def generate_sequences_gpu(v, k1_target, num_sequences):
    #создает матрицу размером num_sequences × v со случайными числами (0.0 - 1.0) возвращает индексы, которые бы отсортировали каждую строку:
    all_indices = cp.random.rand(num_sequences, v).argsort(axis=1)
    #создается маска
    mask = all_indices < k1_target
    #Далее создаем матрицу размером num_sequences × v, заполненную единицами (1)
    results = cp.ones((num_sequences, v), dtype=cp.int32)
    #Применяем маску results[mask] = -1.
    results[mask] = -1
    return results

# Вычисляет циклическую автокорреляцию для пакета последовательностей
def cyclic_autocorrelation_batch(seqs_gpu):
    batch_size, v_len = seqs_gpu.shape
    if v_len == 0:
        return cp.empty((batch_size, 0), dtype=cp.int32)

    data = cp.zeros((batch_size, 2 * v_len), dtype=cp.float32)
    data[:, ::2] = seqs_gpu.astype(cp.float32)

    fft_data = cp.fft.fft(data, axis=1)

    psd = cp.real(fft_data[:, :v_len]) ** 2 + cp.imag(fft_data[:, :v_len]) ** 2

    pafs = cp.fft.ifft(psd, axis=1).real
    pafs_rounded = cp.round(pafs).astype(cp.int32)

    return pafs_rounded

# Вычисляет хэш для пакета автокорреляций
def calculate_hash_n_vectorized_common(pafs_gpu, v, condition_positive):
    if v <= 1:
        return cp.zeros(pafs_gpu.shape[0], dtype=cp.int32)

    max_bit_idx = min(M_BITS_FALLBACK, v - 1)
    if max_bit_idx < 1:
        return cp.zeros(pafs_gpu.shape[0], dtype=cp.int32)

    bits_indices_for_paf = cp.arange(1, max_bit_idx + 1)

    if condition_positive:
        conditions_met = (pafs_gpu[:, bits_indices_for_paf] > 0)
    else:
        conditions_met = (pafs_gpu[:, bits_indices_for_paf] < -2)

    powers_of_2 = (1 << (bits_indices_for_paf - 1)).astype(cp.int32)

    return cp.sum(conditions_met * powers_of_2, axis=1)

# Хэш на основе pafs[k] > 0
def calculate_hash_n1_vectorized(pafs_gpu, v):
    return calculate_hash_n_vectorized_common(pafs_gpu, v, True)

# Хэш на основе pafs[k] < -2
def calculate_hash_n2_vectorized(pafs_gpu, v):
    return calculate_hash_n_vectorized_common(pafs_gpu, v, False)

# Инициализирует структуры бакетов для заданной длины последовательности
def init_buckets(v_val):
    bucket_count = 2 ** M_BITS_FALLBACK
    return (
        cp.zeros((bucket_count, MAX_BUCKET_SIZE_FALLBACK, v_val), dtype=cp.int32),
        cp.zeros((bucket_count, MAX_BUCKET_SIZE_FALLBACK, v_val), dtype=cp.int32),
        cp.zeros(bucket_count, dtype=cp.int32),  # counts_a
        cp.zeros(bucket_count, dtype=cp.int32)  # counts_b
    )

# Обновление бакетов новыми последовательностями на GPU по индексам бакетов
def update_buckets_gpu(buckets, counts, hashes, seqs, v):
    # Вычисляем количество новых элементов на бакет
    new_counts = cp.bincount(hashes, minlength=counts.shape[0])
    # Вычисляем сколько можно добавить без превышения MAX_BUCKET_SIZE_FALLBACK
    allowed = cp.minimum(new_counts, MAX_BUCKET_SIZE_FALLBACK - counts)
    allowed = cp.clip(allowed, 0, None)  # гарантируем неотрицательность
    if not bool(cp.any(allowed > 0)):
        return  # нечего добавлять

    # Сортируем последовательности по индексу бакета
    sort_idx = cp.argsort(hashes)
    sorted_hashes = hashes[sort_idx]

    # Определяем границы групп в отсортированном массиве
    mask_group_start = cp.concatenate((cp.array([True]), sorted_hashes[1:] != sorted_hashes[:-1]))
    group_starts = cp.where(mask_group_start)[0]
    group_ends = cp.concatenate((group_starts[1:], cp.array([sorted_hashes.size])))

    group_ids = cp.searchsorted(group_ends, cp.arange(sorted_hashes.size), side='right')
    pos_in_group_sorted = cp.arange(sorted_hashes.size) - cp.take(group_starts, group_ids)
    pos_in_group = cp.empty_like(pos_in_group_sorted)
    pos_in_group[sort_idx] = pos_in_group_sorted

    # Определяем, какие элементы разрешены по емкости
    allowed_per_seq = cp.take(allowed, hashes)
    mask_allowed = pos_in_group < allowed_per_seq
    idx_allowed = cp.nonzero(mask_allowed)[0]
    if idx_allowed.size == 0:
        return

    # Вычисляем позиции вставки для разрешенных элементов
    bucket_idxs = hashes[idx_allowed]
    offsets = counts[bucket_idxs] + pos_in_group[idx_allowed]

    # Вставляем новые последовательности в бакеты
    buckets[bucket_idxs, offsets, ...] = seqs[idx_allowed, ...]

    # Обновляем счетчики для каждого бакета
    counts += allowed

# Основная функция поиска пар последовательностей
def find_pairs_by_randomized_bucketing(n_val_outer , v, k1_target):
    found_pairs = []

    print(f"Поиск для N={n_val_outer} (v={v}, k1={k1_target})")
    print(
        f"Лимиты: Время <= {MAX_SEARCH_DURATION_SECONDS}s, Всего сгенерированных последовательностей <= {MAX_TOTAL_GENERATED_SEQUENCES}")

    total_sequences_generated = 0
    start_time = time.time()
    iteration_num = 0

    # Расчет количества последовательностей за итерацию
    desired_mem_gb_for_batch_generation = 1.5  # ГБ
    bytes_per_sequence_rough_estimate = v * 40  # Грубая оценка
    if bytes_per_sequence_rough_estimate == 0:
        bytes_per_sequence_rough_estimate = 40

    num_sequences_per_iteration = int(
        (desired_mem_gb_for_batch_generation * (1024 ** 3)) / bytes_per_sequence_rough_estimate)
    num_sequences_per_iteration = SEQ

    if MAX_TOTAL_GENERATED_SEQUENCES is not None:
        num_sequences_per_iteration = min(num_sequences_per_iteration, MAX_TOTAL_GENERATED_SEQUENCES)

    if num_sequences_per_iteration <= 0:
        print("Ошибка: Лимиты не позволяют сгенерировать ни одной последовательности.")
        return []

    print(f"Конфигурация итерации: будет генерироваться ~{num_sequences_per_iteration} последовательностей за раз.")

    while True:
        iteration_num += 1
        current_time_elapsed = time.time() - start_time

        # Рассчитываем количество для генерации в этой итерации
        sequences_to_generate_this_iteration = num_sequences_per_iteration
        if MAX_TOTAL_GENERATED_SEQUENCES is not None:
            remaining_can_generate = MAX_TOTAL_GENERATED_SEQUENCES - total_sequences_generated
            sequences_to_generate_this_iteration = min(num_sequences_per_iteration, remaining_can_generate)

        if sequences_to_generate_this_iteration <= 0:
            print(
                f"\nДостигнут лимит генерации последовательностей перед началом фактической генерации в итерации {iteration_num}.")
            break

        print(
            f"\n--- Итерация {iteration_num} (Всего сгенерировано: {total_sequences_generated}, Время: {current_time_elapsed:.0f}s) ---")
        print(f"Планируется сгенерировать: {sequences_to_generate_this_iteration} последовательностей.")

        try:
            # Инициализация бакетов для текущей итерации
            buckets_a, buckets_b, counts_a, counts_b = init_buckets(v)

            seqs_for_buckets = generate_sequences_gpu(v, k1_target, sequences_to_generate_this_iteration)
            valid_pafs_for_hash = cyclic_autocorrelation_batch(seqs_for_buckets)


            hashes_a = calculate_hash_n1_vectorized(valid_pafs_for_hash, v)
            hashes_b = calculate_hash_n2_vectorized(valid_pafs_for_hash, v)

            update_buckets_gpu(buckets_a, counts_a, hashes_a, seqs_for_buckets, v)
            update_buckets_gpu(buckets_b, counts_b, hashes_b, seqs_for_buckets, v)

            result_pairs = check_collisions(buckets_a, buckets_b, counts_a, counts_b, v)

            total_sequences_generated += sequences_to_generate_this_iteration

            if result_pairs:
                found_pairs.extend(result_pairs)
                print(f"\n[N={n_val_outer}, v={v}] +++ НАЙДЕНА ПАРА(ы)! Всего найдено: {len(found_pairs)}.")
                print(f"Всего сгенерировано последовательностей: {total_sequences_generated}.")
                print(f"Затраченное время: {time.time() - start_time:.2f}s.")
                return found_pairs

            print(f"Итерация {iteration_num} завершена, пар не найдено в этой итерации.")

            del valid_pafs_for_hash, seqs_for_buckets, hashes_a, hashes_b
            del buckets_a, buckets_b, counts_a, counts_b
            cp.get_default_memory_pool().free_all_blocks()

        except cp.cuda.MemoryError as e:
            print(f"\n!!! КРИТИЧЕСКАЯ ОШИБКА: Недостаточно памяти CuPy в итерации {iteration_num}: {e}")
            print(
                f"!!! Попробуйте уменьшить MAX_TOTAL_GENERATED_SEQUENCES или использовать GPU с большим объемом памяти.")
            print("!!! Поиск будет остановлен.")
            break
        except Exception as e:
            print(f"\n!!! КРИТИЧЕСКАЯ ОШИБКА: Непредвиденная ошибка в итерации {iteration_num}: {e}")
            import traceback
            traceback.print_exc()
            print("!!! Поиск будет остановлен из-за непредвиденной ошибки.")
            break

    final_duration = time.time() - start_time
    if not found_pairs:
        print(f"\nПоиск завершен БЕЗ РЕЗУЛЬТАТА.")
    else:
        print(f"\nПоиск завершен. Найдено пар: {len(found_pairs)}.")

    print(f"Всего было сгенерировано последовательностей: {total_sequences_generated}.")
    print(f"Общее время поиска: {final_duration:.2f}s.")

    return found_pairs


if __name__ == '__main__':
    time1 = time.time()
    OUTPUT_FILENAME = "balonin_" + str(N) + "_fast.js"

    print(f"Цель: Найти первую пару для n = {N} и завершить программу.")

    k1_target = calculate_k1(v)
    unique_pairs_for_n = find_pairs_by_randomized_bucketing(N , v , k1_target)

    if unique_pairs_for_n:
        print(f"Найдено {len(unique_pairs_for_n)} пар. Запись в файл {OUTPUT_FILENAME}...")
        with open(OUTPUT_FILENAME, "w") as writer:
            writer.write(f"// VERSION: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            writer.write(f"// Поиск для n={N}\n")
            writer.write(f"\nvar n={N}; function example(seq_n, k) {{\n")

            worker_args = []
            for k_idx, pair_obj in enumerate(unique_pairs_for_n):
                pair_a_np = cp.asnumpy(pair_obj.a)
                pair_b_np = cp.asnumpy(pair_obj.b)
                worker_args.append((k_idx, pair_a_np, pair_b_np, N))

            num_processes = len(worker_args) if worker_args else 0
            formatted_js_parts = []
            if num_processes > 0:
                formatted_js_parts = []
                for args_tuple in worker_args:
                    formatted_js_parts.append(format_pair_for_js_worker(args_tuple))
                for part_js in formatted_js_parts:
                    writer.write(part_js)
            else:
                for args_tuple in worker_args:
                    writer.write(format_pair_for_js_worker(args_tuple))

            writer.write(f"\n    puts(\"Pair for n=\" + n + \", k=\" + k);\n")
            writer.write(f"    puts(\"a=[\"+a+\"];\"); puts(\"b=[\"+b+\"];\");\n")
            writer.write(f"}}\n\n")
            writer.write(f"example(n, 0);\n")

        print(f"Поиск завершен. Результаты записаны в {OUTPUT_FILENAME}")
    else:
        print(f"Пары для n = {N} не найдены.")

    time2 = time.time()
    print(f"Общее время выполнения скрипта: {time2 - time1:.4f} секунд.")