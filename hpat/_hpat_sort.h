#ifndef HPAT_SORT_H_
#define HPAT_SORT_H_

#define __HPAT_MIN_MERGE_SIZE 64
#define MIN(x, y) ((x) < (y) ? (x) : (y))

#include <vector>

typedef struct
{
    uint64_t start;
    uint64_t length;
} __HPAT_TIMSORT_RUN;

typedef struct
{
    __HPAT_TIMSORT_RUN _stack[100];
    uint64_t size;
} __HPAT_TIMSORT_RUN_STACK;

typedef struct
{
    size_t size;
    int64_t* buffer;
} __HPAT_TIMSORT_TEMP_BUFFER;

int __hpat_sort_compare(int64_t x, int64_t y)
{
    if (x < y)
    {
        return -1;
    }
    else if (x == y)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

void __hpat_sort_swap(int64_t* x, int64_t* y)
{
    int64_t tmp = *x;
    *x = *y;
    *y = tmp;
}

int64_t __hpat_timsort_calcuate_minrun(int64_t size)
{
    int64_t ratio = 0;
    while (size >= __HPAT_MIN_MERGE_SIZE)
    {
        ratio |= (size & 1);
        size >>= 1;
    }
    return size + ratio;
}

//Declarations
static void __hpat_binary_insertionsort_index(
    int64_t* comp_arr, const size_t start, const size_t size, int64_t** all_arrs, const size_t all_arrs_len);
static int64_t __hpat_binary_insertionsort_search(int64_t* comp_arr, const int64_t x, const size_t size);
static void __hpat_timsort_resize_buffer(__HPAT_TIMSORT_TEMP_BUFFER* store, const size_t new_size);
void __hpat_timsort(int64_t* comp_arr, const size_t size, int64_t** all_arrs, const size_t all_arrs_len);
static void __hpat_timsort_reverse_run(
    int64_t* comp_arr, int64_t start, int64_t end, int64_t** all_arrs, const size_t all_arrs_len);
static int __hpat_timsort_getrun(int64_t* comp_arr,
                                 const size_t size,
                                 __HPAT_TIMSORT_TEMP_BUFFER* store,
                                 const uint64_t minrun,
                                 __HPAT_TIMSORT_RUN_STACK* run_stack,
                                 uint64_t* curr,
                                 int64_t** all_arrs,
                                 const size_t all_arrs_len);
static int __hpat_timsort_checkrules(__HPAT_TIMSORT_RUN_STACK* run_stack);
static int __hpat_timsort_applyrules(int64_t* comp_arr,
                                     __HPAT_TIMSORT_RUN_STACK* stack,
                                     __HPAT_TIMSORT_TEMP_BUFFER* store,
                                     const size_t size,
                                     int64_t** all_arrs,
                                     const size_t all_arrs_len);
static int64_t __hpat_timsort_count_run(
    int64_t* comp_arr, const uint64_t start, const size_t size, int64_t** all_arrs, const size_t all_arrs_len);
static void __hpat_timsort_merge_run(int64_t* comp_arr,
                                     const __HPAT_TIMSORT_RUN_STACK* run_stack,
                                     __HPAT_TIMSORT_TEMP_BUFFER* store,
                                     int64_t** all_arrs,
                                     const size_t all_arrs_len);

// Basic binary search to find the right position
static int64_t __hpat_binary_insertionsort_search(int64_t* comp_arr, const int64_t elem, const size_t size)
{
    int64_t low, mid, high, pivot;
    low = 0;
    high = size - 1;
    mid = high >> 1;
    // If it is less than low
    if (__hpat_sort_compare(elem, comp_arr[0]) < 0)
    {
        return 0;
    }
    else if (__hpat_sort_compare(elem, comp_arr[high]) > 0)
    {
        return high;
    }
    pivot = comp_arr[mid];
    while (1)
    {
        if (__hpat_sort_compare(elem, pivot) < 0)
        {
            if (mid - low <= 1)
            {
                return mid;
            }
            high = mid;
        }
        else
        {
            if (high - mid <= 1)
            {
                return mid + 1;
            }
            low = mid;
        }
        mid = low + ((high - low) >> 1);
        pivot = comp_arr[mid];
    }
}

// Binary search with different starting index
static void __hpat_binary_insertionsort_index(
    int64_t* comp_arr, const size_t start, const size_t size, int64_t** all_arrs, const size_t all_arrs_len)
{
    for (size_t ind_start = start; ind_start < size; ind_start++)
    {
        int64_t ind_curr, elem, pivot;
        // Already sorted
        if (__hpat_sort_compare(comp_arr[ind_start - 1], comp_arr[ind_start]) <= 0)
        {
            continue;
        }
        elem = comp_arr[ind_start];
        std::vector<int64_t> temp_x(all_arrs_len);
        for (size_t k = 0; k < all_arrs_len; k++)
        {
            int64_t* cur_arr = all_arrs[k];
            temp_x[k] = cur_arr[ind_start];
        }
        pivot = __hpat_binary_insertionsort_search(comp_arr, elem, ind_start);
        for (ind_curr = ind_start - 1; ind_curr >= pivot; ind_curr--)
        {
            //std::cout << "moving forward" << std::endl;
            comp_arr[ind_curr + 1] = comp_arr[ind_curr];
            for (size_t k = 0; k < all_arrs_len; k++)
            {
                int64_t* cur_arr = all_arrs[k];
                cur_arr[ind_curr + 1] = cur_arr[ind_curr];
            }
        }
        comp_arr[pivot] = elem;
        for (size_t k = 0; k < all_arrs_len; k++)
        {
            int64_t* cur_arr = all_arrs[k];
            cur_arr[pivot] = temp_x[k];
        }
    }
}

void __hpat_timsort(int64_t* comp_arr, const size_t size, int64_t** all_arrs, const size_t all_arrs_len)
{
    if (size <= 1)
    {
        return;
    }
    if (size < __HPAT_MIN_MERGE_SIZE)
    {
        __hpat_binary_insertionsort_index(comp_arr, 1, size, all_arrs, all_arrs_len);
        return;
    }

    uint64_t minrun;
    __HPAT_TIMSORT_TEMP_BUFFER _store, *store;
    __HPAT_TIMSORT_RUN_STACK run_stack;
    run_stack.size = 0;
    uint64_t curr = 0;

    minrun = __hpat_timsort_calcuate_minrun(size);
    // temporary buffer for merge
    store = &_store;
    store->size = 0;
    store->buffer = NULL;

    if (!__hpat_timsort_getrun(comp_arr, size, store, minrun, &run_stack, &curr, all_arrs, all_arrs_len))
    {
        return;
    }

    if (!__hpat_timsort_getrun(comp_arr, size, store, minrun, &run_stack, &curr, all_arrs, all_arrs_len))
    {
        return;
    }

    if (!__hpat_timsort_getrun(comp_arr, size, store, minrun, &run_stack, &curr, all_arrs, all_arrs_len))
    {
        return;
    }
    while (1)
    {
        if (!__hpat_timsort_checkrules(&run_stack))
        {
            run_stack.size = __hpat_timsort_applyrules(comp_arr, &run_stack, store, size, all_arrs, all_arrs_len);
            continue;
        }
        if (!__hpat_timsort_getrun(comp_arr, size, store, minrun, &run_stack, &curr, all_arrs, all_arrs_len))
        {
            return;
        }
    }
}

static int __hpat_timsort_checkrules(__HPAT_TIMSORT_RUN_STACK* run_stack)
{
    /*
      Rules are taken from https://en.wikipedia.org/wiki/Timsort
      X > Y + Z
      Y > Z
    */
    if (run_stack->size < 2)
    {
        return 1;
    }
    int64_t X, Y, Z;
    if (run_stack->size == 2)
    {
        const int64_t X = run_stack->_stack[run_stack->size - 2].length;
        const int64_t Y = run_stack->_stack[run_stack->size - 1].length;
        if (X <= Y)
        {
            return 0;
        }
        return 1;
    }
    X = run_stack->_stack[run_stack->size - 3].length;
    Y = run_stack->_stack[run_stack->size - 2].length;
    Z = run_stack->_stack[run_stack->size - 1].length;
    if ((X <= Y + Z) || (Y <= Z))
    {
        // Rules are not satisfied
        return 0;
    }
    return 1;
}

static void __hpat_timsort_reverse_run(
    int64_t* comp_arr, int64_t start, int64_t end, int64_t** all_arrs, const size_t all_arrs_len)
{
    while (1)
    {
        if (start >= end)
        {
            return;
        }
        __hpat_sort_swap(&comp_arr[start], &comp_arr[end]);
        for (size_t k = 0; k < all_arrs_len; k++)
        {
            int64_t* curr_arr = all_arrs[k];
            __hpat_sort_swap(&curr_arr[start], &curr_arr[end]);
        }
        start++;
        end--;
    } // while
}

static int64_t __hpat_timsort_count_run(
    int64_t* comp_arr, const uint64_t start, const size_t size, int64_t** all_arrs, const size_t all_arrs_len)
{
    uint64_t run_count = start + 2;
    ;
    if (size - start == 1)
    {
        return 1;
    }
    // only two elements left check and swap
    if (start >= size - 2)
    {
        if (__hpat_sort_compare(comp_arr[size - 2], comp_arr[size - 1]) > 0)
        {
            __hpat_sort_swap(&comp_arr[size - 2], &comp_arr[size - 1]);
            for (size_t k = 0; k < all_arrs_len; k++)
            {
                int64_t* curr_arr = all_arrs[k];
                __hpat_sort_swap(&curr_arr[size - 2], &curr_arr[size - 1]);
            }
        }
        return 2;
    }
    // Finding ascending run
    if (__hpat_sort_compare(comp_arr[start], comp_arr[start + 1]) <= 0)
    {
        while ((run_count != size - 1) && (__hpat_sort_compare(comp_arr[run_count - 1], comp_arr[run_count]) <= 0))
        {
            run_count++;
        }
        return run_count - start;
    }
    else
    {
        // Finding strictly descending run
        while ((run_count != size - 1) && (__hpat_sort_compare(comp_arr[run_count - 1], comp_arr[run_count]) > 0))
        {
            run_count++;
        }
        // descending run should be reversed
        __hpat_timsort_reverse_run(comp_arr, start, run_count - 1, all_arrs, all_arrs_len);
        return run_count - start;
    }
}

static int __hpat_timsort_getrun(int64_t* comp_arr,
                                 const size_t size,
                                 __HPAT_TIMSORT_TEMP_BUFFER* store,
                                 const uint64_t minrun,
                                 __HPAT_TIMSORT_RUN_STACK* run_stack,
                                 uint64_t* curr,
                                 int64_t** all_arrs,
                                 const size_t all_arrs_len)
{
    uint64_t run_size = __hpat_timsort_count_run(comp_arr, *curr, size, all_arrs, all_arrs_len);
    uint64_t run = minrun;
    if (run > size - *curr)
    {
        run = size - *curr;
    }
    if (run > run_size)
    {
        std::vector<int64_t*> temp_all_arrs(all_arrs_len);
        // As we are starting from different index/start
        for (size_t k = 0; k < all_arrs_len; k++)
        {
            int64_t* curr_arr = all_arrs[k];
            temp_all_arrs[k] = &curr_arr[*curr];
        }
        __hpat_binary_insertionsort_index(&comp_arr[*curr], run_size, run, temp_all_arrs.data(), all_arrs_len);
        run_size = run;
    }
    run_stack->_stack[run_stack->size].start = *curr;
    run_stack->_stack[run_stack->size].length = run_size;
    run_stack->size++;
    *curr += run_size;
    if (*curr == size)
    {
        // Done with all runs
        while (run_stack->size > 1)
        {
            __hpat_timsort_merge_run(comp_arr, run_stack, store, all_arrs, all_arrs_len);
            run_stack->_stack[run_stack->size - 2].length += run_stack->_stack[run_stack->size - 1].length;
            run_stack->size--;
        }
        if (store->buffer != NULL)
        {
            free(store->buffer);
            store->buffer = NULL;
        }
        return 0;
    }
    return 1;
}

static int __hpat_timsort_applyrules(int64_t* comp_arr,
                                     __HPAT_TIMSORT_RUN_STACK* run_stack,
                                     __HPAT_TIMSORT_TEMP_BUFFER* store,
                                     const size_t size,
                                     int64_t** all_arrs,
                                     const size_t all_arrs_len)
{
    /*
      RULE 1 = X > Y + Z
      RULE 2 = Y > Z
  */
    while (run_stack->size > 1)
    {
        int64_t X, Y, Z, ZZ;
        int rule_1, rule_2, rule_zz;
        // only two runs in stack merge them
        if ((run_stack->size == 2) && (run_stack->_stack[0].length + run_stack->_stack[1].length == size))
        {
            __hpat_timsort_merge_run(comp_arr, run_stack, store, all_arrs, all_arrs_len);
            run_stack->_stack[0].length += run_stack->_stack[1].length;
            run_stack->size--;
            break;
        }
        // Check Rule 2 for only two elements
        else if ((run_stack->size == 2) && (run_stack->_stack[0].length <= run_stack->_stack[1].length))
        {
            __hpat_timsort_merge_run(comp_arr, run_stack, store, all_arrs, all_arrs_len);
            run_stack->_stack[0].length += run_stack->_stack[1].length;
            run_stack->size--;
            break;
        }
        else if (run_stack->size == 2)
        {
            break;
        }

        X = run_stack->_stack[run_stack->size - 3].length;
        Y = run_stack->_stack[run_stack->size - 2].length;
        Z = run_stack->_stack[run_stack->size - 1].length;

        if (run_stack->size >= 4)
        {
            ZZ = run_stack->_stack[run_stack->size - 4].length;
            rule_zz = (ZZ <= X + Y);
        }
        else
        {
            rule_zz = 0;
        }

        rule_1 = (X <= Y + Z) || rule_zz;
        rule_2 = (Y <= Z);

        // Rules hold
        if (!rule_1 && !rule_2)
        {
            break;
        }

        /* left merge */
        if (rule_1 && !rule_2)
        {
            int64_t curr_size = run_stack->size;
            run_stack->size--;
            __hpat_timsort_merge_run(comp_arr, run_stack, store, all_arrs, all_arrs_len);
            run_stack->_stack[curr_size - 3].length += run_stack->_stack[curr_size - 2].length;
            run_stack->_stack[curr_size - 2] = run_stack->_stack[curr_size - 1];
        }
        else
        {
            /* right merge */
            __hpat_timsort_merge_run(comp_arr, run_stack, store, all_arrs, all_arrs_len);
            run_stack->_stack[run_stack->size - 2].length += run_stack->_stack[run_stack->size - 1].length;
            run_stack->size--;
        }
    }

    return run_stack->size;
}

static void __hpat_timsort_resize_buffer(__HPAT_TIMSORT_TEMP_BUFFER* store, const size_t new_size)
{
    if (store->size < new_size)
    {
        int64_t* temp_buffer = (int64_t*)realloc(store->buffer, new_size * sizeof(int64_t));
        store->buffer = temp_buffer;
        store->size = new_size;
    }
}
static void __hpat_timsort_mergeleft_run(int64_t* comp_arr,
                                         const int64_t run1_len,
                                         const int64_t run2_len,
                                         int64_t* temp_buffer,
                                         const int64_t stack_ptr,
                                         int64_t** all_arrs,
                                         const size_t all_arrs_len,
                                         const size_t min_len)
{
    // array layout [ run 1 || run 2]
    int64_t temp_buffer_ind, run2_low, curr;
    // Make temporary copy
    memcpy(temp_buffer, &comp_arr[stack_ptr], run1_len * sizeof(int64_t));

    std::vector<int64_t*> temp_x(all_arrs_len);
    for (size_t k = 0; k < all_arrs_len; k++)
    {
        int64_t* curr_arr = all_arrs[k];
        temp_x[k] = (int64_t*)malloc(sizeof(int64_t) * min_len);
        memcpy(temp_x[k], &curr_arr[stack_ptr], run1_len * sizeof(int64_t));
    }
    temp_buffer_ind = 0;
    run2_low = stack_ptr + run1_len;
    int64_t total_len = stack_ptr + run1_len + run2_len;
    for (curr = stack_ptr; curr < total_len; curr++)
    {
        if ((temp_buffer_ind < run1_len) && (run2_low < total_len))
        {
            if (__hpat_sort_compare(temp_buffer[temp_buffer_ind], comp_arr[run2_low]) <= 0)
            {
                comp_arr[curr] = temp_buffer[temp_buffer_ind];

                for (size_t k = 0; k < all_arrs_len; k++)
                {
                    int64_t* curr_arr = all_arrs[k];
                    int64_t* temp_curr_arr = temp_x[k];
                    curr_arr[curr] = temp_curr_arr[temp_buffer_ind];
                }
                temp_buffer_ind++;
            }
            else
            {
                comp_arr[curr] = comp_arr[run2_low];

                for (size_t k = 0; k < all_arrs_len; k++)
                {
                    int64_t* curr_arr = all_arrs[k];
                    curr_arr[curr] = curr_arr[run2_low];
                }
                run2_low++;
            }
        }
        else if (temp_buffer_ind < run1_len)
        {
            comp_arr[curr] = temp_buffer[temp_buffer_ind];

            for (size_t k = 0; k < all_arrs_len; k++)
            {
                int64_t* curr_arr = all_arrs[k];
                int64_t* temp_curr_arr = temp_x[k];
                curr_arr[curr] = temp_curr_arr[temp_buffer_ind];
            }
            temp_buffer_ind++;
        }
        else
        {
            comp_arr[curr] = comp_arr[run2_low];

            for (size_t k = 0; k < all_arrs_len; k++)
            {
                int64_t* curr_arr = all_arrs[k];
                curr_arr[curr] = curr_arr[run2_low];
            }
            run2_low++;
        }
    }
    // deleting malloc
    for (size_t k = 0; k < all_arrs_len; k++)
    {
        free(temp_x[k]);
    }
}
static void __hpat_timsort_mergeright_run(int64_t* comp_arr,
                                          const int64_t run1_len,
                                          const int64_t run2_len,
                                          int64_t* temp_buffer,
                                          const int64_t stack_ptr,
                                          int64_t** all_arrs,
                                          const size_t all_arrs_len,
                                          const size_t min_len)
{
    // array layout [ run 1 || run 2]
    int64_t temp_buffer_ind, run1_high, curr;
    // Make temporary copy
    memcpy(temp_buffer, &comp_arr[stack_ptr + run1_len], run2_len * sizeof(int64_t));

    std::vector<int64_t*> temp_x(all_arrs_len);
    for (size_t k = 0; k < all_arrs_len; k++)
    {
        int64_t* curr_arr = all_arrs[k];
        temp_x[k] = (int64_t*)malloc(sizeof(int64_t) * min_len);
        memcpy(temp_x[k], &curr_arr[stack_ptr + run1_len], run2_len * sizeof(int64_t));
    }

    temp_buffer_ind = run2_len - 1;
    run1_high = stack_ptr + run1_len - 1;
    int64_t total_len = stack_ptr + run1_len + run2_len;

    for (curr = total_len - 1; curr >= stack_ptr; curr--)
    {
        if ((temp_buffer_ind >= 0) && (run1_high >= stack_ptr))
        {
            if (__hpat_sort_compare(comp_arr[run1_high], temp_buffer[temp_buffer_ind]) > 0)
            {
                comp_arr[curr] = comp_arr[run1_high];

                for (size_t k = 0; k < all_arrs_len; k++)
                {
                    int64_t* curr_arr = all_arrs[k];
                    curr_arr[curr] = curr_arr[run1_high];
                }
                run1_high--;
            }
            else
            {
                comp_arr[curr] = temp_buffer[temp_buffer_ind];

                for (size_t k = 0; k < all_arrs_len; k++)
                {
                    int64_t* curr_arr = all_arrs[k];
                    int64_t* temp_curr_arr = temp_x[k];
                    curr_arr[curr] = temp_curr_arr[temp_buffer_ind];
                }
                temp_buffer_ind--;
            }
        }
        else if (temp_buffer_ind >= 0)
        {
            comp_arr[curr] = temp_buffer[temp_buffer_ind];

            for (size_t k = 0; k < all_arrs_len; k++)
            {
                int64_t* curr_arr = all_arrs[k];
                int64_t* temp_curr_arr = temp_x[k];
                curr_arr[curr] = temp_curr_arr[temp_buffer_ind];
            }
            temp_buffer_ind--;
        }
        else
        {
            comp_arr[curr] = comp_arr[run1_high];

            for (size_t k = 0; k < all_arrs_len; k++)
            {
                int64_t* curr_arr = all_arrs[k];
                curr_arr[curr] = curr_arr[run1_high];
            }
            run1_high--;
        }
    }
    // deleting malloc
    for (size_t k = 0; k < all_arrs_len; k++)
    {
        free(temp_x[k]);
    }
}

static void __hpat_timsort_merge_run(int64_t* comp_arr,
                                     const __HPAT_TIMSORT_RUN_STACK* run_stack,
                                     __HPAT_TIMSORT_TEMP_BUFFER* store,
                                     int64_t** all_arrs,
                                     const size_t all_arrs_len)
{
    const int64_t len1 = run_stack->_stack[run_stack->size - 2].length;
    const int64_t len2 = run_stack->_stack[run_stack->size - 1].length;
    const int64_t stack_ptr = run_stack->_stack[run_stack->size - 2].start;
    int64_t* temp_buffer;
    int64_t min_len = MIN(len1, len2);
    __hpat_timsort_resize_buffer(store, min_len);
    temp_buffer = store->buffer;
    if (len1 < len2)
    {
        __hpat_timsort_mergeleft_run(comp_arr, len1, len2, temp_buffer, stack_ptr, all_arrs, all_arrs_len, min_len);
    }
    else
    {
        __hpat_timsort_mergeright_run(comp_arr, len1, len2, temp_buffer, stack_ptr, all_arrs, all_arrs_len, min_len);
    }
}

int __hpat_quicksort_partition(int64_t** arr, int64_t size, int64_t* comp_arr, int low, int high)
{
    int64_t pivot, t;
    int i, j;
    pivot = comp_arr[low];
    i = low;
    j = high + 1;
    while (1)
    {
        do
        {
            ++i;
        } while (comp_arr[i] <= pivot && i <= high);
        do
        {
            --j;
        } while (comp_arr[j] > pivot);
        if (i >= j)
        {
            break;
        }
        for (int index = 0; index < size; index++)
        {
            int64_t* curr_arr = arr[index];
            t = curr_arr[i];
            curr_arr[i] = curr_arr[j];
            curr_arr[j] = t;
        }
        t = comp_arr[i];
        comp_arr[i] = comp_arr[j];
        comp_arr[j] = t;
    }
    for (int index = 0; index < size; index++)
    {
        int64_t* curr_arr = arr[index];
        t = curr_arr[low];
        curr_arr[low] = curr_arr[j];
        curr_arr[j] = t;
        //__hpat_quicksort_swap(curr_arr[i],curr_arr[j]);
    }
    t = comp_arr[low];
    comp_arr[low] = comp_arr[j];
    comp_arr[j] = t;
    return j;
}

void __hpat_quicksort(int64_t** arr, int size, int64_t* comp_arr, int low, int high)
{
    int pivot;
    if (low < high)
    {
        pivot = __hpat_quicksort_partition(arr, size, comp_arr, low, high);
        __hpat_quicksort(arr, size, comp_arr, low, pivot - 1);
        __hpat_quicksort(arr, size, comp_arr, pivot + 1, high);
    }
}

#endif /* HPAT_SORT_H_ */
