#include "utils.hpp"
#include "tbb/parallel_invoke.h"
#include <iostream>

using namespace utils;

namespace
{

template<class T>
struct buffer_queue
{
    using v_type = T;
    v_type* head;
    v_type* tail;

    buffer_queue(v_type* _head, int size, int _item_size)
    {
        (void)_item_size;

        head = _head;
        tail = head + size;
    }

    inline v_type* pop() { return head++; }

    inline bool not_empty() { return head < tail; }

    inline void push(v_type* val) { *(tail++) = *val; }

    inline int size() { return tail - head; }

    inline int copy_size() { return size(); }
};

template<>
struct buffer_queue<void>
{
    using v_type = void;
    quant* head;
    quant* tail;
    const int item_size;

    buffer_queue(v_type* _head, int size, int _item_size) : item_size(_item_size)
    {
        head = as_quant(_head);
        tail = head + size*item_size;
    }

    inline v_type* pop()
    {
        auto _h = head;
        head += item_size;
        return _h;
    }

    inline bool not_empty() { return head < tail; }

    inline void push(v_type* val)
    {
        std::copy_n(as_quant(val),
                    item_size,
                    tail);

        tail += item_size;
    }

    inline int size() { return copy_size()/item_size; }

    inline int copy_size() { return tail - head; }

    inline quant* as_quant(v_type* data) { return reinterpret_cast<quant*>(data); }
};

template<class T>
inline void merge_sorted_main_loop(buffer_queue<T>& left, buffer_queue<T>& right, buffer_queue<T>& out, void* compare = nullptr)
{
    if (compare)
    {
        auto less = reinterpret_cast<compare_func>(compare);
        while (left.not_empty() && right.not_empty())
        {
            if (less(right.head, left.head))
                out.push(right.pop());
            else
                out.push(left.pop());
        }
    }
    else
    {
        while (left.not_empty() && right.not_empty())
        {
            if (*right.head < *left.head)
                out.push(right.pop());
            else
                out.push(left.pop());
        }
    }
}


template<>
inline void merge_sorted_main_loop<void>(buffer_queue<void>& left, buffer_queue<void>& right, buffer_queue<void>& out, void* compare)
{
    if (!compare)
    {
        std::cout << "compare is nullptr for non-aritmetic type" << std::endl;
        abort();
    }

    auto less = reinterpret_cast<compare_func>(compare);
    while (left.not_empty() && right.not_empty())
    {
        if (less(right.head, left.head))
            out.push(right.pop());
        else
            out.push(left.pop());
    }
}

template<class T>
void merge_sorted(T* left, int left_size, T* right, int right_size, int item_size, T* out, void* compare = nullptr)
{
    auto left_buffer  = buffer_queue<T>(left,  left_size,  item_size);
    auto right_buffer = buffer_queue<T>(right, right_size, item_size);

    auto out_buffer = buffer_queue<T>(out, 0, item_size);

    merge_sorted_main_loop(left_buffer, right_buffer, out_buffer, compare);

    // only one buffer still have items, don't need to shift out_buffer.tail
    std::copy_n(left_buffer.head, left_buffer.copy_size(), out_buffer.tail);

    if (out_buffer.tail != right_buffer.head)
        std::copy_n(right_buffer.head, right_buffer.copy_size(), out_buffer.tail);
}

template<class T>
void merge_sorted_parallel(T* left, int left_size, T* right, int right_size, int item_size, T* out, void* compare = nullptr)
{
    auto split = [](T* first, int f_size, T* second, int s_size, int item_size, T* out, void* compare = nullptr)
    {
        auto f_middle_pos = f_size/2;

        auto first_middle = advance(first,  f_middle_pos, item_size);
        auto second_end   = advance(second, s_size,  item_size);

        auto second_middle = upper_bound(second, second_end, first_middle, s_size, item_size, compare);

        auto s_middle_pos = distance(second, second_middle, item_size);

        auto out_middle = advance(out, f_middle_pos + s_middle_pos, item_size);

        tbb::parallel_invoke(
            [&] () { merge_sorted_parallel<T>(first, f_middle_pos, second, s_middle_pos, item_size, out, compare); },
            [&] () { merge_sorted_parallel<T>(first_middle, f_size - f_middle_pos, second_middle, s_size - s_middle_pos, item_size, out_middle, compare); }
        );
    };

    auto constexpr limit = 512;
    if (left_size >= right_size && left_size > limit)
    {
        split(left, left_size, right, right_size, item_size, out, compare);
    }
    else if (left_size < right_size && right_size > limit)
    {
        split(right, right_size, left, left_size, item_size, out, compare);
    }
    else
    {
        merge_sorted<T>(left, left_size, right, right_size, item_size, out, compare);
    }
}

template<class T>
void stable_sort_inner_sort(T* data, int begin, int end, int item_size, void* compare)
{
    auto less = reinterpret_cast<compare_func>(compare);

    if (less == nullptr)
    {
        std::stable_sort(data + begin, data + end);
    }
    else
    {
        auto range  = exact_void_range<sizeof(T)>(reinterpret_cast<void*>(data), end, sizeof(T));
        auto _begin = range.begin() + begin;
        auto _end   = range.end();

        std::stable_sort(_begin, _end, less);
    }
}

template<>
void stable_sort_inner_sort<void>(void* data, int begin, int end, int item_size, void* compare)
{
    auto less = reinterpret_cast<compare_func>(compare);

    if (less == nullptr)
    {
        std::cout << "compare is nullptr for non-aritmetic type" << std::endl;
        abort();
    }

    // TODO move this dispatch to upper level
#define run_sort(range_type) \
{ \
    auto range  = range_type(data, end, item_size); \
    auto _begin = range.begin() + begin; \
    auto _end   = range.end(); \
    std::stable_sort(_begin, _end, less); \
}

    switch(item_size)
    {
    case 1:
        run_sort(exact_void_range<1>);
        break;
    case 2:
        run_sort(exact_void_range<2>);
        break;
    case 4:
        run_sort(exact_void_range<4>);
        break;
    case 8:
        run_sort(exact_void_range<8>);
        break;
    default:
        // fallback to own implementation?
        if      (item_size <= 4)    run_sort(_void_range<4>)
        else if (item_size <= 8)    run_sort(_void_range<8>)
        else if (item_size <= 16)   run_sort(_void_range<16>)
        else if (item_size <= 32)   run_sort(_void_range<32>)
        else if (item_size <= 64)   run_sort(_void_range<64>)
        else if (item_size <= 128)  run_sort(_void_range<128>)
        else if (item_size <= 256)  run_sort(_void_range<256>)
        else if (item_size <= 512)  run_sort(_void_range<512>)
        else if (item_size <= 1024) run_sort(_void_range<1024>)
        else
        {
            std::cout << "Unsupported item size " << item_size << std::endl;
            abort();
        }
        break;
    }

#undef run_sort
}

template<class T>
T* stable_sort_impl(T* data, T* temp, int begin, int end, int item_size, void* compare)
{
    auto constexpr limit = 512;
    if (end - begin <= limit)
    {
        stable_sort_inner_sort<T>(data, begin, end, item_size, compare);

        return data;
    }
    auto middle = begin + (end - begin) / 2;

    T* left = nullptr;
    T* right = nullptr;

    tbb::parallel_invoke(
        [&] () { left  = stable_sort_impl<T>(data, temp, begin,  middle, item_size, compare); },
        [&] () { right = stable_sort_impl<T>(data, temp, middle, end,    item_size, compare); }
    );

    auto out = data;

    if (left == data)
        out = temp;

    merge_sorted_parallel<T>(advance(left, begin, item_size),
                             middle - begin,
                             advance(right, middle, item_size),
                             end - middle,
                             item_size,
                             advance(out, begin, item_size),
                             compare);

    return out;
}

template<class T>
void parallel_stable_sort_(T* data, int size, int item_size, void* compare)
{
    std::unique_ptr<quant[]> temp(new quant[size*item_size]);

    auto result = stable_sort_impl<T>(data, reinterpret_cast<T*>(temp.get()), 0, size, item_size, compare);

    if (reinterpret_cast<quant*>(result) == temp.get())
    {
        std::copy_n(reinterpret_cast<quant*>(result), size*item_size, reinterpret_cast<quant*>(data));
    }
}

template<class T>
inline void parallel_stable_sort__(T* data, int size)
{
    return parallel_stable_sort_<T>(data, size, sizeof(T), nullptr);
}

} // namespace

#define declare_sort(prefix, ty) \
void parallel_stable_sort_##prefix(void* begin, uint64_t len) { parallel_stable_sort__<ty>(reinterpret_cast<ty*>(begin), len); }

#define declare_int_sort(bits) \
declare_sort(i##bits, int##bits##_t) \
declare_sort(u##bits, uint##bits##_t)

extern "C"
{

declare_int_sort(8)
declare_int_sort(16)
declare_int_sort(32)
declare_int_sort(64)

declare_sort(f32, float)
declare_sort(f64, double)

void parallel_stable_sort(void* begin, uint64_t len, uint64_t size, void* compare)
{
    parallel_stable_sort_<void>(begin, len, size, compare);
}

}

#undef declare_int_sort
#undef declare_sort
