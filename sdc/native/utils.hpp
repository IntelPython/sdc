#pragma once

#include <cstdint>
#include <algorithm>
#include "tbb/task_arena.h"

namespace utils
{
using quant = uint8_t;

template<class T>
inline T* advance(T* ptr, int64_t pos, int64_t size)
{
    (void)size;
    return ptr + pos;
}

template<>
inline void* advance(void* ptr, int64_t pos, int64_t size)
{
    return reinterpret_cast<quant*>(ptr) + pos*size;
}

template<class T>
inline uint64_t distance(T* start, T* end, int64_t size)
{
    (void)size;
    return end - start;
}


template<>
inline uint64_t distance(void* start, void* end, int64_t size)
{
    return (reinterpret_cast<quant*>(end) - reinterpret_cast<quant*>(start))/size;
}

template<uint64_t item_size> struct exact_void_data_type;

template<> struct exact_void_data_type<1> { using type = uint8_t;  };
template<> struct exact_void_data_type<2> { using type = uint16_t; };
template<> struct exact_void_data_type<4> { using type = uint32_t; };
template<> struct exact_void_data_type<8> { using type = uint64_t; };

template<uint64_t item_size> struct exact_void_data;
template<uint64_t item_size> struct void_data;

template<uint64_t item_size>
exact_void_data<item_size>& copy(const exact_void_data<item_size>& src, exact_void_data<item_size>& dst)
{
    *dst.ptr = *src.ptr;
    return dst;
}

template<uint64_t item_size>
void_data<item_size>& copy(const void_data<item_size>& src, void_data<item_size>& dst)
{
    using data_type = typename void_data<item_size>::data_type;
    std::copy_n(reinterpret_cast<data_type*>(src.ptr),
                src.actual_size,
                reinterpret_cast<data_type*>(dst.ptr));

    dst.actual_size = src.actual_size;

    return dst;
}

template<uint64_t item_size, class data_type>
struct void_range
{
    void_range(void* begin, uint64_t len, uint64_t size)
    {
        _begin = begin;
        _end   = advance(begin, len, size);
        _size  = size;
    }

    class iterator
    {
    public:
        using value_type        = data_type;
        using difference_type   = int64_t;
        using reference         = data_type&;
        using pointer           = data_type*;
        using iterator_category = std::random_access_iterator_tag;

        iterator(void* ptr, uint64_t size): _ptr(ptr), _size(size) {}

        iterator(const iterator& other): _ptr(other._ptr), _size(other._size) {}

        iterator& operator ++()
        {
            _ptr = advance(_ptr, 1, _size);

            return *this;
        }

        iterator operator ++(int)
        {
            iterator result(*this);
            _ptr = advance(_ptr, 1, _size);

            return result;
        }

        iterator& operator --()
        {
            _ptr = advance(_ptr, -1, _size);

            return *this;
        }

        iterator operator --(int)
        {
            iterator result(*this);
            _ptr = advance(_ptr, -1, _size);

            return result;
        }

        const data_type operator * () const
        {
            return data_type(_ptr, _size);
        }

        data_type operator * ()
        {
            return data_type(_ptr, _size);
        }

        size_t operator - (const iterator& other) const
        {
            return distance(other._ptr, _ptr, _size);
        }

        iterator& operator += (difference_type shift)
        {
            _ptr = advance(_ptr, shift, _size);
            return *this;
        }

        iterator operator + (difference_type shift) const
        {
            auto r = iterator(_ptr, _size);
            return r += shift;
        }

        iterator& operator -= (difference_type shift)
        {
            return *this += (-shift);
        }

        iterator operator - (difference_type shift) const
        {
            auto r = iterator(_ptr, _size);
            return r -= shift;
        }

        bool operator > (const iterator& rhs) const
        {
            return _ptr > rhs._ptr;
        }

        bool operator < (const iterator& rhs) const
        {
            return _ptr < rhs._ptr;
        }

        bool operator == (const iterator& rhs) const
        {
            return _ptr == rhs._ptr;
        }

        bool operator != (const iterator& rhs) const
        {
            return !(*this == rhs);
        }

        bool operator >= (const iterator& rhs) const
        {
            return _ptr >= rhs._ptr;
        }

        bool operator <= (const iterator& rhs) const
        {
            return _ptr <= rhs._ptr;
        }

        data_type operator[] (int i)
        {
            return *(*this + i);
        }

        const data_type operator[] (int i) const
        {
            return *(*this + i);
        }

    private:
        void*    _ptr  = nullptr;
        uint64_t _size = 0;
    };

    iterator begin()
    {
        return iterator(_begin, _size);
    }

    iterator end()
    {
        return iterator(_end, _size);
    }

private:

    void*    _begin = nullptr;
    void*    _end   = nullptr;
    uint64_t _size  = 0;
};

template<uint64_t item_size>
struct exact_void_data
{
    using data_type = typename exact_void_data_type<item_size>::type;

    exact_void_data() { }

    explicit exact_void_data(void* in_ptr, uint64_t) { ptr = reinterpret_cast<data_type*>(in_ptr); }

    exact_void_data(const exact_void_data& other) { copy(other, *this); }

    exact_void_data& operator = (const exact_void_data& rhs) { return copy(rhs, *this); }

    operator void* () { return ptr; }

    operator const void* () const { return ptr; }

    data_type  data = 0;
    data_type* ptr  = &data;
};

template<uint64_t item_size>
struct void_data
{
    using data_type     = quant;
    using data_type_arr = data_type[item_size];

    void_data() { }

    explicit void_data(void* in_ptr, uint64_t in_size)
    {
        ptr = in_ptr;
        actual_size = in_size;
    }

    void_data(const void_data& other) { copy(other, *this); }

    void_data& operator = (const void_data& rhs) { return copy(rhs, *this); }

    operator void* () { return ptr; }

    operator const void* () const { return ptr; }

    data_type_arr data        = {};
    void*         ptr         = &data;
    uint64_t      actual_size = 0;
};

using compare_func = bool (*)(const void*, const void*);
template<uint64_t size> using exact_void_range = void_range<size, exact_void_data<size>>;
template<uint64_t size> using _void_range = void_range<size, void_data<size>>;

template<uint64_t size>
void swap(exact_void_data<size> a, exact_void_data<size> b)
{
    std::swap(*a.ptr, *b.ptr);
}

template<uint64_t size>
void swap(void_data<size> a, void_data<size> b)
{
    auto tmp(a);
    copy(b, a);
    copy(tmp, b);
}

template<class T>
T* upper_bound(T* first, T* last, T* value, int size, int item_size, void* compare)
{
    T* it = nullptr;
    auto count = size;

    auto less = reinterpret_cast<compare_func>(compare);

    if (less)
    {
        while (count > 0) {
            it = first;
            auto step = count / 2;
            it = advance(it, step, item_size);
            if (!less(value, it)) {
                first = advance(it, 1, item_size);
                count -= step + 1;
            }
            else
                count = step;
        }
    }
    else
    {
        while (count > 0) {
            it = first;
            auto step = count / 2;
            it = advance(it, step, item_size);
            if (!(*value < *it)) {
                first = advance(it, 1, item_size);
                count -= step + 1;
            }
            else
                count = step;
        }
    }

    return first;
}

template<>
void* upper_bound(void* first, void* last, void* value, int size, int item_size, void* compare);

tbb::task_arena& get_arena();

void set_threads_num(uint64_t);

} // namespace
