#include <algorithm>
#include <cstdint>

#include <vector>

#include <iostream>
#include <chrono>
#include "tbb/parallel_sort.h"

#define TBB_PREVIEW_GLOBAL_CONTROL 1
#include "tbb/global_control.h"
#include "tbb/task_scheduler_init.h"
#include <stdexcept>

extern "C"
{
    void parallel_sort(void* begin, uint64_t len, uint64_t size, void* compare);

    void parallel_sort_i8(void* begin, uint64_t len);
    void parallel_sort_u8(void* begin, uint64_t len);
    void parallel_sort_i16(void* begin, uint64_t len);
    void parallel_sort_u16(void* begin, uint64_t len);
    void parallel_sort_i32(void* begin, uint64_t len);
    void parallel_sort_u32(void* begin, uint64_t len);
    void parallel_sort_i64(void* begin, uint64_t len);
    void parallel_sort_u64(void* begin, uint64_t len);

    void parallel_sort_f32(void* begin, uint64_t len);
    void parallel_sort_f64(void* begin, uint64_t len);
}

namespace
{
using quant = uint8_t;

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
        _end   = reinterpret_cast<quant*>(begin) + len*size;
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
            _ptr = static_cast<char*>(_ptr) + _size;

            return *this;
        }

        iterator operator ++(int)
        {
            iterator result(*this);
            _ptr = static_cast<char*>(_ptr) + _size;

            return result;
        }

        iterator& operator --()
        {
            _ptr = static_cast<char*>(_ptr) - _size;

            return *this;
        }

        iterator operator --(int)
        {
            iterator result(*this);
            _ptr = static_cast<char*>(_ptr) - _size;

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
            return (reinterpret_cast<quant*>(_ptr) - reinterpret_cast<quant*>(other._ptr))/_size;
        }

        iterator& operator += (difference_type shift)
        {
            _ptr = reinterpret_cast<quant*>(_ptr) + shift*_size;
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

} // namespace

template<typename T>
void parallel_sort_(void* begin, uint64_t len)
{
    std::cout << "parallel_sort_" << " " << len << std::endl;
    auto _begin = reinterpret_cast<T*>(begin);
    auto _end   = _begin + len;

    for(int i = 0; i < len; ++i)
        std::cout << (int64_t)(_begin[i]) << " ";

    std::cout << std::endl;
    tbb::parallel_sort(_begin, _end);
}

#define declare_sort(prefix, ty) \
void parallel_sort_##prefix(void* begin, uint64_t len) { parallel_sort_<ty>(begin, len); }

#define declare_int_sort(bits) \
declare_sort(i##bits, int##bits##_t) \
declare_sort(u##bits, uint##bits##_t)

declare_int_sort(8)
declare_int_sort(16)
declare_int_sort(32)
declare_int_sort(64)

declare_sort(f32, float)
declare_sort(f64, double)

#undef declare_int_sort
#undef declare_sort

void parallel_sort(void* begin, uint64_t len, uint64_t size, void* compare)
{
    auto compare_f = reinterpret_cast<compare_func>(compare);

#define run_sort(range_type) \
{ \
    auto range  = range_type(begin, len, size); \
    auto _begin = range.begin(); \
    auto _end   = range.end(); \
    tbb::parallel_sort(_begin, _end, compare_f); \
}

    switch(size)
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
        // fallback to c qsort?
        if      (size < 4)    run_sort(_void_range<4>)
        else if (size < 8)    run_sort(_void_range<8>)
        else if (size < 16)   run_sort(_void_range<16>)
        else if (size < 32)   run_sort(_void_range<32>)
        else if (size < 64)   run_sort(_void_range<64>)
        else if (size < 128)  run_sort(_void_range<128>)
        else if (size < 256)  run_sort(_void_range<256>)
        else if (size < 512)  run_sort(_void_range<512>)
        else if (size < 1024) run_sort(_void_range<1024>)
        else throw std::runtime_error(std::string("Unsupported item size " + size));
        break;
    }

#undef run_sort
}
