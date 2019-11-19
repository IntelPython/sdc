//*****************************************************************************
// Copyright (c) 2019, Intel Corporation All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//*****************************************************************************

/*
 * Implementation of dictionaries using std::unordered_map.
 *
 * Provides most common maps of simple data types:
 *   {int*, double, float, string} -> {int*, double, float, string}
 * C-Functions are exported as Python module, types are part of their names
 *   dict_<key-type>_<value-type_{init, setitem, getitem, in}.
 * Also provides a dict which maps a byte-array to a int64.
 *
 * We define our own dictionary template class.
 * To get external C-functions per key/value-type we use a macro-factory
 * which generates C-Functions calling our C++ dictionary.
 */

#include <Python.h>
#include <algorithm>
#include <boost/functional/hash/hash.hpp>
#include <boost/preprocessor/list/for_each_product.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <random>
#include <unordered_map>
#include <vector>

// we need a few typedefs to make our macro factory work
// It requires types to end with '_t'
typedef std::string unicode_type_t;
typedef bool bool_t;
typedef int int_t;
typedef float float32_t;
typedef double float64_t;
typedef std::vector<unsigned char> byte_vec_t;

namespace std
{
    // hash function for byte arrays
    template <>
    struct hash<byte_vec_t>
    {
        typedef byte_vec_t argument_type;
        typedef std::size_t result_type;

        // interpret byte-array as array of given integer type (T) and produce hash
        // we use boost::hash_combine for generating an aggregated hash value
        template <typename T>
        result_type hashit(const T* ptr, size_t n) const
        {
            // we do this only for aligned pointers (at both ends!)
            if (n >= sizeof(T) && reinterpret_cast<uintptr_t>(ptr) % sizeof(T) == 0 && n % sizeof(T) == 0)
            {
                auto _ptr = reinterpret_cast<const T*>(ptr);
                n /= sizeof(T);
                std::size_t seed = 0;
                for (size_t i = 0; i < n; ++i)
                {
                    boost::hash_combine(seed, _ptr[i]);
                }
                //std::cout << "[" << sizeof(T) << "] " << seed << std::endl;
                return seed;
            }
            else
            {
                return 0;
            }
        }

        // returns the hash-value for given vector of bytes
        // Note: this will return different hash-values for the same key depending on
        //       the pointer alignment. E.g. if the data of a given vector is aligned
        //       at 8-byte boundary and its size is a multiple of 8 we use a different
        //       hashing algorithm than when the same bytes are unaligned.
        //       We might need to revise this depending on how we use it
        // We might want to produce specializations for specific sizes
        result_type operator()(argument_type const& x) const
        {
            std::size_t n = x.size();
            if (n == 0)
            {
                return 0;
            }
            const argument_type::value_type* ptr = x.data();
            size_t h;
            // we now try to reinterpret bytes as integers of different size, starting with long integers
            if (sizeof(uintmax_t) > sizeof(uint64_t) && (h = hashit(reinterpret_cast<const uintmax_t*>(ptr), n)) != 0)
            {
                return h;
            }
            if ((h = hashit(reinterpret_cast<const uint64_t*>(ptr), n)) != 0)
            {
                return h;
            }
            if ((h = hashit(reinterpret_cast<const uint32_t*>(ptr), n)) != 0)
            {
                return h;
            }
            if ((h = hashit(reinterpret_cast<const uint16_t*>(ptr), n)) != 0)
            {
                return h;
            }
            // This is our fall-back, probably pretty slow
            if ((h = hashit(reinterpret_cast<const uint8_t*>(ptr), n)) != 0)
            {
                return h;
            }

            // everything must align with 1 byte, so we should not ever get here
            std::cerr << "Unexpected code path taken in hash operation";
            return static_cast<result_type>(-1);
        }
    };
} // namespace std

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v)
{
    if (!v.empty())
    {
        out << '[';
        std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
        out << "\b\b]";
    }
    return out;
}

// Type trait to allow by pointer/reference/value distinction for different types.
// Some keys/values are passed by value, others by reference(pointer!)
// This template struct defines how the dict keys/values appear on the dict interface
// Generic struct defines the defaults as by-value
template <typename T>
struct IFTYPE
{
    typedef T in_t;
    typedef T out_t;
    static out_t out(T& o) { return o; }
};

// strings appear by reference/pointer on the interface
template <>
struct IFTYPE<std::string>
{
    typedef std::string& in_t;
    typedef std::string* out_t;
    static out_t out(std::string& o) { return &o; }
};

// byte-vectors appear by reference on the interface
template <>
struct IFTYPE<byte_vec_t>
{
    typedef byte_vec_t& in_t;
    // hopefully we never need the out types
    typedef byte_vec_t out_t;
    static out_t out(byte_vec_t& o) { return o; }
};

// Generic template dict class
template <typename IDX, typename VAL>
class dict
{
private:
    std::unordered_map<IDX, VAL> m_dict;

public:
    typedef typename IFTYPE<IDX>::in_t idx_in_t;
    typedef typename IFTYPE<IDX>::out_t idx_out_t;
    typedef typename IFTYPE<VAL>::in_t val_in_t;
    typedef typename IFTYPE<VAL>::out_t val_out_t;

    dict()
        : m_dict()
    {
    }

    // sets given value for given index
    void setitem(idx_in_t index, val_in_t value)
    {
        m_dict[index] = value;
        return;
    }

    // @return value for given index, entry must exist
    val_out_t getitem(const idx_in_t index) { return IFTYPE<VAL>::out(m_dict.at(index)); }

    // @return true if given index is found in dict, false otherwise
    bool in(const idx_in_t index) { return (m_dict.find(index) != m_dict.end()); }

    // print the entire dict
    void print()
    {
        // TODO: return python string and print in native mode
        for (auto& x : m_dict)
        {
            std::cout << x.first << ": " << x.second << std::endl;
        }
        return;
    }

    // @return value for given index or default_val if not in dict
    val_out_t get(const idx_in_t index, val_in_t default_val)
    {
        auto val = m_dict.find(index);
        if (val == m_dict.end())
        {
            return IFTYPE<VAL>::out(default_val);
        }
        return IFTYPE<VAL>::out((*val).second);
    }

    // deletes entry from dict
    // @return value for given index
    val_out_t pop(const idx_in_t index)
    {
        auto val = IFTYPE<VAL>::out(m_dict.at(index));
        m_dict.erase(index);
        return val;
    }

    void* keys()
    {
        // TODO: return actual iterator
        return this;
    }

    // @return maximum value (not key!) in dict
    val_out_t min()
    {
        // TODO: use actual iterator
        auto res = std::numeric_limits<VAL>::max();
        typename std::unordered_map<IDX, VAL>::iterator it = m_dict.end();
        for (typename std::unordered_map<IDX, VAL>::iterator x = m_dict.begin(); x != m_dict.end(); ++x)
        {
            if (x->second < res)
            {
                res = x->second;
                it = x;
            }
        }
        return IFTYPE<VAL>::out(it->second);
    }

    // @return maximum value (not key!) in dict
    val_out_t max()
    {
        // TODO: use actual iterator
        auto res = std::numeric_limits<VAL>::min();
        typename std::unordered_map<IDX, VAL>::iterator it = m_dict.end();
        for (typename std::unordered_map<IDX, VAL>::iterator x = m_dict.begin(); x != m_dict.end(); ++x)
        {
            if (x->second > res)
            {
                res = x->second;
                it = x;
            }
        }
        return IFTYPE<VAL>::out(it->second);
    }

    // @return true if dict is not empty, false otherwise
    bool not_empty() { return !m_dict.empty(); }
};

// macro expanding to C-functions
#define DEF_DICT(_IDX_, _VAL_)                                                                                         \
    dict<_IDX_##_t, _VAL_##_t>* dict_##_IDX_##_##_VAL_##_init() { return new dict<_IDX_##_t, _VAL_##_t>(); }           \
    void dict_##_IDX_##_##_VAL_##_setitem(                                                                             \
        dict<_IDX_##_t, _VAL_##_t>* m, IFTYPE<_IDX_##_t>::in_t index, IFTYPE<_VAL_##_t>::in_t value)                   \
    {                                                                                                                  \
        m->setitem(index, value);                                                                                      \
    }                                                                                                                  \
    IFTYPE<_VAL_##_t>::out_t dict_##_IDX_##_##_VAL_##_getitem(dict<_IDX_##_t, _VAL_##_t>* m,                           \
                                                              const IFTYPE<_IDX_##_t>::in_t index)                     \
    {                                                                                                                  \
        return m->getitem(index);                                                                                      \
    }                                                                                                                  \
    bool dict_##_IDX_##_##_VAL_##_in(dict<_IDX_##_t, _VAL_##_t>* m, const IFTYPE<_IDX_##_t>::in_t index)               \
    {                                                                                                                  \
        return m->in(index);                                                                                           \
    }                                                                                                                  \
    void dict_##_IDX_##_##_VAL_##_print(dict<_IDX_##_t, _VAL_##_t>* m) { m->print(); }                                 \
    IFTYPE<_VAL_##_t>::out_t dict_##_IDX_##_##_VAL_##_get(                                                             \
        dict<_IDX_##_t, _VAL_##_t>* m, const IFTYPE<_IDX_##_t>::in_t index, IFTYPE<_VAL_##_t>::in_t default_val)       \
    {                                                                                                                  \
        return m->get(index, default_val);                                                                             \
    }                                                                                                                  \
    IFTYPE<_VAL_##_t>::out_t dict_##_IDX_##_##_VAL_##_pop(dict<_IDX_##_t, _VAL_##_t>* m,                               \
                                                          const IFTYPE<_IDX_##_t>::in_t index)                         \
    {                                                                                                                  \
        return m->pop(index);                                                                                          \
    }                                                                                                                  \
    IFTYPE<_VAL_##_t>::out_t dict_##_IDX_##_##_VAL_##_min(dict<_IDX_##_t, _VAL_##_t>* m) { return m->min(); }          \
    IFTYPE<_VAL_##_t>::out_t dict_##_IDX_##_##_VAL_##_max(dict<_IDX_##_t, _VAL_##_t>* m) { return m->max(); }          \
    bool dict_##_IDX_##_##_VAL_##_not_empty(dict<_IDX_##_t, _VAL_##_t>* m) { return m->not_empty(); }                  \
    void* dict_##_IDX_##_##_VAL_##_keys(dict<_IDX_##_t, _VAL_##_t>* m) { return m->keys(); }

/*
 * Byte vectors are special, we need to somehow create them without copying stuff several times.
 * To create such a vector
 *     - first get a handle with byte_vec_init
 *     - use the handle to set it's value with byte_vec_set or byte_vec_append
 * Free memory by calling byte_vec_init when done with it.
 */

// forward decl
void byte_vec_set(byte_vec_t* vec, size_t pos, const unsigned char* val, size_t n);

/**
 * Init byte vector with given size and optional content
 * @param n make vector n bytes long (to be used with byte_vec_set) [default: 0]
 * @param val if != NULL copy n-bytes from here to new vector
 * @return new byte vector of size n
 **/
byte_vec_t* byte_vec_init(size_t n, const unsigned char* val)
{
    auto v = new byte_vec_t(n);
    if (val)
    {
        byte_vec_set(v, 0, val, n);
    }
    return v;
}
// in vec, set n bytes starting at position pos to content of val
// vec must have been initialized with size at least pos+n
void byte_vec_set(byte_vec_t* vec, size_t pos, const unsigned char* val, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        (*vec)[pos + i] = val[i];
    }
}
// resize vector to given length
void byte_vec_resize(byte_vec_t* vec, size_t n)
{
    vec->resize(n);
}
// free vector
void byte_vec_free(byte_vec_t* vec)
{
    delete vec;
}

// multimap for hash join
typedef std::unordered_multimap<int64_t, int64_t> multimap_int64_t;
typedef std::pair<multimap_int64_t::iterator, multimap_int64_t::iterator> multimap_int64_it_t;

multimap_int64_t* multimap_int64_init()
{
    return new multimap_int64_t();
}

void multimap_int64_insert(multimap_int64_t* m, int64_t k, int64_t v)
{
    m->insert(std::make_pair(k, v));
    return;
}

multimap_int64_it_t* multimap_int64_equal_range(multimap_int64_t* m, int64_t k)
{
    return new multimap_int64_it_t(m->equal_range(k));
}

multimap_int64_it_t* multimap_int64_equal_range_alloc()
{
    return new multimap_int64_it_t;
}

void multimap_int64_equal_range_dealloc(multimap_int64_it_t* r)
{
    delete r;
}

void multimap_int64_equal_range_inplace(multimap_int64_t* m, int64_t k, multimap_int64_it_t* r)
{
    *r = m->equal_range(k);
}

// auto range = map.equal_range(1);
// for (auto it = range.first; it != range.second; ++it) {
//     std::cout << it->first << ' ' << it->second << '\n';
// }

bool multimap_int64_it_is_valid(multimap_int64_it_t* r)
{
    return r->first != r->second;
}

int64_t multimap_int64_it_get_value(multimap_int64_it_t* r)
{
    return (r->first)->second;
}

void multimap_int64_it_inc(multimap_int64_it_t* r)
{
    (r->first)++;
    return;
}

// all the types that we support for keys and values
#define TYPES                                                                                                          \
    BOOST_PP_TUPLE_TO_LIST(                                                                                            \
        12, (int, int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool, float32, float64, unicode_type))

// Bring our generic dict to life
DEF_DICT(byte_vec, int64);

// Now use some macro-magic from boost to support dicts for above types
#define APPLY_DEF_DICT(r, product) DEF_DICT product
BOOST_PP_LIST_FOR_EACH_PRODUCT(APPLY_DEF_DICT, 2, (TYPES, TYPES))

// declaration of dict functions in python module
#define DEC_MOD_METHOD(func) PyObject_SetAttrString(m, BOOST_PP_STRINGIZE(func), PyLong_FromVoidPtr((void*)(&func)))
#define DEC_DICT_MOD(_IDX_, _VAL_)                                                                                     \
    DEC_MOD_METHOD(dict_##_IDX_##_##_VAL_##_init);                                                                     \
    DEC_MOD_METHOD(dict_##_IDX_##_##_VAL_##_setitem);                                                                  \
    DEC_MOD_METHOD(dict_##_IDX_##_##_VAL_##_getitem);                                                                  \
    DEC_MOD_METHOD(dict_##_IDX_##_##_VAL_##_in);                                                                       \
    DEC_MOD_METHOD(dict_##_IDX_##_##_VAL_##_print);                                                                    \
    DEC_MOD_METHOD(dict_##_IDX_##_##_VAL_##_get);                                                                      \
    DEC_MOD_METHOD(dict_##_IDX_##_##_VAL_##_pop);                                                                      \
    DEC_MOD_METHOD(dict_##_IDX_##_##_VAL_##_keys);                                                                     \
    DEC_MOD_METHOD(dict_##_IDX_##_##_VAL_##_min);                                                                      \
    DEC_MOD_METHOD(dict_##_IDX_##_##_VAL_##_max);                                                                      \
    DEC_MOD_METHOD(dict_##_IDX_##_##_VAL_##_not_empty);

// module initiliziation
// make our C-functions available
PyMODINIT_FUNC PyInit_hdict_ext(void)
{
    PyObject* m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "hdict_ext",
        "No docs",
        -1,
        NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
    {
        return NULL;
    }

    // Add our generic dict
    DEC_DICT_MOD(byte_vec, int64);
    // And all the other speicialized dicts
#define APPLY_DEC_DICT_MOD(r, product) DEC_DICT_MOD product
    BOOST_PP_LIST_FOR_EACH_PRODUCT(APPLY_DEC_DICT_MOD, 2, (TYPES, TYPES));

    PyObject_SetAttrString(m, "byte_vec_init", PyLong_FromVoidPtr((void*)(&byte_vec_init)));
    PyObject_SetAttrString(m, "byte_vec_set", PyLong_FromVoidPtr((void*)(&byte_vec_set)));
    PyObject_SetAttrString(m, "byte_vec_free", PyLong_FromVoidPtr((void*)(&byte_vec_free)));
    PyObject_SetAttrString(m, "byte_vec_resize", PyLong_FromVoidPtr((void*)(&byte_vec_resize)));
    DEC_MOD_METHOD(multimap_int64_init);
    DEC_MOD_METHOD(multimap_int64_insert);
    DEC_MOD_METHOD(multimap_int64_equal_range);
    DEC_MOD_METHOD(multimap_int64_it_is_valid);
    DEC_MOD_METHOD(multimap_int64_it_get_value);
    DEC_MOD_METHOD(multimap_int64_it_inc);
    DEC_MOD_METHOD(multimap_int64_equal_range_alloc);
    DEC_MOD_METHOD(multimap_int64_equal_range_dealloc);
    DEC_MOD_METHOD(multimap_int64_equal_range_inplace);
    return m;
}
