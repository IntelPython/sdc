// *****************************************************************************
// Copyright (c) 2021, Intel Corporation All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//     Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
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
// *****************************************************************************

#include <cstdint>
#include <utility>
#include <memory>
#include <type_traits>

#ifdef SDC_DEBUG_NATIVE
#include <iostream>
#endif

#include "utils.hpp"
#include "tbb/tbb.h"
#include "tbb/concurrent_unordered_map.h"
#include "numba/core/runtime/nrt_external.h"


using voidptr_hash_type = size_t (*)(void* key_ptr);
using voidptr_eq_type = bool (*)(void* lhs_ptr, void* rhs_ptr);
using voidptr_refcnt = void (*)(void* key_ptr);

using iter_state = std::pair<void*, void*>;


class CustomVoidPtrHasher
{
private:
    voidptr_hash_type ptr_hash_callback;

public:
    CustomVoidPtrHasher(void* ptr_func) {
        ptr_hash_callback = reinterpret_cast<voidptr_hash_type>(ptr_func);
    }

    size_t operator()(void* data_ptr) const {
        auto res = ptr_hash_callback(data_ptr);
        return res;
    }
};


class CustomVoidPtrEquality
{
private:
    voidptr_eq_type ptr_eq_callback;

public:
    CustomVoidPtrEquality(void* ptr_func) {
        ptr_eq_callback = reinterpret_cast<voidptr_eq_type>(ptr_func);
    }

    size_t operator()(void* lhs, void* rhs) const {
        return ptr_eq_callback(lhs, rhs);
    }
};


struct VoidPtrHashCompare {
private:
    voidptr_hash_type ptr_hash_callback;
    voidptr_eq_type ptr_eq_callback;

public:
    size_t hash(void* data_ptr) const {
        return ptr_hash_callback(data_ptr);
    }
    bool equal(void* lhs, void* rhs) const {
        return ptr_eq_callback(lhs, rhs);
    }

    VoidPtrHashCompare(void* ptr_hash, void* ptr_equality) {
        ptr_hash_callback = reinterpret_cast<voidptr_hash_type>(ptr_hash);
        ptr_eq_callback = reinterpret_cast<voidptr_eq_type>(ptr_equality);
    }

    VoidPtrHashCompare() = delete;
    VoidPtrHashCompare(const VoidPtrHashCompare&) = default;
    VoidPtrHashCompare& operator=(const VoidPtrHashCompare&) = default;
    VoidPtrHashCompare(VoidPtrHashCompare&&) = default;
    VoidPtrHashCompare& operator=(VoidPtrHashCompare&&) = default;
    ~VoidPtrHashCompare() = default;
};


struct VoidPtrTypeInfo {
    voidptr_refcnt incref;
    voidptr_refcnt decref;
    uint64_t size;

    VoidPtrTypeInfo(void* incref_addr, void* decref_addr, uint64_t val_size) {
        incref = reinterpret_cast<voidptr_refcnt>(incref_addr);
        decref = reinterpret_cast<voidptr_refcnt>(decref_addr);
        size = val_size;
    }

    VoidPtrTypeInfo() = delete;
    VoidPtrTypeInfo(const VoidPtrTypeInfo&) = default;
    VoidPtrTypeInfo& operator=(const VoidPtrTypeInfo&) = default;
    VoidPtrTypeInfo& operator=(VoidPtrTypeInfo&&) = default;
    ~VoidPtrTypeInfo() = default;

    void delete_voidptr(void* ptr_data) {
        this->decref(ptr_data);
        free(ptr_data);
    }
};


template<typename Key,
         typename Val,
         typename Hasher=std::hash<Key>,
         typename Equality=std::equal_to<Key>
>
class NumericHashmapType {
public:
    using map_type = typename tbb::concurrent_unordered_map<Key, Val, Hasher, Equality>;
    using iterator_type = typename map_type::iterator;
    map_type map;

    NumericHashmapType()
    : map(0, Hasher(), Equality()) {}
    // TO-DO: support copying for all hashmaps and ConcurrentDict's .copy() method in python?
    NumericHashmapType(const NumericHashmapType&) = delete;
    NumericHashmapType& operator=(const NumericHashmapType&) = delete;
    NumericHashmapType(NumericHashmapType&& rhs) = delete;
    NumericHashmapType& operator=(NumericHashmapType&& rhs) = delete;
    ~NumericHashmapType() {}

    uint64_t size() {
        return this->map.size();
    }

    void set(Key key, Val val) {
        this->map[key] = val;
    }

    int8_t contains(Key key) {
        auto it = this->map.find(key);
        return it != this->map.end();
    }
    int8_t lookup(Key key, Val* res) {
        auto it = this->map.find(key);
        bool found = it != this->map.end();
        if (found)
            *res = (*it).second;

        return found;
    }
    void clear() {
        this->map.clear();
    }

    int8_t pop(Key key, Val* res) {
        auto node_handle = this->map.unsafe_extract(key);
        auto found = !node_handle.empty();
        if (found)
            *res = node_handle.mapped();

        return found;
    }

    void update(NumericHashmapType& other) {
        this->map.merge(other.map);
    }

    void* getiter() {
        auto p_it = new iterator_type(this->map.begin());
        auto state = new iter_state((void*)p_it, (void*)this);
        return state;
    }
};


template<typename Key,
         typename Val,
         typename HashCompare=tbb::tbb_hash_compare<Key>
>
class GenericHashmapBase {
public:
    using map_type = typename tbb::concurrent_hash_map<Key, Val, HashCompare>;
    using iterator_type = typename map_type::iterator;
    map_type map;

    // FIXME: 0 default size is suboptimal, can we optimize this?
    GenericHashmapBase() : map(0, HashCompare()) {}
    GenericHashmapBase(const HashCompare& hash_compare) : map(0, hash_compare) {}

    GenericHashmapBase(const GenericHashmapBase&) = delete;
    GenericHashmapBase& operator=(const GenericHashmapBase&) = delete;
    GenericHashmapBase(GenericHashmapBase&& rhs) = delete;
    GenericHashmapBase& operator=(GenericHashmapBase&& rhs) = delete;
    virtual ~GenericHashmapBase() {
    }

    uint64_t size() {
        return this->map.size();
    }

    int8_t contains(Key key) {
        bool found = false;
        {
            typename map_type::const_accessor result;
            found = this->map.find(result, key);
            result.release();
        }
        return found;
    }

    int8_t lookup(Key key, Val* res) {
        bool found = false;
        {
            typename map_type::const_accessor result;
            found = this->map.find(result, key);
            if (found)
                *res = result->second;
            result.release();
        }

        return found;
    }

    virtual void set(Key key, Val val) = 0;

    void update(GenericHashmapBase& other) {
        tbb::parallel_for(
                other.map.range(),
                [this](const typename map_type::range_type& r) {
                    for (typename map_type::iterator i = r.begin(); i != r.end(); ++i) {
                        this->set(i->first, i->second);
                    }
        });
    }

    void* getiter() {
        auto p_it = new iterator_type(this->map.begin());
        auto state = new iter_state((void*)p_it, (void*)this);
        return state;
    }
};


/* primary template for GenericHashmapType */
template<typename Key,
         typename Val,
         typename HashCompare=tbb::tbb_hash_compare<Key>
>
class GenericHashmapType : public GenericHashmapBase<Key, Val, HashCompare> {
public:
    // TO-DO: make VoidPtrTypeInfo templates and unify modifiers impl via calls to template funcs
    using map_type = typename GenericHashmapBase<Key, Val, HashCompare>::map_type;
    VoidPtrTypeInfo key_info;
    VoidPtrTypeInfo val_info;

    GenericHashmapType(const VoidPtrTypeInfo& ki,
                       const VoidPtrTypeInfo& vi,
                       const HashCompare& hash_compare)
    : GenericHashmapBase<Key, Val, HashCompare>(hash_compare),
      key_info(ki),
      val_info(vi) {}
    GenericHashmapType(const VoidPtrTypeInfo& ki, const VoidPtrTypeInfo& vi) : GenericHashmapType(ki, vi, HashCompare()) {}

    GenericHashmapType() = delete;
    GenericHashmapType(const GenericHashmapType&) = delete;
    GenericHashmapType& operator=(const GenericHashmapType&) = delete;
    GenericHashmapType(GenericHashmapType&& rhs) = delete;
    GenericHashmapType& operator=(GenericHashmapType&& rhs) = delete;
    virtual ~GenericHashmapType() {};


    void clear() {
        this->map.clear();
    }

    int8_t pop(Key key, void* res) {
        bool found = false;
        {
            typename map_type::const_accessor result;
            found = this->map.find(result, key);
            if (found)
            {
                memcpy(res, &(result->second), this->val_info.size);
                this->map.erase(result);
            }
            result.release();
        }

        return found;
    }

    void set(Key key, Val val) {
        typename map_type::value_type inserted_node(key, val);
        {
            typename map_type::accessor existing_node;
            bool ok = this->map.insert(existing_node, inserted_node);
            if (!ok)
            {
                // insertion failed key already exists
                existing_node->second = val;
            }
        }
    }
};


/* generic-value partial specialization */
template<typename Key, typename HashCompare>
class GenericHashmapType<Key, void*, HashCompare> : public GenericHashmapBase<Key, void*, HashCompare> {
public:
    using map_type = typename GenericHashmapBase<Key, void*, HashCompare>::map_type;
    VoidPtrTypeInfo key_info;
    VoidPtrTypeInfo val_info;

    GenericHashmapType(const VoidPtrTypeInfo& ki,
                       const VoidPtrTypeInfo& vi,
                       const HashCompare& hash_compare)
    : GenericHashmapBase<Key, void*, HashCompare>(hash_compare),
      key_info(ki),
      val_info(vi) {}
    GenericHashmapType(const VoidPtrTypeInfo& ki, const VoidPtrTypeInfo& vi) : GenericHashmapType(ki, vi, HashCompare()) {}

    GenericHashmapType() = delete;
    GenericHashmapType(const GenericHashmapType&) = delete;
    GenericHashmapType& operator=(const GenericHashmapType&) = delete;
    GenericHashmapType(GenericHashmapType&& rhs) = delete;
    GenericHashmapType& operator=(GenericHashmapType&& rhs) = delete;
    virtual ~GenericHashmapType() {};

    void clear();
    virtual void set(Key key, void* val) override;
    int8_t pop(Key key, void* val);
};


/* generic-key partial specialization */
template<typename Val>
class GenericHashmapType<void*, Val, VoidPtrHashCompare> : public GenericHashmapBase<void*, Val, VoidPtrHashCompare> {
public:
    using map_type = typename GenericHashmapBase<void*, Val, VoidPtrHashCompare>::map_type;
    VoidPtrTypeInfo key_info;
    VoidPtrTypeInfo val_info;

    GenericHashmapType(const VoidPtrTypeInfo& ki,
                       const VoidPtrTypeInfo& vi,
                       const VoidPtrHashCompare& hash_compare)
    : GenericHashmapBase<void*, Val, VoidPtrHashCompare>(hash_compare),
      key_info(ki),
      val_info(vi) {}

    GenericHashmapType() = delete;
    GenericHashmapType(const GenericHashmapType&) = delete;
    GenericHashmapType& operator=(const GenericHashmapType&) = delete;
    GenericHashmapType(GenericHashmapType&& rhs) = delete;
    GenericHashmapType& operator=(GenericHashmapType&& rhs) = delete;
    virtual ~GenericHashmapType() {};

    void clear();
    virtual void set(void* key, Val val) override;
    int8_t pop(void* key, void* val);
};


/* generic-key-and-value partial specialization */
template<>
class GenericHashmapType<void*, void*, VoidPtrHashCompare> : public GenericHashmapBase<void*, void*, VoidPtrHashCompare> {
public:
    using map_type = typename GenericHashmapType<void*, void*, VoidPtrHashCompare>::map_type;
    VoidPtrTypeInfo key_info;
    VoidPtrTypeInfo val_info;

    GenericHashmapType(const VoidPtrTypeInfo& ki,
                       const VoidPtrTypeInfo& vi,
                       const VoidPtrHashCompare& hash_compare)
    : GenericHashmapBase<void*, void*, VoidPtrHashCompare>(hash_compare),
      key_info(ki),
      val_info(vi) {}

    GenericHashmapType() = delete;
    GenericHashmapType(const GenericHashmapType&) = delete;
    GenericHashmapType& operator=(const GenericHashmapType&) = delete;
    GenericHashmapType(GenericHashmapType&& rhs) = delete;
    GenericHashmapType& operator=(GenericHashmapType&& rhs) = delete;
    virtual ~GenericHashmapType() {};

    void clear();
    virtual void set(void* key, void* val) override;
    int8_t pop(void* key, void* val);
};


template <typename Key, typename Val>
using numeric_hashmap = NumericHashmapType<Key, Val>;

template <typename Val>
using generic_key_hashmap = GenericHashmapType<void*, Val, VoidPtrHashCompare>;

template <typename Key, typename HashCompare=tbb::tbb_hash_compare<Key>>
using generic_value_hashmap = GenericHashmapType<Key, void*, HashCompare>;

using generic_hashmap = GenericHashmapType<void*, void*, VoidPtrHashCompare>;


template<typename key_type, typename val_type>
numeric_hashmap<key_type, val_type>*
reinterpet_hashmap_ptr(void* p_hash_map,
                       typename std::enable_if<
                           !std::is_same<key_type, void*>::value &&
                           !std::is_same<val_type, void*>::value>::type* = 0)
{
    return reinterpret_cast<numeric_hashmap<key_type, val_type>*>(p_hash_map);
}

template<typename key_type, typename val_type>
generic_hashmap*
reinterpet_hashmap_ptr(void* p_hash_map,
                       typename std::enable_if<
                           std::is_same<key_type, void*>::value &&
                           std::is_same<val_type, void*>::value>::type* = 0)
{
    return reinterpret_cast<generic_hashmap*>(p_hash_map);
}

template<typename key_type, typename val_type>
generic_value_hashmap<key_type>*
reinterpet_hashmap_ptr(void* p_hash_map,
                       typename std::enable_if<
                           !std::is_same<key_type, void*>::value &&
                           std::is_same<val_type, void*>::value>::type* = 0)
{
    return reinterpret_cast<generic_value_hashmap<key_type>*>(p_hash_map);
}

template<typename key_type, typename val_type>
generic_key_hashmap<val_type>*
reinterpet_hashmap_ptr(void* p_hash_map,
                       typename std::enable_if<
                           std::is_same<key_type, void*>::value &&
                           !std::is_same<val_type, void*>::value>::type* = 0)
{
    return reinterpret_cast<generic_key_hashmap<val_type>*>(p_hash_map);
}


template <typename value_type>
void delete_generic_key_hashmap(void* p_hash_map)
{
    auto p_hash_map_spec = (generic_key_hashmap<value_type>*)p_hash_map;
    for (auto kv_pair: p_hash_map_spec->map) {
        p_hash_map_spec->key_info.delete_voidptr(kv_pair.first);
    }
    delete p_hash_map_spec;
}

template <typename key_type>
void delete_generic_value_hashmap(void* p_hash_map)
{

    auto p_hash_map_spec = (generic_value_hashmap<key_type>*)p_hash_map;
    for (auto kv_pair: p_hash_map_spec->map) {
        p_hash_map_spec->val_info.delete_voidptr(kv_pair.second);
    }
    delete p_hash_map_spec;
}

void delete_generic_hashmap(void* p_hash_map)
{
    auto p_hash_map_spec = (generic_hashmap*)p_hash_map;
    for (auto kv_pair: p_hash_map_spec->map) {
        p_hash_map_spec->key_info.delete_voidptr(kv_pair.first);
        p_hash_map_spec->val_info.delete_voidptr(kv_pair.second);
    }
    delete p_hash_map_spec;
}

template <typename key_type, typename value_type>
void delete_numeric_hashmap(void* p_hash_map)
{

    auto p_hash_map_spec = (numeric_hashmap<key_type, value_type>*)p_hash_map;
    delete p_hash_map_spec;
}


template <typename key_type, typename value_type>
void delete_iter_state(void* p_iter_state)
{
    auto p_iter_state_spec = reinterpret_cast<iter_state*>(p_iter_state);
    auto p_hash_map_spec = reinterpet_hashmap_ptr<key_type, value_type>(p_iter_state_spec->second);
    using itertype = typename std::remove_reference<decltype(*p_hash_map_spec)>::type::iterator_type;
    auto p_hash_map_iter = reinterpret_cast<itertype*>(p_iter_state_spec->first);

    delete p_hash_map_iter;
    delete p_iter_state_spec;
}


template<typename Key,
         typename HashCompare
>
void GenericHashmapType<Key, void*, HashCompare>::set(Key key, void* val)
{
    auto vsize = this->val_info.size;
    void* _val = malloc(vsize);
    memcpy(_val, val, vsize);

    typename map_type::value_type inserted_node(key, _val);
    {
        typename map_type::accessor existing_node;
        bool ok = this->map.insert(existing_node, inserted_node);
        if (ok)
        {
            // insertion succeeded need to incref value
            this->val_info.incref(val);
        }
        else
        {
            // insertion failed key already exists
            this->val_info.delete_voidptr(existing_node->second);
            existing_node->second = _val;
            this->val_info.incref(val);
        }
    }
}

template<typename Key,
         typename HashCompare
>
void GenericHashmapType<Key, void*, HashCompare>::clear()
{
    for (auto kv_pair: this->map) {
        this->val_info.delete_voidptr(kv_pair.second);
    }
    this->map.clear();
}


template<typename Key,
         typename HashCompare
>
int8_t GenericHashmapType<Key, void*, HashCompare>::pop(Key key, void* res) {
    bool found = false;
    {
        typename map_type::const_accessor result;
        found = this->map.find(result, key);
        if (found)
        {
            memcpy(res, result->second, this->val_info.size);
            free(result->second);
            // no decref for value since it would be returned (and no incref on python side!)
            this->map.erase(result);
        }
        result.release();
    }

    return found;
}


template<typename Val>
void GenericHashmapType<void*, Val, VoidPtrHashCompare>::set(void* key, Val val)
{
    auto ksize = this->key_info.size;
    void* _key = malloc(ksize);
    memcpy(_key, key, ksize);

    typename map_type::value_type inserted_node(_key, val);
    {
        typename map_type::accessor existing_node;
        bool ok = this->map.insert(existing_node, inserted_node);
        if (ok)
        {
            // insertion succeeded need to incref key
            this->key_info.incref(key);
        }
        else
        {
            // insertion failed key already exists
            free(_key);
            existing_node->second = val;
        }
    }
}

template<typename Val>
void GenericHashmapType<void*, Val, VoidPtrHashCompare>::clear()
{
    for (auto kv_pair: this->map) {
        this->key_info.delete_voidptr(kv_pair.first);
    }
    this->map.clear();
}

template<typename Val>
int8_t GenericHashmapType<void*, Val, VoidPtrHashCompare>::pop(void* key, void* res) {
    bool found = false;
    {
        typename map_type::const_accessor result;
        found = this->map.find(result, key);
        if (found)
        {
            memcpy(res, &(result->second), this->val_info.size);
            this->key_info.delete_voidptr(result->first);
            // no decref for value since it would be returned (and no incref on python side!)
            this->map.erase(result);
        }
        result.release();
    }

    return found;
}


void GenericHashmapType<void*, void*, VoidPtrHashCompare>::set(void* key, void* val)

{
    auto ksize = this->key_info.size;
    void* _key = malloc(ksize);
    memcpy(_key, key, ksize);

    auto vsize = this->val_info.size;
    void* _val = malloc(vsize);
    memcpy(_val, val, vsize);

    typename map_type::value_type inserted_node(_key, _val);
    {
        typename map_type::accessor existing_node;
        bool ok = this->map.insert(existing_node, inserted_node);
        if (ok)
        {
            this->key_info.incref(key);
            this->val_info.incref(val);
        }
        else
        {
            // insertion failed key already exists
            free(_key);

            this->val_info.delete_voidptr(existing_node->second);
            existing_node->second = _val;
            this->val_info.incref(val);
        }
    }
}

void GenericHashmapType<void*, void*, VoidPtrHashCompare>::clear()
{
    for (auto kv_pair: this->map) {
        this->key_info.delete_voidptr(kv_pair.first);
        this->val_info.delete_voidptr(kv_pair.second);
    }
    this->map.clear();
}


int8_t GenericHashmapType<void*, void*, VoidPtrHashCompare>::pop(void* key, void* res) {
    bool found = false;
    {
        typename map_type::const_accessor result;
        found = this->map.find(result, key);
        if (found)
        {
            memcpy(res, result->second, this->val_info.size);

            free(result->second);
            this->key_info.delete_voidptr(result->first);
            // no decref for value since it would be returned (and no incref on python side!)
            this->map.erase(result);
        }
        result.release();
    }

    return found;
}


template<typename key_type, typename val_type>
void hashmap_create(NRT_MemInfo** meminfo,
                    void* nrt_table,
                    int8_t gen_key,
                    int8_t gen_val,
                    void* hash_func_ptr,
                    void* eq_func_ptr,
                    void* key_incref_func_ptr,
                    void* key_decref_func_ptr,
                    void* val_incref_func_ptr,
                    void* val_decref_func_ptr,
                    uint64_t key_size,
                    uint64_t val_size)
{
    auto nrt = (NRT_api_functions*)nrt_table;

    // it is essential for all specializations to have common ctor signature, taking both key_info and val_info
    // since all specializations should be instantiable with different key_type/value_type, so e.g.
    // generic_key_hashmap<val_type> with val_type = void* would match full specialization. TO-DO: consider refactoring
    auto key_info = VoidPtrTypeInfo(key_incref_func_ptr, key_decref_func_ptr, key_size);
    auto val_info = VoidPtrTypeInfo(val_incref_func_ptr, val_decref_func_ptr, val_size);
    if (gen_key && gen_val)
    {
        auto p_hash_map = new generic_hashmap(key_info, val_info, VoidPtrHashCompare(hash_func_ptr, eq_func_ptr));
        (*meminfo) = nrt->manage_memory((void*)p_hash_map, delete_generic_hashmap);
    }
    else if (gen_key)
    {
        auto p_hash_map = new generic_key_hashmap<val_type>(key_info, val_info, VoidPtrHashCompare(hash_func_ptr, eq_func_ptr));
        (*meminfo) = nrt->manage_memory((void*)p_hash_map, delete_generic_key_hashmap<val_type>);
    }
    else if (gen_val)
    {
        auto p_hash_map = new generic_value_hashmap<key_type>(key_info, val_info);
        (*meminfo) = nrt->manage_memory((void*)p_hash_map, delete_generic_value_hashmap<key_type>);
    }
    else
    {
        // numeric_hashmap is actually an instance of NumericHashmapType, not a specialization of
        // GenericHashmapType since it's built upon tbb::concurrent_unordered_map. TO-DO: consider
        // moving to one impl later if there's no performance penalty
        auto p_hash_map = new numeric_hashmap<key_type, val_type>;
        (*meminfo) = nrt->manage_memory((void*)p_hash_map, delete_numeric_hashmap<key_type, val_type>);
    }

    return;
}


template<typename key_type, typename val_type>
uint64_t hashmap_size(void* p_hash_map)
{
    auto p_hash_map_spec = reinterpet_hashmap_ptr<key_type, val_type>(p_hash_map);
    return p_hash_map_spec->size();
}


template<typename key_type, typename val_type>
void hashmap_set(void* p_hash_map, key_type key, val_type val)
{
    auto p_hash_map_spec = reinterpet_hashmap_ptr<key_type, val_type>(p_hash_map);
    p_hash_map_spec->set(key, val);
}


template<typename key_type, typename val_type>
int8_t hashmap_lookup(void* p_hash_map,
                      key_type key,
                      val_type* res)
{
    auto p_hash_map_spec = reinterpet_hashmap_ptr<key_type, val_type>(p_hash_map);
    return p_hash_map_spec->lookup(key, res);
}


template<typename key_type, typename val_type>
void hashmap_clear(void* p_hash_map)
{
    auto p_hash_map_spec = reinterpet_hashmap_ptr<key_type, val_type>(p_hash_map);
    p_hash_map_spec->clear();
}


template<typename key_type, typename val_type>
int8_t hashmap_unsafe_extract(void* p_hash_map, key_type key, val_type* res)
{
    auto p_hash_map_spec = reinterpet_hashmap_ptr<key_type, val_type>(p_hash_map);
    return p_hash_map_spec->pop(key, res);
}


template<typename key_type, typename val_type>
void hashmap_numeric_from_arrays(NRT_MemInfo** meminfo, void* nrt_table, key_type* keys, val_type* values, uint64_t size)
{
    auto nrt = (NRT_api_functions*)nrt_table;
    auto p_hash_map = new numeric_hashmap<key_type, val_type>;
    (*meminfo) = nrt->manage_memory((void*)p_hash_map, delete_numeric_hashmap<key_type, val_type>);

    // FIXME: apply arena to make this react on changing NUMBA_NUM_THREADS
    tbb::parallel_for(tbb::blocked_range<size_t>(0, size),
                 [=](const tbb::blocked_range<size_t>& r) {
                     for(size_t i=r.begin(); i!=r.end(); ++i) {
                         auto kv_pair = std::pair<const key_type, val_type>(keys[i], values[i]);
                         p_hash_map->map.insert(
                             std::move(kv_pair)
                         );
                     }
                 }
    );
}


template<typename key_type, typename val_type>
void hashmap_update(void* p_self_hash_map, void* p_arg_hash_map)
{
    auto p_self_hash_map_spec = reinterpet_hashmap_ptr<key_type, val_type>(p_self_hash_map);
    auto p_arg_hash_map_spec = reinterpet_hashmap_ptr<key_type, val_type>(p_arg_hash_map);
    p_self_hash_map_spec->update(*p_arg_hash_map_spec);
    return;
}


#ifdef SDC_DEBUG_NATIVE
template<typename key_type, typename val_type>
void hashmap_dump(void* p_hash_map)
{
    auto p_hash_map_spec = reinterpet_hashmap_ptr<key_type, val_type>(p_hash_map);
    auto size = p_hash_map_spec->map.size();
    std::cout << "Hashmap at: " << p_hash_map_spec << ", size = " << size << std::endl;
    for (auto kv_pair: p_hash_map_spec->map)
    {
        std::cout << "key, value: " << kv_pair.first << ", " << kv_pair.second << std::endl;
    }
    return;
}
#endif


template<typename key_type, typename val_type>
void* hashmap_getiter(NRT_MemInfo** meminfo, void* nrt_table, void* p_hash_map)
{
    auto p_hash_map_spec = reinterpet_hashmap_ptr<key_type, val_type>(p_hash_map);
    auto p_iter_state = p_hash_map_spec->getiter();

    auto nrt = (NRT_api_functions*)nrt_table;
    (*meminfo) = nrt->manage_memory((void*)p_iter_state, delete_iter_state<key_type, val_type>);
    return p_iter_state;
}


template<typename key_type, typename val_type>
int8_t hashmap_iternext(void* p_iter_state, key_type* ret_key, val_type* ret_val)
{
    auto p_iter_state_spec = reinterpret_cast<iter_state*>(p_iter_state);
    auto p_hash_map_spec = reinterpet_hashmap_ptr<key_type, val_type>(p_iter_state_spec->second);
    using itertype = typename std::remove_reference<decltype(*p_hash_map_spec)>::type::iterator_type;
    auto p_hash_map_iter = reinterpret_cast<itertype*>(p_iter_state_spec->first);

    int8_t status = 1;
    if (*p_hash_map_iter != p_hash_map_spec->map.end())
    {
        *ret_key = (*p_hash_map_iter)->first;
        *ret_val = (*p_hash_map_iter)->second;
        status = 0;
        ++(*p_hash_map_iter);
    }

    return status;
}
