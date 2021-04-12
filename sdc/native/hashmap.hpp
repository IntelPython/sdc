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
#include <iostream>
#include <utility>
#include <memory>
#include <type_traits>

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
    VoidPtrHashCompare(const VoidPtrHashCompare& rhs) = default;
    VoidPtrHashCompare& operator=(const VoidPtrHashCompare&) = default;
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
    std::unique_ptr<map_type> p_map;

    NumericHashmapType()
    : p_map(std::unique_ptr<map_type>(new map_type(0, Hasher(), Equality()))) {}
    // TO-DO: support copying for all hashmaps and ConcurrentDict's .copy() method in python?
    NumericHashmapType(const NumericHashmapType&) = delete;
    NumericHashmapType& operator=(const NumericHashmapType&) = delete;
    NumericHashmapType(NumericHashmapType&& rhs) : p_map(std::move(rhs->p_map)) {};
    NumericHashmapType& operator=(NumericHashmapType&& rhs) {
        this->p_map.reset(std::move(rhs->p_map));
    };
    ~NumericHashmapType() {}

    uint64_t size() {
        return this->p_map->size();
    }

    void set(Key key, Val val) {
        (*this->p_map)[key] = val;
    }

    int8_t contains(Key key) {
        auto it = this->p_map->find(key);
        return it != this->p_map->end();
    }
    int8_t lookup(Key key, Val* res) {
        auto it = this->p_map->find(key);
        bool found = it != this->p_map->end();
        if (found)
            *res = (*it).second;

        return found;
    }
    void clear() {
        this->p_map->clear();
    }

    int8_t pop(Key key, Val* res) {
        auto node_handle = this->p_map->unsafe_extract(key);
        auto found = !node_handle.empty();
        if (found)
            *res = node_handle.mapped();

        return found;
    }

    void update(const NumericHashmapType& other) {
        this->p_map->merge(*other.p_map);
    }

    void* getiter() {
        auto p_it = new iterator_type(this->p_map->begin());
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
    std::unique_ptr<map_type> p_map;

    // FIXME: 0 default size is suboptimal, can we optimize this?
    GenericHashmapBase() : p_map(std::unique_ptr<map_type>(new map_type(0, HashCompare()))) {}
    GenericHashmapBase(const HashCompare& hash_compare) {
        this->p_map.reset(new map_type(0, hash_compare));
    }
    GenericHashmapBase(std::unique_ptr<map_type>&& rhs_p_map) : p_map(std::move(rhs_p_map)) {};

    GenericHashmapBase(const GenericHashmapBase&) = delete;
    GenericHashmapBase& operator=(const GenericHashmapBase&) = delete;
    GenericHashmapBase(GenericHashmapBase&& rhs) : p_map(std::move(rhs->p_map)) {};
    GenericHashmapBase& operator=(GenericHashmapBase&& rhs) {
        this->p_map.reset(std::move(rhs->p_map));
    };
    virtual ~GenericHashmapBase() {
    }

    uint64_t size() {
        return this->p_map->size();
    }

    int8_t contains(Key key) {
        bool found = false;
        {
            typename map_type::const_accessor result;
            found = this->p_map->find(result, key);
            result.release();
        }
        return found;
    }

    int8_t lookup(Key key, Val* res) {
        bool found = false;
        {
            typename map_type::const_accessor result;
            found = this->p_map->find(result, key);
            if (found)
                *res = result->second;
            result.release();
        }

        return found;
    }

    virtual void set(Key key, Val val) = 0;

    void update(const GenericHashmapBase& other) {
        tbb::parallel_for(
                other.p_map->range(),
                [this](const typename map_type::range_type& r) {
                    for (typename map_type::iterator i = r.begin(); i != r.end(); ++i) {
                        this->set(i->first, i->second);
                    }
        });
    }

    void* getiter() {
        auto p_it = new iterator_type(this->p_map->begin());
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
    using map_type = typename GenericHashmapBase<Key, Val, HashCompare>::map_type;
    VoidPtrTypeInfo key_info;
    VoidPtrTypeInfo val_info;

    GenericHashmapType(const VoidPtrTypeInfo& ki,
                       const VoidPtrTypeInfo& vi,
                       const HashCompare& hash_compare)
    : key_info(ki),
      val_info(vi),
      GenericHashmapBase<Key, Val, HashCompare>(hash_compare) {}
    GenericHashmapType(const VoidPtrTypeInfo& ki, const VoidPtrTypeInfo& vi) : GenericHashmapType(ki, vi, HashCompare()) {}

    GenericHashmapType() = delete;
    GenericHashmapType(const GenericHashmapType&) = delete;
    GenericHashmapType& operator=(const GenericHashmapType&) = delete;
    GenericHashmapType(GenericHashmapType&& rhs)
    : GenericHashmapBase<Key, Val, HashCompare>(std::move(rhs.p_map)),
      key_info(std::move(rhs.key_info)),
      val_info(std::move(rhs.val_info)) {};
    GenericHashmapType& operator=(GenericHashmapType&& rhs) {
        this->p_map = std::move(rhs.p_map);
        this->key_info = std::move(rhs.key_info);
        this->val_info = std::move(rhs.val_info);
    };
    virtual ~GenericHashmapType() {};


    void clear() {
        this->p_map->clear();
    }

    int8_t pop(Key key, void* res) {
        bool found = false;
        {
            typename map_type::const_accessor result;
            found = this->p_map->find(result, key);
            if (found)
            {
                memcpy(res, &(result->second), this->val_info.size);
                this->p_map->erase(result);
            }
            result.release();
        }

        return found;
    }

    void set(Key key, Val val) {
        typename map_type::value_type inserted_node(key, val);
        {
            typename map_type::accessor existing_node;
            bool ok = this->p_map->insert(existing_node, inserted_node);
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
    : key_info(ki),
      val_info(vi),
      GenericHashmapBase<Key, void*, HashCompare>(hash_compare) {}
    GenericHashmapType(const VoidPtrTypeInfo& ki, const VoidPtrTypeInfo& vi) : GenericHashmapType(ki, vi, HashCompare()) {}

    GenericHashmapType() = delete;
    GenericHashmapType(const GenericHashmapType&) = delete;
    GenericHashmapType& operator=(const GenericHashmapType&) = delete;
    GenericHashmapType(GenericHashmapType&& rhs)
    : GenericHashmapBase<Key, void*, HashCompare>(std::move(rhs.p_map)),
      key_info(std::move(rhs.key_info)),
      val_info(std::move(rhs.val_info)) {};
    GenericHashmapType& operator=(GenericHashmapType&& rhs) {
        this->p_map = std::move(rhs.p_map);
        this->key_info = std::move(rhs.key_info);
        this->val_info = std::move(rhs.val_info);
    };
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
    : key_info(ki),
      val_info(vi),
      GenericHashmapBase<void*, Val, VoidPtrHashCompare>(hash_compare) {}

    GenericHashmapType() = delete;
    GenericHashmapType(const GenericHashmapType&) = delete;
    GenericHashmapType& operator=(const GenericHashmapType&) = delete;
    GenericHashmapType(GenericHashmapType&& rhs)
    : GenericHashmapBase<void*, Val, VoidPtrHashCompare>(std::move(rhs.p_map)),
      key_info(std::move(rhs.key_info)),
      val_info(std::move(rhs.val_info)) {};
    GenericHashmapType& operator=(GenericHashmapType&& rhs) {
        this->p_map = std::move(rhs.p_map);
        this->key_info = std::move(rhs.key_info);
        this->val_info = std::move(rhs.val_info);
    };
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
    : key_info(ki),
      val_info(vi),
      GenericHashmapBase<void*, void*, VoidPtrHashCompare>(hash_compare) {}

    GenericHashmapType() = delete;
    GenericHashmapType(const GenericHashmapType&) = delete;
    GenericHashmapType& operator=(const GenericHashmapType&) = delete;
    GenericHashmapType(GenericHashmapType&& rhs)
    : GenericHashmapBase<void*, void*, VoidPtrHashCompare>(std::move(rhs.p_map)),
      key_info(std::move(rhs.key_info)),
      val_info(std::move(rhs.val_info)) {};
    GenericHashmapType& operator=(GenericHashmapType&& rhs) {
        this->p_map = std::move(rhs.p_map);
        this->key_info = std::move(rhs.key_info);
        this->val_info = std::move(rhs.val_info);
    };
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
auto reinterpet_hashmap_ptr(void* p_hash_map,
                            std::enable_if_t<
                                !std::is_same<key_type, void*>::value &&
                                !std::is_same<val_type, void*>::value>* = 0)
{
    return reinterpret_cast<numeric_hashmap<key_type, val_type>*>(p_hash_map);
}

template<typename key_type, typename val_type>
auto reinterpet_hashmap_ptr(void* p_hash_map,
                            std::enable_if_t<
                                std::is_same<key_type, void*>::value &&
                                std::is_same<val_type, void*>::value>* = 0)
{
    return reinterpret_cast<generic_hashmap*>(p_hash_map);
}

template<typename key_type, typename val_type>
auto reinterpet_hashmap_ptr(void* p_hash_map,
                            std::enable_if_t<
                                !std::is_same<key_type, void*>::value &&
                                std::is_same<val_type, void*>::value>* = 0)
{
    return reinterpret_cast<generic_value_hashmap<key_type>*>(p_hash_map);
}

template<typename key_type, typename val_type>
auto reinterpet_hashmap_ptr(void* p_hash_map,
                            std::enable_if_t<
                                std::is_same<key_type, void*>::value &&
                                !std::is_same<val_type, void*>::value>* = 0)
{
    return reinterpret_cast<generic_key_hashmap<val_type>*>(p_hash_map);
}


template <typename value_type>
void delete_generic_key_hashmap(void* p_hash_map)
{
    auto p_hash_map_spec = (generic_key_hashmap<value_type>*)p_hash_map;
    for (auto kv_pair: (*p_hash_map_spec->p_map)) {
        p_hash_map_spec->key_info.decref(kv_pair.first);
        free(kv_pair.first);
    }
    delete p_hash_map_spec;
}

template <typename key_type>
void delete_generic_value_hashmap(void* p_hash_map)
{

    auto p_hash_map_spec = (generic_value_hashmap<key_type>*)p_hash_map;
    for (auto kv_pair: (*p_hash_map_spec->p_map)) {
        p_hash_map_spec->val_info.decref(kv_pair.second);
        free(kv_pair.second);
    }
    delete p_hash_map_spec;
}

void delete_generic_hashmap(void* p_hash_map)
{
    auto p_hash_map_spec = (generic_hashmap*)p_hash_map;
    for (auto kv_pair: (*p_hash_map_spec->p_map)) {
        p_hash_map_spec->key_info.decref(kv_pair.first);
        free(kv_pair.first);
        p_hash_map_spec->val_info.decref(kv_pair.second);
        free(kv_pair.second);
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
    using itertype = typename std::remove_reference_t<decltype(*p_hash_map_spec)>::iterator_type;
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
        bool ok = this->p_map->insert(existing_node, inserted_node);
        if (ok)
        {
            // isnertion succeeded need to incref value
            this->val_info.incref(val);
        }
        else
        {
            // insertion failed key already exists
            this->val_info.decref(existing_node->second);
            free(existing_node->second);
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
    for (auto kv_pair: (*this->p_map)) {
        this->val_info.decref(kv_pair.second);
        free(kv_pair.second);
    }
    this->p_map->clear();
}


template<typename Key,
         typename HashCompare
>
int8_t GenericHashmapType<Key, void*, HashCompare>::pop(Key key, void* res) {
    bool found = false;
    {
        typename map_type::const_accessor result;
        found = this->p_map->find(result, key);
        if (found)
        {
            memcpy(res, result->second, this->val_info.size);
            free(result->second);
            // no decref for value since it would be returned (and no incref on python side!)
            this->p_map->erase(result);
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
        bool ok = this->p_map->insert(existing_node, inserted_node);
        if (ok)
        {
            // isnertion succeeded need to incref key
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
    for (auto kv_pair: (*this->p_map)) {
        this->key_info.decref(kv_pair.first);
        free(kv_pair.first);
    }
    this->p_map->clear();
}

template<typename Val>
int8_t GenericHashmapType<void*, Val, VoidPtrHashCompare>::pop(void* key, void* res) {
    bool found = false;
    {
        typename map_type::const_accessor result;
        found = this->p_map->find(result, key);
        if (found)
        {
            memcpy(res, &(result->second), this->val_info.size);
            this->key_info.decref(result->first);
            free(result->first);
            // no decref for value since it would be returned (and no incref on python side!)
            this->p_map->erase(result);
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
        bool ok = this->p_map->insert(existing_node, inserted_node);
        if (ok)
        {
            this->key_info.incref(key);
            this->val_info.incref(val);
        }
        else
        {
            // insertion failed key already exists
            free(_key);

            this->val_info.decref(existing_node->second);
            free(existing_node->second);
            existing_node->second = _val;
            this->val_info.incref(val);
        }
    }
}

void GenericHashmapType<void*, void*, VoidPtrHashCompare>::clear()
{
    for (auto kv_pair: (*this->p_map)) {
        this->key_info.decref(kv_pair.first);
        free(kv_pair.first);
        this->val_info.decref(kv_pair.second);
        free(kv_pair.second);
    }
    this->p_map->clear();
}


int8_t GenericHashmapType<void*, void*, VoidPtrHashCompare>::pop(void* key, void* res) {
    bool found = false;
    {
        typename map_type::const_accessor result;
        found = this->p_map->find(result, key);
        if (found)
        {
            memcpy(res, result->second, this->val_info.size);
            free(result->second);
            this->key_info.decref(result->first);
            free(result->first);
            // no decref for value since it would be returned (and no incref on python side!)
            this->p_map->erase(result);
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
                         p_hash_map->p_map->insert(
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


template<typename key_type, typename val_type>
void hashmap_dump(void* p_hash_map)
{
    auto p_hash_map_spec = reinterpet_hashmap_ptr<key_type, val_type>(p_hash_map);
    auto size = p_hash_map_spec->p_map->size();
    std::cout << "Hashmap at: " << p_hash_map_spec << ", size = " << size << std::endl;
    for (auto kv_pair: (*p_hash_map_spec->p_map))
    {
        std::cout << "key, value: " << kv_pair.first << ", " << kv_pair.second << std::endl;
    }
    return;
}


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
    using itertype = typename std::remove_reference_t<decltype(*p_hash_map_spec)>::iterator_type;
    auto p_hash_map_iter = reinterpret_cast<itertype*>(p_iter_state_spec->first);
    static int iteration = 0;

    int8_t status = 1;
    if (*p_hash_map_iter != p_hash_map_spec->p_map->end())
    {
        *ret_key = (*p_hash_map_iter)->first;
        *ret_val = (*p_hash_map_iter)->second;
        status = 0;
        ++(*p_hash_map_iter);
    }

    return status;
}
