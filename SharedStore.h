#include <boost/python.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/numpy.hpp>

#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>

#include <string>
#include <array>

#define MB *1014*1024

namespace py = boost::python;
namespace np = boost::python::numpy;
namespace sh = boost::interprocess;


template<class T>
class SharedStore {
private:
    using ShTAllocType = sh::allocator<T, sh::managed_shared_memory::segment_manager>;

    using CKeyType       = int;
    using PyKeyType      = py::str;
    using CValueType     = std::vector<T, ShTAllocType>;
    using PyValueType    = np::ndarray;

    using MapPairType    = std::pair<const CKeyType, CValueType>;
    using ShMapAllocType = sh::allocator<MapPairType, sh::managed_shared_memory::segment_manager>;
    using StoreType      = sh::map<CKeyType, CValueType , std::less<>, ShMapAllocType>;

public:
    SharedStore() {
        throw std::runtime_error("Default constructor is missing");
    }
    SharedStore(const typename std::reference_wrapper<const struct SharedStore<T>>::type& type) {
        throw std::runtime_error("Copy constructor is missing");
    }

    SharedStore(const char* name, size_t size, bool is_server);
    ~SharedStore();

    // ToDo: use named mutex for sync and write method with dict/list
    void insert(const PyKeyType& key, PyValueType& value);
    PyValueType get(const PyKeyType& key, PyValueType& value);

private:
    sh::managed_shared_memory _segment;
    sh::offset_ptr<StoreType> _store;
    static bool _inited;
    std::string _name;
    bool _is_server;
};


template<class T, class A>
void from_vector(const std::vector<T, A>& vector, np::ndarray* ndarray) {
    assert(vector.size() == ndarray->shape(0));
    memcpy(ndarray->get_data(), vector.data(), vector.size() * sizeof(T));
}

template<class T, class A>
void from_ndarray(const np::ndarray& ndarray, std::vector<T, A>* vector) {
    assert(vector->size() == ndarray.shape(0));
    memcpy(vector->data(), ndarray.get_data(), ndarray.shape(0) * sizeof(T));
}

template<class T>
SharedStore<T>::SharedStore(const char* name, size_t size, bool is_server) {
    _is_server = is_server;
    assert(not _inited);
    _inited = true;

    std::string _name(name);
    std::string store_name = _name + "_store";

    if (is_server) {
        sh::shared_memory_object::remove(name);
        _segment = sh::managed_shared_memory(sh::create_only, name, size MB);
        ShMapAllocType sh_allocator(_segment.get_segment_manager());
        _store = _segment.construct<StoreType>(store_name.c_str())(std::less<>(), sh_allocator);
    } else {
        _segment = sh::managed_shared_memory(sh::open_or_create, name, size MB);
        _store = _segment.find<StoreType>(store_name.c_str()).first;
    }
}

template<class T>
SharedStore<T>::~SharedStore<T>() {
    if (_is_server) {
        sh::shared_memory_object::remove(_name.c_str());
    }
}

template<class T>
void SharedStore<T>::insert(const SharedStore::PyKeyType& key, SharedStore::PyValueType& value) {
    int new_key = std::atoi(py::extract<const char*>(key));

    if (!_store->count(new_key)) {
        SharedStore::ShTAllocType sh_allocator(_segment.get_segment_manager());
        _store->insert(std::make_pair(new_key, SharedStore::CValueType(value.shape(0), sh_allocator)));
    }
    
    from_ndarray<T>(value, &_store->at(new_key));
}

template<class T>
typename SharedStore<T>::PyValueType SharedStore<T>::get(
        const SharedStore::PyKeyType& key, SharedStore<T>::PyValueType& output) {
    int new_key = std::atoi(py::extract<const char*>(key));
    from_vector(_store->at(new_key), &output);
    return output;
}

template<class T> bool SharedStore<T>::_inited = false;
