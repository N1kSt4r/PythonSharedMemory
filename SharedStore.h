#include <boost/python.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/numpy.hpp>

#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>

#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>

#include <chrono>
#include <vector>
#include <string>

#include <iostream>


#define MB *1014*1024

namespace py = boost::python;
namespace np = boost::python::numpy;
namespace sh = boost::interprocess;

sh::named_mutex mutex(sh::open_or_create, "Check");
sh::named_mutex init_store_lock(sh::open_or_create, "init_store_lock");


template<class T>
class SharedStore {
private:
    using ShTAllocType = sh::allocator<T, sh::managed_shared_memory::segment_manager>;

    using CKeyType       = size_t;
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

    void finalize();
    double get_time();
    size_t key2int(const PyKeyType&);
//    void active_wait(const PyKeyType&);

    // ToDo: use named mutex for sync and write method with dict/list
    void insert(const PyKeyType& key, const PyValueType& value);
    PyValueType get(const PyKeyType& key, PyValueType& value);

    void insert_dict(const py::dict& dict);
    void get_dict(py::dict& dict);

private:
//    sh::offset_ptr<sh::interprocess_semaphore> _reader;
//    sh::offset_ptr<sh::interprocess_mutex> _writer;

    sh::managed_shared_memory _segment;
    sh::offset_ptr<double> _timestamp;
    sh::offset_ptr<StoreType> _store;
    std::hash<std::string> _hasher;
    static bool _inited;
    std::string _name;
    bool _is_server;

//    int _active_wait_attempts;
//    int _active_wait_secs;
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
    sh::scoped_lock<sh::named_mutex> lock(init_store_lock);

    _is_server = is_server;
    assert(not _inited);
    _inited = true;
    _name = name;

    if (is_server) {
        sh::shared_memory_object::remove(name);
        _segment = sh::managed_shared_memory(sh::create_only, name, size MB);
        ShMapAllocType sh_allocator(_segment.get_segment_manager());

        _store = _segment.construct<StoreType>("store")(std::less<>(), sh_allocator);
        _timestamp = _segment.construct<double>("timestamp")(get_time());
    } else {
        _segment = sh::managed_shared_memory(sh::open_or_create, name, size MB);
        _timestamp = _segment.find<double>("timestamp").first;
        _store = _segment.find<StoreType>("store").first;
    }
}

template<class T>
SharedStore<T>::~SharedStore<T>() {
    finalize();
}

template<class T>
void SharedStore<T>::finalize() {
    if (_is_server) {
        sh::shared_memory_object::remove(_name.c_str());
    }
}

template<class T>
size_t SharedStore<T>::key2int(const PyKeyType& key) {
    return _hasher(py::extract<std::string>(key));
}

//template<class T>
//void SharedStore<T>::active_wait(const PyKeyType& key) {
//    auto c_key = key2int(key);
//    for (int i = 0; i < _active_wait_attempts; ++i) {
//        if (_store->count(c_key)) {
//            return;
//        }
//        sleep(_active_wait_secs);
//    }
//    throw std::runtime_error("Key not found");
//}

template<class T>
double SharedStore<T>::get_time() {
    return std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::system_clock::now().time_since_epoch()).count();
}

template<class T>
void SharedStore<T>::insert(const SharedStore::PyKeyType& key, const SharedStore::PyValueType& value) {
    auto c_key = key2int(key);

    if (!_store->count(c_key)) {
        SharedStore::ShTAllocType sh_allocator(_segment.get_segment_manager());
        _store->insert(std::make_pair(c_key, SharedStore::CValueType(value.shape(0), sh_allocator)));
    }
    
    from_ndarray<T>(value, &_store->at(c_key));
}

template<class T>
typename SharedStore<T>::PyValueType SharedStore<T>::get(
        const SharedStore::PyKeyType& key, SharedStore<T>::PyValueType& output) {
    auto c_key = key2int(key);
    from_vector(_store->at(c_key), &output);
    return output;
}

template<class T>
void SharedStore<T>::insert_dict(const py::dict& dict) {
    sh::scoped_lock<sh::named_mutex> lock(mutex);

    std::cout << "insert start" << std::endl;
    auto keys = dict.keys();
    auto values = dict.values();
    for (size_t i = 0; i < len(keys); ++i) {
         py::str key = py::extract<py::str>(keys[i]);
         np::ndarray value = py::extract<np::ndarray>(values[i]);
         insert(key, value);
    }
    std::cout << "insert end" << std::endl;

    *_timestamp = get_time();
    printf("set %lf %lf\n", *_timestamp, get_time());
}

template<class T>
void SharedStore<T>::get_dict(py::dict& dict) {
    sh::scoped_lock<sh::named_mutex> lock(mutex);

    std::cout << "get start" << std::endl;
    auto keys = dict.keys();
    auto values = dict.values();
    for (size_t i = 0; i < len(keys); ++i) {
        py::str key = py::extract<py::str>(keys[i]);
        np::ndarray value = py::extract<np::ndarray>(values[i]);
        get(key, value);
    }
    std::cout << "get end" << std::endl;

    std::cout << get_time() - *_timestamp << std::endl;
    printf("get %lf %lf\n", *_timestamp, get_time());
}

template<class T> bool SharedStore<T>::_inited = false;
