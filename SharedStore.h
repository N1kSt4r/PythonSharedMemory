#include <boost/python.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/numpy.hpp>

#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>

#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>

#include <chrono>
#include <vector>
#include <string>

#define MB *1014*1024

namespace py = boost::python;
namespace np = boost::python::numpy;
namespace sh = boost::interprocess;

sh::named_mutex init_store_lock(sh::open_or_create, "init_store_lock");


struct SharedSync {
    int read_count = 0;
    sh::interprocess_mutex global_mutex;
    sh::interprocess_mutex writer_mutex;
    sh::interprocess_mutex reader_mutex;
};


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
    void active_wait(const PyKeyType&);

    void insert(const PyKeyType& key, const PyValueType& value);
    PyValueType get(const PyKeyType& key, PyValueType& output);

    void insert_dict(const py::dict& dict);
    void get_dict(py::dict& dict);

private:
    const double _max_timestamp_diff = 5;
    const int _wait_secs = 5;
    const int _attempts = 5;

    sh::managed_shared_memory _segment;
    sh::offset_ptr<double> _timestamp;
    sh::offset_ptr<SharedSync> _sync;
    sh::offset_ptr<StoreType> _store;
    std::hash<std::string> _hasher;
    static bool _inited;
    std::string _name;
    bool _is_server;
};


size_t get_size(const np::ndarray& ndarray) {
    return py::extract<size_t>(ndarray.attr("size"));
}

#include <iostream>
template<class T, class A>
void from_vector(const std::vector<T, A>& vector, np::ndarray* ndarray) {
    assert(ndarray->get_flags() & np::ndarray::bitflag::C_CONTIGUOUS);
    assert(vector.size() == get_size(*ndarray));
    memcpy(ndarray->get_data(), vector.data(), vector.size() * sizeof(T));
}

template<class T, class A>
void from_ndarray(const np::ndarray& ndarray, std::vector<T, A>* vector) {
    assert(ndarray.get_flags() & np::ndarray::bitflag::C_CONTIGUOUS);
    assert(vector->size() == get_size(ndarray));
    memcpy(vector->data(), ndarray.get_data(), vector->size() * sizeof(T));
}

template<class T>
SharedStore<T>::SharedStore(const char* name, size_t size, bool is_server) {
    for (int ntry = 0; ; ++ntry) {
        try {
            sh::scoped_lock<sh::named_mutex> lock(init_store_lock);
            if (is_server) {
                sh::shared_memory_object::remove(name);
                _segment = sh::managed_shared_memory(sh::create_only, name, size MB);
                ShMapAllocType sh_allocator(_segment.get_segment_manager());

                _store = _segment.construct<StoreType>("store")(std::less<>(), sh_allocator);
                _timestamp = _segment.construct<double>("timestamp")(get_time());
                _sync = _segment.construct<SharedSync>("sync")();
            } else {
                _segment = sh::managed_shared_memory(sh::open_only, name);

                _timestamp = _segment.find<double>("timestamp").first;
                _store = _segment.find<StoreType>("store").first;
                _sync = _segment.find<SharedSync>("sync").first;
            }

            _is_server = is_server;
            assert(not _inited);
            _inited = true;
            _name = name;
            break;
        } catch (std::exception& exc) {
            if (ntry >= _attempts) {
                throw std::runtime_error(exc.what());
            } else {
                std::cout << name << ": " << exc.what() << std::endl;
                sleep(_wait_secs);
            }
        }
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

template<class T>
void SharedStore<T>::active_wait(const PyKeyType& key) {
   auto c_key = key2int(key);
   for (int i = 0; i < _attempts; ++i) {
       if (_store->count(c_key)) {
           return;
       }
       sleep(_wait_secs);
   }
   throw std::runtime_error("Key not found");
}

template<class T>
double SharedStore<T>::get_time() {
    return std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::system_clock::now().time_since_epoch()).count();
}

template<class T>
void SharedStore<T>::insert(const PyKeyType& key, const PyValueType& value) {
    auto c_key = key2int(key);

    if (!_store->count(c_key)) {
        SharedStore::ShTAllocType sh_allocator(_segment.get_segment_manager());
        _store->insert(std::make_pair(c_key, SharedStore::CValueType(get_size(value), sh_allocator)));
    }
    
    from_ndarray<T>(value, &_store->at(c_key));
}

template<class T>
typename SharedStore<T>::PyValueType SharedStore<T>::get(const PyKeyType& key, PyValueType& output) {
    from_vector(_store->at(key2int(key)), &output);
    return output;
}

template<class T>
void SharedStore<T>::insert_dict(const py::dict& dict) {
    sh::scoped_lock<sh::interprocess_mutex> reader_lock(_sync->reader_mutex);
    sh::scoped_lock<sh::interprocess_mutex> writer_lock(_sync->writer_mutex);

    auto keys = dict.keys();
    auto values = dict.values();
    for (size_t i = 0; i < len(keys); ++i) {
         py::str key = py::extract<py::str>(keys[i]);
         np::ndarray value = py::extract<np::ndarray>(values[i]);
         insert(key, value);
    }

    *_timestamp = get_time();
}

template<class T>
void SharedStore<T>::get_dict(py::dict& dict) {
    {
        sh::scoped_lock<sh::interprocess_mutex> reader_lock(_sync->reader_mutex);
        sh::scoped_lock<sh::interprocess_mutex> global_lock(_sync->global_mutex);
        if (++_sync->read_count == 1) {
            _sync->writer_mutex.lock();
        }
    }

    if (get_time() - *_timestamp > _max_timestamp_diff) {
        throw std::runtime_error("Too old last update");
    }

    auto keys = dict.keys();
    auto values = dict.values();
    for (size_t i = 0; i < len(keys); ++i) {
        py::str key = py::extract<py::str>(keys[i]);
        np::ndarray value = py::extract<np::ndarray>(values[i]);
        get(key, value);
    }

    {
        sh::scoped_lock<sh::interprocess_mutex> global_lock(_sync->global_mutex);
        if (--_sync->read_count == 0) {
            _sync->writer_mutex.unlock();
        }
    }
}

template<class T> bool SharedStore<T>::_inited = false;
