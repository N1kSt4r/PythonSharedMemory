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
#include <iostream>
#include <unordered_set>

#define MB *1024*1024

namespace py = boost::python;
namespace np = boost::python::numpy;
namespace sh = boost::interprocess;

sh::named_mutex init_store_lock(sh::open_or_create, "init_store_lock");
std::unordered_set<std::string> _inited;


struct SharedSync {
    int read_count = 0;
    sh::interprocess_mutex global_mutex;
    sh::interprocess_mutex writer_mutex;
    sh::interprocess_mutex reader_mutex;
};


class ReaderLock {
public:
    ReaderLock(SharedSync* sync) {
        _sync = sync;

        sh::scoped_lock<sh::interprocess_mutex> reader_lock(_sync->reader_mutex);
        sh::scoped_lock<sh::interprocess_mutex> global_lock(_sync->global_mutex);
        if (++_sync->read_count == 1) {
            _sync->writer_mutex.lock();
        }
    }

    ~ReaderLock() {
        sh::scoped_lock<sh::interprocess_mutex> global_lock(_sync->global_mutex);
        if (--_sync->read_count == 0) {
            _sync->writer_mutex.unlock();
        }
    }

private:
    SharedSync* _sync;
};


template<class T>
class SharedStore {
private:
    using ShTAllocType = sh::allocator<T, sh::managed_shared_memory::segment_manager>;

    static const size_t max_dim = 3;

    using PyKeyType      = py::str;
    using PyValueType    = np::ndarray;
    using ShapeType      = std::array<size_t, max_dim + 1>;

    using CKeyType       = size_t;
    using CValueType     = std::vector<T, ShTAllocType>;
    using CValuePairType = std::pair<ShapeType, CValueType>;

    using MapPairType    = std::pair<const CKeyType, CValuePairType>;
    using ShMapAllocType = sh::allocator<MapPairType, sh::managed_shared_memory::segment_manager>;
    using StoreType      = sh::map<CKeyType, CValuePairType, std::less<>, ShMapAllocType>;

    ShapeType pyshape2cshape(const np::ndarray& array);
    py::tuple cshape2pyshape(const ShapeType& shape);

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

    template<bool is_init>
    void get_dict_meta(py::dict& dict);
    void insert_dict(const py::dict& dict);

    auto& __enter__();
    void __exit__(const py::object& a, const py::object& c, const py::object& d);

private:
    const double _max_timestamp_diff = 5;
    const int _wait_secs = 5;
    const int _attempts = 5;

    sh::managed_shared_memory _segment;
    sh::offset_ptr<double> _timestamp;
    sh::offset_ptr<SharedSync> _sync;
    sh::offset_ptr<StoreType> _store;
    std::hash<std::string> _hasher;
    std::string _name;
    bool _is_server;
};


size_t get_size(const np::ndarray& ndarray) {
    return py::extract<size_t>(ndarray.attr("size"));
}

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
    for (int ntry = 0; ntry < _attempts; ++ntry) {
        try {
            sh::scoped_lock<sh::named_mutex> lock(init_store_lock);
            if (_inited.count(name)) {
                std::cout << "\x1b[31mWarning: double init " << name << " store\x1b[0m" << std::endl;
            }

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
            _inited.insert(name);
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
        CValueType c_value(get_size(value), sh_allocator);
        CValuePairType value_pair = std::make_pair(pyshape2cshape(value), c_value);
        _store->insert(std::make_pair(c_key, value_pair));
    }
    
    from_ndarray<T>(value, &_store->at(c_key).second);
}

template<class T>
typename SharedStore<T>::PyValueType SharedStore<T>::get(const PyKeyType& key, PyValueType& output) {
    from_vector(_store->at(key2int(key)).second, &output);
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
template<bool is_init>
void SharedStore<T>::get_dict_meta(py::dict& dict) {
    ReaderLock(_sync.get());

    if (get_time() - *_timestamp > _max_timestamp_diff) {
        throw std::runtime_error("Too old last update");
    }

    auto keys = dict.keys();
    auto values = dict.values();
    for (size_t i = 0; i < len(keys); ++i) {
        PyKeyType key = py::extract<py::str>(keys[i]);
        if constexpr (is_init) {
            active_wait(key);
            py::tuple shape = cshape2pyshape(_store->at(key2int(key)).first);
            np::dtype type = np::dtype::get_builtin<T>();
            PyValueType value = np::empty(shape, type);
            dict[key] = value;
            get(key, value);
        } else {
            PyValueType value = py::extract<PyValueType>(values[i]);
            get(key, value);
        }
    }
}

template<class T>
typename SharedStore<T>::ShapeType SharedStore<T>::pyshape2cshape(const np::ndarray& array) {
    size_t dims = array.get_nd();
    assert(dims <= max_dim);
    ShapeType shape;

    shape[0] = dims;
    for (int i = 0; i < dims; ++i) {
        shape[i + 1] = array.shape(i);
    }
    return shape;
}

template<class T>
py::tuple SharedStore<T>::cshape2pyshape(const ShapeType& shape) {
    py::list pyshape;
    for (int i = 0; i < shape[0]; ++i) {
        pyshape.append(shape[i + 1]);
    }
    return py::tuple(pyshape);
}

template<class T>
auto& SharedStore<T>::__enter__() {
    return *this;
}
template<class T>
void SharedStore<T>::__exit__(const py::object& a, const py::object& c, const py::object& d) {
    finalize();
}
