#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include "SharedStore.h"

namespace py = boost::python;
namespace np = boost::python::numpy;

template<class T>
py::class_<SharedStore<T>> register_store(const char* name) {
    return py::class_<SharedStore<T>>(name)
            .def( py::init<const char*, size_t, bool>(py::args("name", "size", "is_server")) )
            .def( "insert" , &SharedStore<T>::insert, py::args("key", "value"))
            .def( "get", &SharedStore<T>::get, py::args("key", "value"))
            .def( "finalize", &SharedStore<T>::finalize);
}

BOOST_PYTHON_MODULE( shared_tools ) {
    Py_Initialize();
    np::initialize();

    register_store<float>("SharedStore_fp32");
    register_store<uint8_t>("SharedStore_uint8");
}
