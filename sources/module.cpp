#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include "SharedStore.h"

namespace py = boost::python;
namespace np = boost::python::numpy;

template<class T>
py::class_<SharedStore<T>> register_store(const char* name) {
    return py::class_<SharedStore<T>>(name)
        .def( py::init<const char*, size_t, bool>(py::args("name", "size", "is_server")) )
        .def( "get_dict_init" , &SharedStore<T>::template get_dict_meta<true>, py::args("dict") )
        .def( "get_dict" , &SharedStore<T>::template get_dict_meta<false>, py::args("dict") )
        .def( "insert_dict" , &SharedStore<T>::insert_dict, py::args("dict") )
        .def( "finalize", &SharedStore<T>::finalize )

        .def( "__enter__", &SharedStore<T>::__enter__, py::return_value_policy<py::reference_existing_object>() )
        .def( "__exit__", &SharedStore<T>::__exit__ );
}

BOOST_PYTHON_MODULE( shared_store ) {
    Py_Initialize();
    np::initialize();

    register_store<uint8_t>("SharedStore_uint8");
    register_store<float>("SharedStore_fp32");
}
