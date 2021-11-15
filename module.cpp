#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include "SharedStore.h"

namespace py = boost::python;
namespace np = boost::python::numpy;


np::ndarray greet(np::ndarray const & array) {
    return array;
}


py::dict get_dict(const py::dict& dict) {
    return dict;
}


py::str get_str(const py::str& str) {
    return str;
}


BOOST_PYTHON_MODULE( shared_tools ) {
    Py_Initialize();
    np::initialize();

    py::def("greet", greet);
    py::def("get_dict", get_dict);

    py::class_<SharedStore<float>>("SharedStore_fp32")
            .def( py::init<const char*, size_t, bool>(py::args("name", "size", "is_server")) )
            .def( "insert" , &SharedStore<float>::insert, py::args("key", "value"))
            .def( "get", &SharedStore<float>::get, py::args("key", "value"));

    py::class_<SharedStore<uint8_t>>("SharedStore_uint8")
            .def( py::init<const char*, size_t, bool>(py::args("name", "size", "is_server")) )
            .def( "insert" , &SharedStore<float>::insert, py::args("key", "value"))
            .def( "get", &SharedStore<float>::get, py::args("key", "value"));
}
