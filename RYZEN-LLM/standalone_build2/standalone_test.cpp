#include <pybind11/pybind11.h>

PYBIND11_MODULE(standalone_test, m) {
    m.def("simple_func", []() { return 999; });
    m.def("add_func", [](int a, int b) { return a + b; });
}