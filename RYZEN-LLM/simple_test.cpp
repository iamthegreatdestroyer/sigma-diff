#include <pybind11/pybind11.h>

PYBIND11_MODULE(simple_test, m)
{
    m.def("simple_func", []()
          { return 999; });
}