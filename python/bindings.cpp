#include <pybind11/pybind11.h>
#include "Mesh2D.h"

namespace py = pybind11;

PYBIND11_MODULE(thermomech, m) {
    m.doc() = "thermomech: Python interface of the C++ FEM core";

    py::class_<Mesh2D>(m, "Mesh2D")
        .def_property_readonly("num_nodes", &Mesh2D::num_nodes)
        .def_property_readonly("num_triangles", &Mesh2D::num_triangles);

    m.def("rectangle_mesh", 
        static_cast<Mesh2D (*)(double, double, std::size_t, std::size_t)>(&rectangle_mesh),
        py::arg("length_x"), 
        py::arg("length_y"), 
        py::arg("nx"), 
        py::arg("ny"));

}