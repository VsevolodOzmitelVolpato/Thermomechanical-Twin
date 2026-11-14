#include <pybind11/pybind11.h>
#include "Mesh2D.h"
#include <pybind11/numpy.h>
#include <cstdint>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(thermomech, m) {
    m.doc() = "thermomech: Python interface of the C++ FEM core";

    py::class_<Mesh2D>(m, "Mesh2D")
        .def_property_readonly("num_nodes", &Mesh2D::num_nodes)
        .def_property_readonly("num_triangles", &Mesh2D::num_triangles)

        .def_property_readonly("x", [](const Mesh2D& mesh) {
            const std::size_t n = mesh.num_nodes();
            py::array_t<double> arr(n);
            double* ptr = arr.mutable_data();
            
            for (std::size_t i = 0; i < n; ++i) {
                ptr[i] = mesh.x(i);
            }
            arr.attr("setflags")(py::arg("write") = false);

            return arr;
        })

        .def_property_readonly("y", [](const Mesh2D& mesh) {
            const std::size_t n = mesh.num_nodes();
            py::array_t<double> arr(n);
            double* ptr = arr.mutable_data();
            
            for (std::size_t i = 0; i < n; ++i) {
                ptr[i] = mesh.y(i);
            }
            arr.attr("setflags")(py::arg("write") = false);
            return arr;
        })

        .def_property_readonly("triangles", [](const Mesh2D& mesh) {
            const std::size_t nt = mesh.num_triangles();
            py::array_t<std::int64_t> arr({nt, std::size_t{3}});
            auto ptr = arr.mutable_unchecked<2>(); // Accesso veloce 2D senza controlli di bordo ridondanti
            
            for (std::size_t e = 0; e < nt; ++e) {
                const Triangle& t = mesh.triangle(e);
                for (std::size_t k = 0; k < 3; ++k) {
                    ptr(e, k) = static_cast<std::int64_t>(t[k]);
                }
            }
            arr.attr("setflags")(py::arg("write") = false);
            return arr;
        });

    m.def("rectangle_mesh", 
        static_cast<Mesh2D (*)(double, double, std::size_t, std::size_t)>(&rectangle_mesh),
        py::arg("length_x"), 
        py::arg("length_y"), 
        py::arg("nx"), 
        py::arg("ny"));

}