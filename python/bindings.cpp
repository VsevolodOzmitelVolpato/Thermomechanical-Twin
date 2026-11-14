#include <pybind11/pybind11.h>
#include "Mesh2D.h"
#include <pybind11/numpy.h>
#include <cstdint>
#include <vector>
#include <pybind11/functional.h>
#include "Assembly2D.h"
#include "Triplets.h"

namespace py = pybind11;

namespace {
    py::array_t<double> to_array(const std::vector<double>& v) {
    py::array_t<double> arr(v.size());

    double* ptr = arr.mutable_data();

    for (std::size_t i = 0; i < v.size(); ++i) {
        ptr[i] = v[i];}

    return arr;
}

py::object to_csr(const std::vector<Triplet>& triplets, std::size_t n) {
    const std::size_t nnz = triplets.size();

    py::array_t<std::int64_t> rows(nnz);
    py::array_t<std::int64_t> cols(nnz);
    py::array_t<double> vals(nnz);

    std::int64_t* r_ptr = rows.mutable_data();
    std::int64_t* c_ptr = cols.mutable_data();
    double* v_ptr = vals.mutable_data();

    for (std::size_t i = 0; i < nnz; ++i) {
        r_ptr[i] = static_cast<std::int64_t>(triplets[i].row);
        c_ptr[i] = static_cast<std::int64_t>(triplets[i].col);
        v_ptr[i] = triplets[i].val;
    }

    py::object coo_matrix = py::module_::import("scipy.sparse").attr("coo_matrix");
    py::object coo = coo_matrix(py::make_tuple(vals, py::make_tuple(rows, cols)),
                                py::arg("shape") = py::make_tuple(n, n));
    return coo.attr("tocsr")();
}
}

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

    m.def("assemble_stiffness_2d", [](const Mesh2D& mesh, double k) {
        return to_csr(stiffness_triplets_2d(mesh, k), mesh.num_nodes());
    },  py::arg("mesh"), py::arg("k"));
    
    m.def("assemble_mass_2d", [](const Mesh2D& mesh, double rho_c) {
        return to_csr(mass_triplets_2d(mesh, rho_c), mesh.num_nodes());
    },  py::arg("mesh"), py::arg("rho_c"));

    m.def("assemble_load_2d", [](const Mesh2D& mesh, double q) {
        return to_array(assemble_load_2d(mesh, q));
    },  py::arg("mesh"), py::arg("q"));

    m.def("assemble_load_2d", [](const Mesh2D& mesh, const std::function<double(double, double)>& q) {
        if (!q) {
            throw py::type_error("q must be a float or a callable function, not None");}
        return to_array(assemble_load_2d(mesh, q));
    },  py::arg("mesh"), py::arg("q"));

}