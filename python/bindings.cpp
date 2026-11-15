#include <pybind11/pybind11.h>
#include "Mesh2D.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "Solver2D.h"
#include <cstdint>
#include <vector>
#include <pybind11/functional.h>
#include "Assembly2D.h"
#include "Thermoelastic2D.h"
#include "Elasticity2D.h"
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

    m.def("solve_steady_2d", [](const Mesh2D& mesh,
                                double k,
                                const std::vector<double>& f,
                                const std::vector<std::size_t>& fixed_nodes,
                                const std::vector<double>& fixed_values) {
        return to_array(solve_steady_2d(mesh, k, f, fixed_nodes, fixed_values));
    }, py::arg("mesh"),
       py::arg("k"),
       py::arg("f"),
       py::arg("fixed_nodes"),
       py::arg("fixed_values"));

    m.def("solve_transient_2d", [](const Mesh2D& mesh,
                                   double rho_c,
                                   double k,
                                   const std::vector<double>& f,
                                   const std::vector<std::size_t>& fixed_nodes,
                                   const std::vector<double>& fixed_values,
                                   const std::vector<double>& T0,
                                   double dt,
                                   std::size_t n_steps,
                                   double theta) {
        return to_array(solve_transient_2d(mesh, rho_c, k, f, fixed_nodes, fixed_values, T0, dt, n_steps, theta));
    }, py::arg("mesh"),
       py::arg("rho_c"),
       py::arg("k"),
       py::arg("f"),
       py::arg("fixed_nodes"),
       py::arg("fixed_values"),
       py::arg("T0"),
       py::arg("dt"),
       py::arg("n_steps"),
       py::arg("theta"));


    m.def("solve_thermomechanical_2d", [](
        const Mesh2D& mesh,
        double rho_c,
        double k,
        const std::vector<double>& f,
        const std::vector<std::size_t>& fixed_nodes,
        const std::vector<double>& fixed_temperatures,
        const std::vector<double>& T0,
        double dt,
        std::size_t n_steps,
        double theta,
        std::size_t save_every,
        double E,
        double nu,
        double alpha,
        double T_ref,
        const std::vector<std::size_t>& fixed_dofs,
        const std::vector<double>& fixed_displacements
    ) {
        ThermoMechanicalProblem2D problem;
        problem.rho_c = rho_c;
        problem.k = k;
        problem.f = f;
        problem.fixed_nodes = fixed_nodes;
        problem.fixed_temperatures = fixed_temperatures;
        problem.T0 = T0;
        problem.dt = dt;
        problem.n_steps = n_steps;
        problem.theta = theta;
        problem.save_every = save_every;
        problem.E = E;
        problem.nu = nu;
        problem.alpha = alpha;
        problem.T_ref = T_ref;
        problem.fixed_dofs = fixed_dofs;
        problem.fixed_displacements = fixed_displacements;

        auto states = solve_thermomechanical_2d(mesh, problem);

        const std::size_t m = states.size();
        const std::size_t n = mesh.num_nodes();
        const std::size_t n_tri = mesh.num_triangles();

        std::vector<double> times;
        std::vector<double> T_all;
        std::vector<double> u_all;
        std::vector<double> sigma_all;
        std::vector<double> vm_all;

        times.reserve(m);
        T_all.reserve(m * n);
        u_all.reserve(m * 2 * n);
        sigma_all.reserve(m * n_tri * 3);
        vm_all.reserve(m * n_tri);

        for (const auto& state : states) {
            times.push_back(state.time);
            
            T_all.insert(T_all.end(), state.T.begin(), state.T.end());
            u_all.insert(u_all.end(), state.u.begin(), state.u.end());

            for (const auto& sig : state.sigma) {
                sigma_all.push_back(sig[0]); // xx
                sigma_all.push_back(sig[1]); // yy
                sigma_all.push_back(sig[2]); // xy
            }

            auto vm = von_mises_field_2d(mesh, state.sigma, E, nu, alpha, state.T, T_ref);
            vm_all.insert(vm_all.end(), vm.begin(), vm.end());
        }

        py::dict result;
        result["time"] = to_array(times);
        result["T"] = to_array(T_all).attr("reshape")(m, n);
        result["u"] = to_array(u_all).attr("reshape")(m, 2 * n);
        result["sigma"] = to_array(sigma_all).attr("reshape")(m, n_tri, 3);
        result["von_mises"] = to_array(vm_all).attr("reshape")(m, n_tri);

        return result;
    },
    py::arg("mesh"),
    py::arg("rho_c"),
    py::arg("k"),
    py::arg("f"),
    py::arg("fixed_nodes"),
    py::arg("fixed_temperatures"),
    py::arg("T0"),
    py::arg("dt"),
    py::arg("n_steps"),
    py::arg("theta"),
    py::arg("save_every"),
    py::arg("E"),
    py::arg("nu"),
    py::arg("alpha"),
    py::arg("T_ref"),
    py::arg("fixed_dofs"),
    py::arg("fixed_displacements"));

}