#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <vector>

#include "nl2pc_ckks.h"


namespace py = pybind11;


PYBIND11_MODULE(nl2pc, m) {

    m.doc() = "2PC nonlinear based on CKKS";

    py::enum_<cs_role>(m, "ckks_role", py::arithmetic())
        .value("SERVER", cs_role::ckksSERVER)
        .value("CLIENT", cs_role::ckksCLIENT)
        .export_values();

    py::class_<NLParty>(m, "NLParty")
        .def(py::init<cs_role, const std::string&, const uint16_t, uint32_t, seal::EncryptionParameters, seal::SecretKey, seal::PublicKey, seal::GaloisKeys, double, bool, bool>())

        .def("cmp", &NLParty::cmp, py::return_value_policy::reference_internal)

        .def("relu", &NLParty::relu, py::return_value_policy::reference_internal,
            py::arg("input"),
            py::arg("rot") = false)
        
        .def("maxpool2d", &NLParty::maxpool2d, py::return_value_policy::reference_internal,
            py::arg("input"),
            py::arg("patch_size"),
            py::arg("rot") = false)

        .def("is_server", &NLParty::is_server)
        
        .def("__repr__", [](const NLParty &a) {
            return "<relu2pc.NLParty>";
        });

    m.def("Create", &create, py::return_value_policy::reference_internal, "construct a NLParty",
        py::arg("ckks_role"),
        py::arg("address") = "127.0.0.1",
        py::arg("port") = 7766,
        py::arg("nthreads") = 1,
        py::arg("cmpr") = false,
        py::arg("verbose") = false);
}