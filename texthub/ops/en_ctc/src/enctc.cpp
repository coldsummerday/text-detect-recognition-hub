#include "enctc.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("enctc_forward", &enctc_forward, "enctc_forward (cuda)");
  m.def("enctc_backward", &enctc_backward, "enctc_backward (cuda)");
}