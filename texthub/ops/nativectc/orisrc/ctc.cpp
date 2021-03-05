#include "ctc.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nativectc_forward", &nativectc_forward, "nativectc_forward (cuda)");
  m.def("nativectc_backward", &nativectc_backward, "nativectc_backward (cuda)");
}