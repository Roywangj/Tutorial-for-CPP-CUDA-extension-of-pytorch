#include <torch/extension.h>
#include "trillnear_devox_diff_R.h"

// pybind11 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &trilinear_devoxelize_diffR_forward, "TEST forward");
}