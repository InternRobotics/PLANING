#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_convexes", &RasterizeConvexesCUDA);
  m.def("rasterize_convexes_backward", &RasterizeConvexesBackwardCUDA);
  m.def("mark_visible", &markVisible);
}