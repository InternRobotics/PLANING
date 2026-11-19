# Reconstruction source

This directory retains the `plane_v1` model, shared keyframe/optimizer utilities,
mesh export, and the two custom CUDA extensions. See the [root README](../README.md)
for the supported entry point, dependency inventory, and static cleanup status.

The Gaussian CUDA extension is still required for its custom Adam operators even
though `plane_v1` renders Gaussians through `gsplat`. Do not substitute an unrelated
upstream rasterizer build. The triangle CUDA extension and its bundled headers are
also retained unchanged.

Original copyright headers and [license](LICENSE.md) remain in place. Each CUDA
extension contains its own license files.
