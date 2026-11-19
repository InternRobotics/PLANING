#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> subGeometryBuffer, 
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* convex_points,
			const float* delta,
			const float* sigma,
			const int* num_split_per_convex,     
			const int* cumsum_of_subtriangles_per_convex,    // for sub triangle
			const int sub_triangle_total_num,           // for sub triangle
			const int* num_points_per_convex,
			const int* cumsum_of_points_per_convex,
			const int total_nb_points,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			float* scaling,
			float* density_factor,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			float* out_others,
			int* radii = nullptr,
			float* impact = nullptr,                                                                      
			bool debug = false);

		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* convex_points,
			const float* delta,
			const float* sigma,
			const int* num_points_per_convex,
			const int* cumsum_of_points_per_convex,
			const int total_nb_points,
			const float* shs,
			const float* colors_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const int sub_triangle_total_num,           // num_sub_triangle
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* subGeom_buffer,      // for sub triangle
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			const float* dL_depths,                             
			float* dL_dmeans3D,
			float* dL_dmeans2D,
			float* dL_dtransMat,      // for geo Loss
			float* dL_da,             // for geo Loss
			float* dL_dnormal3D,      // for geo Loss
			float* dL_dcov3D,
			float* dL_dconvex,
			float* dL_ddelta,
			float* dL_dsigma,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dsh,
			bool debug);
	};
};

#endif