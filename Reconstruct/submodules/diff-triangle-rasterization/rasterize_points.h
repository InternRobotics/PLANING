#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeConvexesCUDA(
	const torch::Tensor& background,
	const torch::Tensor& convex_points,
	const torch::Tensor& delta,
	const torch::Tensor& sigma,
	const torch::Tensor& num_split_per_convex, 
	const torch::Tensor& cumsum_of_subtriangles_per_convex,  
	const torch::Tensor& num_points_per_convex,
	const torch::Tensor& cumsum_of_points_per_convex,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	torch::Tensor& scaling,
	torch::Tensor& density_factor,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const int number_of_points,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeConvexesBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& convex_points,
	const torch::Tensor& delta,
	const torch::Tensor& sigma,
	const torch::Tensor& num_points_per_convex,
	const torch::Tensor& cumsum_of_points_per_convex,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const int number_of_points,
	const int sub_triangle_total_num,                      // sub_triangle_total_num
	const float tan_fovx, 
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_others,             // dL_dout_others for geometry Loss
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& subGeometryBuffer,         // subGeometryBuffer for sub triangle
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug);
		
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);