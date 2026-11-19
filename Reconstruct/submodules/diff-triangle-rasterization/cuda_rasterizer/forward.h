#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	void preprocess(int P, int D, int M,
		const float* convex_points,
		const float* delta,
		const float* sigma,
		const int* num_split_per_convex,     
		const int* cumsum_of_subtriangles_per_convex,    
		const int* num_points_per_convex,
		const int* cumsum_of_points_per_convex,
		const float* opacities,
		float* scaling,
		float* density_factor,
		const float* shs,
		bool* clamped,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* normals,
		float* offsets,
		int* num_points_per_convex_view,
		float4* p_hom,
		float* p_w,
		float3* p_proj,
		float2* p_image,
		int* hull,
		int* indices,
		float2* points_xy_image,
		float* depths,
		float* transMats,                                        
		float* colors,
		float4* conic_opacity,
		float* cov3Ds,
		const dim3 grid,
		uint32_t* tiles_touched,
		uint32_t* sub_triangle_father_index,    
		float3* sub_points_world,               
		bool*  sub_triangle_visibility,         
		bool prefiltered);

	void subpreprocess(
		int sub_triangle_total_num,
		const float3* p_world,                       
        bool* sub_triangle_visibility,                                    
		float* depths,                                  
		float2* p_image,                                    
		float2* center_means2D,                          
		const dim3 grid,
		uint32_t* tiles_touched,                           
		int* radii_2D,
		const float* viewmatrix, 
		const float* projmatrix,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy
	);

	void render(
		const dim3 grid, dim3 block,
		int sub_triangle_total_num,              
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* normals,
		const float* offsets,
		const int* num_points_per_convex_view,
		const float2* p_tangent,                                                           
		const float* delta,
		const float* sigma,
		const int* num_points_per_convex,
		const int* cumsum_of_points_per_convex,
		const float* features,
		const float* transMats,        
		const float4* conic_opacity,
		const float* depths,
		float* final_T,
		uint32_t* n_contrib,
		const float* sub_triangle_depths,                        
		const float2* sub_triangle_p_image,                            
		const uint32_t* sub_triangle_father_index,     
		const bool* sub_triangle_visibility,                      
		const float* bg_color,
		float* out_color,
		float* out_others,
		float* impact);                                                                 
}


#endif