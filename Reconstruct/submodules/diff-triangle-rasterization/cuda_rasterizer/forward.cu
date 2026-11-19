#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>


#include <cstdio>


namespace cg = cooperative_groups;

__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3 means, glm::vec3 campos, const float* shs, bool* clamped)
{
	glm::vec3 pos = means;
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

__device__ void compute_transmat(
	const float3& center_convex,
	const float3& convex_p1,
	const float3& convex_p2,
	const float3& convex_p3,
	const float* projmatrix,
	const float* viewmatrix,
	const int W,
	const int H, 
	glm::mat3 &T,
	float3 &normal,
    float2& p1_tangent,    
    float2& p2_tangent,    
    float2& p3_tangent     
) {
	                                                           
	// calculate p1p2 and p1p3
	float3 p1_p2 = convex_p2 - convex_p1;
	float3 p1_p3 = convex_p3 - convex_p1;
 
	float3 normal3D = cross3D(p1_p2, p1_p3);        

	float s_normal3D = magnitude3D(normal3D) + 0.0000001f;              
	float r_normal3D = 1.0 / s_normal3D;   
	  
	normal3D.x = normal3D.x * r_normal3D;
	normal3D.y = normal3D.y * r_normal3D;
	normal3D.z = normal3D.z * r_normal3D;

	 
	                                                                              
	float3 pc_p1 = convex_p1 - center_convex;
	float3 pc_p2 = convex_p2 - center_convex;
	float3 pc_p3 = convex_p3 - center_convex;

	
	float s_x = magnitude3D(pc_p1) + 0.0000001f;                                
	float r_x = 1.0 / s_x;

	                  
	float3 R_x = pc_p1;   
	R_x.x = R_x.x * r_x;
	R_x.y = R_x.y * r_x;
	R_x.z = R_x.z * r_x;

	float3 R_y = cross3D(normal3D, R_x);
	float s_y = fabsf(dot3D(pc_p2, R_y));                           
	           
	float3 sR_y = R_y;  
	sR_y.x = R_y.x * s_y;
	sR_y.y = R_y.y * s_y;
	sR_y.z = R_y.z * s_y;

	                      
	glm::mat3 L = glm::mat3(
		pc_p1.x, pc_p1.y, pc_p1.z, 
		sR_y.x, sR_y.y, sR_y.z,  
		normal3D.x, normal3D.y, normal3D.z
	);

	/**************** calculate p_tangent *****************/
	          
	p1_tangent.x = 1.0;
	p1_tangent.y = 0.0;

	p2_tangent.x = dot3D(pc_p2, R_x) * r_x;                       
	p2_tangent.y = 1.0;
  
	p3_tangent.x = -1 - p2_tangent.x;           // p2_x + p3_x = -1
	p3_tangent.y = -1.0;

	                                           
	/******************************************************/


	/************************ calculate T **********************/
	// center of Gaussians in the camera coordinate
	glm::mat3x4 splat2world = glm::mat3x4(
		glm::vec4(L[0], 0.0),
		glm::vec4(L[1], 0.0),
		glm::vec4(center_convex.x, center_convex.y, center_convex.z, 1)                  
	);

	glm::mat4 world2ndc = glm::mat4(
		projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
		projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
		projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
		projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
	);

	glm::mat3x4 ndc2pix = glm::mat3x4(
		glm::vec4(float(W) / 2.0, 0.0, 0.0, float(W-1) / 2.0),
		glm::vec4(0.0, float(H) / 2.0, 0.0, float(H-1) / 2.0),
		glm::vec4(0.0, 0.0, 0.0, 1.0)
	);

	T = glm::transpose(splat2world) * world2ndc * ndc2pix;
	normal = transformVec4x3({normal3D.x, normal3D.y, normal3D.z}, viewmatrix);     

}


template<int C>
__global__ void subpreprocessCUDA(
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
)
{
	auto idx = cg::this_grid().thread_rank();                            
	if (idx >= sub_triangle_total_num)
		return;

	                          
	radii_2D[idx] = 0;
	tiles_touched[idx] = 0; 

	const int current_start_point_idx = 3 * idx;                     

    bool& curr_visibility = sub_triangle_visibility[idx];                          
	if(!curr_visibility)  return;                                      
 
	                                                                                      
	                                               
	bool check_visible_flag = false;                    
	const float visibility_depth_thr = 0.2f;    // org
	float3 center_triangle_3D = {0.0f, 0.0f, 0.0f};

	                        
	for (int i = 0; i < 3; i++) {
		center_triangle_3D.x += p_world[current_start_point_idx + i].x;       
		center_triangle_3D.y += p_world[current_start_point_idx + i].y;    
		center_triangle_3D.z += p_world[current_start_point_idx + i].z;   
		
		
		      
		float3 tmp_view;
		tmp_view = transformPoint4x3(p_world[current_start_point_idx + i], viewmatrix);
		check_visible_flag = (check_visible_flag || (tmp_view.z > visibility_depth_thr));                                 
	}
 
	center_triangle_3D.x /= 3.0;
	center_triangle_3D.y /= 3.0;
	center_triangle_3D.z /= 3.0;

	// Perform near culling, quit if outside.        
	float3 p_view_center_triangle;                         
	p_view_center_triangle = transformPoint4x3(center_triangle_3D, viewmatrix);

	check_visible_flag = (check_visible_flag || (p_view_center_triangle.z > visibility_depth_thr));                                  

	                                                 
	if (!check_visible_flag)
	{
		curr_visibility = false;           
		return;
	}  


	                                                                                      
	float3 curr_p_world = {0.0f, 0.0f, 0.0f};     
	float4 curr_p_hom = {0.0f, 0.0f, 0.0f, 0.0f};
	float3 curr_p_proj = {0.0f, 0.0f, 0.0f};

	float curr_p_w = 0.0f;

	float2 average_center_2D = {0.0f, 0.0f};                     



	for(int i = 0; i < 3; i++)
	{
		curr_p_world = p_world[current_start_point_idx + i];           
		curr_p_hom = transformPoint4x4(curr_p_world, projmatrix);
		curr_p_w = 1.0f / fabsf(curr_p_hom.w + 0.0000001f);       // test
		curr_p_proj = {curr_p_hom.x * curr_p_w, curr_p_hom.y * curr_p_w, curr_p_hom.z * curr_p_w };
		p_image[current_start_point_idx + i] = { ndc2Pix(curr_p_proj.x, W), ndc2Pix(curr_p_proj.y, H) };	                         

		               
		average_center_2D.x += p_image[current_start_point_idx + i].x;       
		average_center_2D.y += p_image[current_start_point_idx + i].y;   
	}

	               
	average_center_2D.x /= 3.0;
	average_center_2D.y /= 3.0;
      
	float max_distance = 0.0;

	for (int i = 0; i < 3; i++) {
		
		float distance = sqrtf(
            (p_image[current_start_point_idx + i].x - average_center_2D.x) * (p_image[current_start_point_idx + i].x - average_center_2D.x) +
            (p_image[current_start_point_idx + i].y - average_center_2D.y) * (p_image[current_start_point_idx + i].y - average_center_2D.y)
        );

        // Update max_distance if the current distance is greater
        if (distance > max_distance) {
            max_distance = distance;
        }
    }
  
	if(max_distance > 10000.0f){
		curr_visibility = false;              
		return;
	}

	max_distance =  ceil(max_distance * 1.1f);
	uint2 rect_min_convex, rect_max_convex;
	getRect(average_center_2D, max_distance, rect_min_convex, rect_max_convex, grid);
	if ((rect_max_convex.x - rect_min_convex.x) * (rect_max_convex.y - rect_min_convex.y) == 0)
	{
		curr_visibility = false;           
		return;
	}
	
	                       
	depths[idx] = p_view_center_triangle.z;                                              
	tiles_touched[idx] = (rect_max_convex.y - rect_min_convex.y) * (rect_max_convex.x - rect_min_convex.x);                           
	radii_2D[idx] = max_distance;                                   
	center_means2D[idx] = average_center_2D;                        
}



// Perform initial steps for each Convex prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
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
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
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
	float* rgb,
	float4* conic_opacity,
	float* cov3Ds,
	const dim3 grid,
	uint32_t* tiles_touched,
	uint32_t* sub_triangle_father_index,    
	float3* sub_points_world,             
	bool*  sub_triangle_visibility,         
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();                              
	if (idx >= P)
		return;

	const int cumsum_for_convex = cumsum_of_points_per_convex[idx];                                                           
	const int offset = 3 * cumsum_for_convex;                                                            

	radii[idx] = 0;
	tiles_touched[idx] = 0;
	scaling[idx] = 0.0f;
	density_factor[idx] = 0.0f;                     

	
	float3 center_convex = {0.0f, 0.0f, 0.0f};
	for (int i = 0; i < num_points_per_convex[idx]; i++) {
		indices[cumsum_for_convex + i] = i;                       
		center_convex.x += convex_points[offset + 3 * i];       
		center_convex.y += convex_points[offset + 3 * i + 1];
		center_convex.z += convex_points[offset + 3 * i + 2];
	}
 
	center_convex.x /= num_points_per_convex[idx];
	center_convex.y /= num_points_per_convex[idx];
	center_convex.z /= num_points_per_convex[idx];



	// Perform near culling, quit if outside.
	float3 p_view_convex;                               
	p_view_convex = transformPoint4x3(center_convex, viewmatrix);


	float4 p_hom_center = transformPoint4x4(center_convex, projmatrix);
	float p_w_center = 1.0f / (p_hom_center.w + 0.0000001f);
	float3 center_convex_camera_view = { p_hom_center.x * p_w_center, p_hom_center.y * p_w_center, p_hom_center.z * p_w_center };
	float2 center_convex_2D = { ndc2Pix(center_convex_camera_view.x, W), ndc2Pix(center_convex_camera_view.y, H) };
	        
	// Compute transformation matrix            
	const int convex_idx = 3 * cumsum_for_convex;
	float3 convex_p1 = {convex_points[convex_idx + 3 * 0], convex_points[convex_idx + 3 * 0 + 1], convex_points[convex_idx + 3 * 0 + 2]};
	float3 convex_p2 = {convex_points[convex_idx + 3 * 1], convex_points[convex_idx + 3 * 1 + 1], convex_points[convex_idx + 3 * 1 + 2]};
	float3 convex_p3 = {convex_points[convex_idx + 3 * 2], convex_points[convex_idx + 3 * 2 + 1], convex_points[convex_idx + 3 * 2 + 2]};

	glm::mat3 T;
	float3 normal3D;
	float2 p1_tangent;                       
    float2 p2_tangent;                     
    float2 p3_tangent;                         
	compute_transmat(center_convex, convex_p1, convex_p2, convex_p3, projmatrix, viewmatrix, W, H, T, normal3D, p1_tangent, p2_tangent, p3_tangent);
	float3 *T_ptr = (float3*)transMats;
	T_ptr[idx * 3 + 0] = {T[0][0], T[0][1], T[0][2]};
	T_ptr[idx * 3 + 1] = {T[1][0], T[1][1], T[1][2]};
	T_ptr[idx * 3 + 2] = {T[2][0], T[2][1], T[2][2]};


	        
#if DUAL_VISIABLE
	float cos = -sumf3(p_view_convex * normal3D);
	if (cos == 0) return;              
	float multiplier = cos > 0 ? 1: -1;
	normal3D = multiplier * normal3D;
#endif
             
	p_image[cumsum_for_convex + 0] = p1_tangent;                 
	p_image[cumsum_for_convex + 1] = p2_tangent;                 
	p_image[cumsum_for_convex + 2] = p3_tangent;                     

	/**********************************************************************/
	// Calculation of points in 2D image space 
	float cov3D[6] = {0.0f};                
	float2 average_center_2D = {0.0f, 0.0f};                     


	for (int i = 0; i < num_points_per_convex[idx]; i++) {
		float3 convex_point = {convex_points[offset + 3 * i], convex_points[offset + 3 * i + 1], convex_points[offset + 3 * i + 2]};
		p_hom[cumsum_for_convex + i] = transformPoint4x4(convex_point, projmatrix);
		p_w[cumsum_for_convex + i] = 1.0f / (p_hom[cumsum_for_convex + i].w + 0.0000001f);
		p_proj[cumsum_for_convex + i] = { p_hom[cumsum_for_convex + i].x * p_w[cumsum_for_convex + i], p_hom[cumsum_for_convex + i].y * p_w[cumsum_for_convex + i], p_hom[cumsum_for_convex + i].z * p_w[cumsum_for_convex + i] };        
		average_center_2D.x += p_image[cumsum_for_convex + i].x;       
		average_center_2D.y += p_image[cumsum_for_convex + i].y;   

		float3 diff = {
			convex_point.x - center_convex.x,
			convex_point.y - center_convex.y,
			convex_point.z - center_convex.z
		};

		cov3D[0] += diff.x * diff.x;
		cov3D[1] += diff.x * diff.y;
		cov3D[2] += diff.x * diff.z;
		cov3D[3] += diff.y * diff.y;
		cov3D[4] += diff.y * diff.z;
		cov3D[5] += diff.z * diff.z;
	}
             
	average_center_2D.x /= 3.0;
	average_center_2D.y /= 3.0;
	center_convex_2D = average_center_2D;

	// normalize the covariance matrix
	cov3D[0] /= num_points_per_convex[idx];
	cov3D[1] /= num_points_per_convex[idx];
	cov3D[2] /= num_points_per_convex[idx];
	cov3D[3] /= num_points_per_convex[idx];
	cov3D[4] /= num_points_per_convex[idx];
	cov3D[5] /= num_points_per_convex[idx];

	cov3Ds[6 * idx] = cov3D[0];
	cov3Ds[6 * idx + 1] = cov3D[1];
	cov3Ds[6 * idx + 2] = cov3D[2];
	cov3Ds[6 * idx + 3] = cov3D[3];
	cov3Ds[6 * idx + 4] = cov3D[4];
	cov3Ds[6 * idx + 5] = cov3D[5];

	scaling[idx] = max(sqrtf(cov3D[0]), max(sqrtf(cov3D[3]), sqrtf(cov3D[5])));                                    
             
	float max_distance = sqrtf(
            (p_image[cumsum_for_convex].x - center_convex_2D.x) * (p_image[cumsum_for_convex].x - center_convex_2D.x) +
            (p_image[cumsum_for_convex].y - center_convex_2D.y) * (p_image[cumsum_for_convex].y - center_convex_2D.y)
        );
	                                                         
    for (int i = 1; i < num_points_per_convex[idx]; i++) {
		
		float distance = sqrtf(
            (p_image[cumsum_for_convex + i].x - center_convex_2D.x) * (p_image[cumsum_for_convex + i].x - center_convex_2D.x) +
            (p_image[cumsum_for_convex + i].y - center_convex_2D.y) * (p_image[cumsum_for_convex + i].y - center_convex_2D.y)
        );

        // Update max_distance if the current distance is greater
        if (distance > max_distance) {
            max_distance = distance;
        }
    }
                      
	hull[2 * cumsum_for_convex + 0] = 0;
	hull[2 * cumsum_for_convex + (0 + 1)] = 1;

	              
	hull[2 * cumsum_for_convex + 1] = 1;
	hull[2 * cumsum_for_convex + (1 + 1)] = 2;

	              
	hull[2 * cumsum_for_convex + 2] = 2;
	hull[2 * cumsum_for_convex + (2 + 1)] = 0;

	                            
	for (int i = 0; i < 3; i++) {
		// Points forming the segment
		float2 p1_conv = p_image[cumsum_for_convex + i];
		float2 p2_conv = p_image[cumsum_for_convex + (i + 1) % 3];

		// Calculate the normal vector (90-degree counterclockwise rotation)
		float2 normal = { p2_conv.y - p1_conv.y, -(p2_conv.x - p1_conv.x)};

		// Calculate the offset (dot product of normal vector and point p1)
		float offset = - (normal.x * p1_conv.x + normal.y * p1_conv.y);

		                      
		normals[cumsum_for_convex + i] = normal;
    	offsets[cumsum_for_convex + i] = offset; 
	}

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3)(center_convex.x, center_convex.y, center_convex.z), *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	                       
	depths[idx] = p_view_convex.z;                                                       
	max_distance = ceil(max_distance * 1.1f);                
	radii[idx] = max_distance;
	points_xy_image[idx] = center_convex_2D;
	conic_opacity[idx] = {normal3D.x, normal3D.y, normal3D.z, opacities[idx]};                                      

	const int curr_num_split = num_split_per_convex[idx];
	const int curr_start_subtriangle_index = cumsum_of_subtriangles_per_convex[idx];
	const int curr_start_sub_point_index = 3 * curr_start_subtriangle_index;
	const int curr_point_offset = 3 * cumsum_for_convex;                     

	if(curr_num_split == 0)                 
	{
		sub_triangle_father_index[curr_start_subtriangle_index] = idx;                                                  
		sub_triangle_visibility[curr_start_subtriangle_index] = true;                  
		
		                          
		sub_points_world[curr_start_sub_point_index + 0] = {convex_points[curr_point_offset + 3 * 0], convex_points[curr_point_offset + 3 * 0 + 1], convex_points[curr_point_offset + 3 * 0 + 2]};
		sub_points_world[curr_start_sub_point_index + 1] = {convex_points[curr_point_offset + 3 * 1], convex_points[curr_point_offset + 3 * 1 + 1], convex_points[curr_point_offset + 3 * 1 + 2]};
		sub_points_world[curr_start_sub_point_index + 2] = {convex_points[curr_point_offset + 3 * 2], convex_points[curr_point_offset + 3 * 2 + 1], convex_points[curr_point_offset + 3 * 2 + 2]};

	} else {                          
		int curr_sub_triangle_num = 1;               
		float curr_sub_seg_num = 1.0;                
		for(int i = 0; i < curr_num_split; i++)
		{
			curr_sub_triangle_num = curr_sub_triangle_num * 4; 
			curr_sub_seg_num = curr_sub_seg_num * 2;      

		}

		for(int i = 0; i < curr_sub_triangle_num; i++)
		{
			sub_triangle_father_index[curr_start_subtriangle_index + i] = idx;                                                  
			sub_triangle_visibility[curr_start_subtriangle_index + i] = true;                  
		}

		        
		float3 father_p1 = {convex_points[curr_point_offset + 3 * 0], convex_points[curr_point_offset + 3 * 0 + 1], convex_points[curr_point_offset + 3 * 0 + 2]};
		float3 father_p2 = {convex_points[curr_point_offset + 3 * 1], convex_points[curr_point_offset + 3 * 1 + 1], convex_points[curr_point_offset + 3 * 1 + 2]};
		float3 father_p3 = {convex_points[curr_point_offset + 3 * 2], convex_points[curr_point_offset + 3 * 2 + 1], convex_points[curr_point_offset + 3 * 2 + 2]};
		

		float3 p2_p1 = father_p2 - father_p1;    // a
		float3 p3_p1 = father_p3 - father_p1;    // b
		float3 p3_p2 = father_p3 - father_p2;    // c

		float3 a_step = {p2_p1.x / curr_sub_seg_num, p2_p1.y / curr_sub_seg_num, p2_p1.z / curr_sub_seg_num};
		float3 b_step = {p3_p1.x / curr_sub_seg_num, p3_p1.y / curr_sub_seg_num, p3_p1.z / curr_sub_seg_num};
		float3 c_step = {p3_p2.x / curr_sub_seg_num, p3_p2.y / curr_sub_seg_num, p3_p2.z / curr_sub_seg_num};


		float3 level_p1 = father_p1;
		int save_step = curr_start_sub_point_index;

		for(int level_num = 0; level_num < curr_sub_seg_num; level_num++)
		{
			int forward_triangle_num = level_num + 1;
			int backward_triangle_num = level_num;

			float3 forward_p1 = level_p1;
			for(int i = 0; i < forward_triangle_num; i++)
			{
				sub_points_world[save_step] = forward_p1;
				sub_points_world[save_step + 1] = forward_p1 + a_step;
				sub_points_world[save_step + 2] = forward_p1 + b_step;
				
				forward_p1 = forward_p1 + c_step;
				save_step = save_step + 3;
			}

			float3 backward_p1 = level_p1;
			for(int j = 0; j < backward_triangle_num; j++)
			{
				sub_points_world[save_step] = backward_p1;
				sub_points_world[save_step + 1] = backward_p1 + b_step;
				sub_points_world[save_step + 2] = backward_p1 + c_step;

				backward_p1 = backward_p1 + c_step;
				save_step = save_step + 3;
			}

			level_p1 = level_p1 + a_step;

		}
	}

	/**************************************************************/

}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	int sub_triangle_total_num,              
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ normals,
	const float* __restrict__ offsets,
	const int* __restrict__ num_points_per_convex_view,
	const float2* __restrict__ p_tangent,                                                                    
	const float* __restrict__ delta,
	const float* __restrict__ sigma,
	const int* __restrict__ num_points_per_convex,
	const int* __restrict__ cumsum_of_points_per_convex,
	const float* __restrict__ features,
	const float* __restrict__ transMats,   
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ depths,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float*  __restrict__  sub_triangle_depths,                       
	const float2*  __restrict__ sub_triangle_p_image,                            
	const uint32_t*  __restrict__ sub_triangle_father_index,
	const bool* __restrict__ sub_triangle_visibility,                      
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_others,
	float* impact)                                                                               
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	const float distance_amplification_factor = 2000.0;                           

	__shared__ int collected_sub_triangle_id[BLOCK_SIZE];   
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	__shared__ float2 collected_normals[BLOCK_SIZE * MAX_NB_POINTS];
	__shared__ float collected_offsets[BLOCK_SIZE * MAX_NB_POINTS];
	__shared__ int collected_num_points_per_convex_view[BLOCK_SIZE];
	__shared__ float collected_delta[BLOCK_SIZE];
	__shared__ float collected_sigma[BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];
	__shared__ float2 collected_p_tangent[BLOCK_SIZE * 3];                                                                 

	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];

   __shared__ float collected_sub_triangle_depths[BLOCK_SIZE];                                      
   __shared__ uint32_t collected_father_id[BLOCK_SIZE];                                           

   __shared__ float2 collected_sub_triangle_p1_image[BLOCK_SIZE];                                         
   __shared__ float2 collected_sub_triangle_p2_image[BLOCK_SIZE];                                         
   __shared__ float2 collected_sub_triangle_p3_image[BLOCK_SIZE];                                         

	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	// render axutility ouput
	float N[3] = {0};
	float D = { 0 };
	float M1 = {0};
	float M2 = {0};
	float distortion = {0};
	float median_depth = {0};
	// float median_weight = {0};
	float median_contributor = {-1};

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_sub_triangle_id = point_list[range.x + progress];                      
			
			if(!sub_triangle_visibility[coll_sub_triangle_id])
				continue;
				
			collected_sub_triangle_id[block.thread_rank()] = coll_sub_triangle_id;            


			if(coll_sub_triangle_id > sub_triangle_total_num)    
			{
				printf("[ERROR] coll_sub_triangle_id ILLIGLE %d\n", coll_sub_triangle_id);
				continue;
			}   

			collected_father_id[block.thread_rank()] = sub_triangle_father_index[coll_sub_triangle_id];                     
			collected_sub_triangle_depths[block.thread_rank()] = sub_triangle_depths[coll_sub_triangle_id];                      

			collected_sub_triangle_p1_image[block.thread_rank()] = sub_triangle_p_image[3*coll_sub_triangle_id + 0];                           
			collected_sub_triangle_p2_image[block.thread_rank()] = sub_triangle_p_image[3*coll_sub_triangle_id + 1];                           
			collected_sub_triangle_p3_image[block.thread_rank()] = sub_triangle_p_image[3*coll_sub_triangle_id + 2];                                                                                           
			                             
			uint32_t coll_father_id = sub_triangle_father_index[coll_sub_triangle_id];                

			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_father_id];
			collected_num_points_per_convex_view[block.thread_rank()] = 3;   			                                          
			collected_delta[block.thread_rank()] = delta[coll_father_id];
			collected_sigma[block.thread_rank()] = sigma[coll_father_id];
			collected_depths[block.thread_rank()] = depths[coll_father_id];

			for (int k = 0; k < 3; k++) {
				collected_p_tangent[3 * block.thread_rank() + k] = p_tangent[cumsum_of_points_per_convex[coll_father_id] + k];        
			}

			                                         
			for (int k = 0; k < 3; k++) {
    			collected_normals[MAX_NB_POINTS * block.thread_rank() + k] = normals[cumsum_of_points_per_convex[coll_father_id] + k];
			}
			                                         
			for (int k = 0; k < 3; k++) {
    			collected_offsets[MAX_NB_POINTS * block.thread_rank() + k] = offsets[cumsum_of_points_per_convex[coll_father_id] + k];
			}

			// collect transMats
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_father_id+0], transMats[9 * coll_father_id+1], transMats[9 * coll_father_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_father_id+3], transMats[9 * coll_father_id+4], transMats[9 * coll_father_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_father_id+6], transMats[9 * coll_father_id+7], transMats[9 * coll_father_id+8]};
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;                     
			float2 sub_triangle_p1 = collected_sub_triangle_p1_image[j];
			float2 sub_triangle_p2 = collected_sub_triangle_p2_image[j];
			float2 sub_triangle_p3 = collected_sub_triangle_p3_image[j];

			float2 sub_edge_p1p2 = sub_triangle_p2 - sub_triangle_p1;
			float2 sub_edge_p2p3 = sub_triangle_p3 - sub_triangle_p2;
			float2 sub_edge_p3p1 = sub_triangle_p1 - sub_triangle_p3;
			
			float2 p1pix = pixf - sub_triangle_p1;
			float2 p2pix = pixf - sub_triangle_p2;
			float2 p3pix = pixf - sub_triangle_p3;


			       
			float cross1 = crossProduct(sub_edge_p1p2, p1pix);
			float cross2 = crossProduct(sub_edge_p2p3, p2pix);
			float cross3 = crossProduct(sub_edge_p3p1, p3pix);
		
			              
			bool has_neg = (cross1 < 0) || (cross2 < 0) || (cross3 < 0);
			bool has_pos = (cross1 > 0) || (cross2 > 0) || (cross3 > 0);
 
		
			bool inside_flag = !(has_neg && has_pos);                 


			if(!inside_flag)
				continue;                     

			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
			                             
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;

			float3 p = cross3D(k, l);
			if (p.z == 0.0) continue;
			float2 s = {p.x / p.z, p.y / p.z};          

			float depth = (s.x * Tw.x + s.y * Tw.y) + Tw.z;                    
			                                                           
	 
			if (depth < near_n) continue;

			float2 p1_tangent = collected_p_tangent[j * 3 + 0];
			float2 p2_tangent = collected_p_tangent[j * 3 + 1];
			float2 p3_tangent = collected_p_tangent[j * 3 + 2];

			float area = triangleArea(p1_tangent, p2_tangent, p3_tangent);                  
			              

			float near_thr = 0.01;
                     
			float d1_tangent = magnitude2D(s - p1_tangent);
			float d2_tangent = magnitude2D(s - p2_tangent);
			float d3_tangent = magnitude2D(s - p3_tangent);

			bool near_flag = (d1_tangent <= near_thr) || (d2_tangent <= near_thr) || (d3_tangent <= near_thr);

			float4 con_o = collected_conic_opacity[j];
			float normal[3] = {con_o.x, con_o.y, con_o.z};     
			float distances[MAX_NB_POINTS]; 
			float max_val = -INFINITY;
			for (int k = 0; k < collected_num_points_per_convex_view[j]; k++) {
				                                                                                                                                                                                                   
				distances[k] = collected_normals[j * MAX_NB_POINTS + k].x * s.x + collected_normals[j * MAX_NB_POINTS + k].y * s.y + collected_offsets[j * MAX_NB_POINTS + k];                           
				                   
				if(distances[k] > 0.0000001f) 
				{
					distances[k] = distances[k] - 0.025f;
				}
				distances[k] =  distance_amplification_factor * distances[k]; 
				if (distances[k] > max_val) {
					max_val = distances[k];
				}
			}

			float sum_exp = 0.0f;
			for (int k = 0; k < collected_num_points_per_convex_view[j]; k++) {
				sum_exp += expf(1.0 * collected_delta[j] * (distances[k]-max_val));
			}

			float phi_x = 1.0 * collected_delta[j] * max_val + logf(sum_exp);
			float Cx = 1.0f / (1.0f + expf(1.0 * collected_sigma[j] * phi_x));
			float alpha = min(0.99f, con_o.w * Cx);  

			if (alpha < 1.0f / 255.0f)
				continue;                                           

			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			float w = alpha * T;

			// Render depth distortion map
			// Efficient implementation of distortion loss, see 2DGS' paper appendix.
			float A = 1-T;
			float m = far_n / (far_n - near_n) * (1 - near_n / depth);
			distortion += (m * m * A + M2 - 2 * m * M1) * w;
			D  += depth * w;
			M1 += m * w;
			M2 += m * m * w;

			if (T > 0.5) {
				median_depth = depth;
				// median_weight = w;
				median_contributor = contributor;
			}
			// Render normal map
			for (int ch=0; ch<3; ch++) N[ch] += normal[ch] * w;

            for (int ch = 0; ch < CHANNELS; ch++)
                C[ch] += features[collected_father_id[j] * CHANNELS + ch] *  w ;    
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
			atomicAdd(&(impact[ collected_father_id[j] ]), w);

		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

		n_contrib[pix_id + H * W] = median_contributor;                                     
		final_T[pix_id + H * W] = M1;
		final_T[pix_id + 2 * H * W] = M2;
		out_others[pix_id + DEPTH_OFFSET * H * W] = D;
		out_others[pix_id + ALPHA_OFFSET * H * W] = 1 - T;
		for (int ch=0; ch<3; ch++) out_others[pix_id + (NORMAL_OFFSET+ch) * H * W] = N[ch];
		out_others[pix_id + MIDDEPTH_OFFSET * H * W] = median_depth;
		out_others[pix_id + DISTORTION_OFFSET * H * W] = distortion;
		
	}
}

void FORWARD::render(
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
	const float* colors,
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
	float* impact)                                                                 
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		sub_triangle_total_num,                  
		ranges,
		point_list,
		W, H,
		normals,
		offsets,
		num_points_per_convex_view,
		p_tangent,                                                                        
		delta,
		sigma,
		num_points_per_convex,
		cumsum_of_points_per_convex,
		colors,
		transMats,    
		conic_opacity,
		depths,
		final_T,
		n_contrib,
		sub_triangle_depths,                      
		sub_triangle_p_image,                           
		sub_triangle_father_index,      
		sub_triangle_visibility,                   
		bg_color,
		out_color,
		out_others,
		impact);                                                         
}

void FORWARD::preprocess(int P, int D, int M,
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
	float2* means2D,
	float* depths,
	float* transMats,                                        
	float* rgb,
	float4* conic_opacity,
	float* cov3Ds,
	const dim3 grid,
	uint32_t* tiles_touched,
	uint32_t* sub_triangle_father_index,     
	float3* sub_points_world,              
	bool*  sub_triangle_visibility,          
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		convex_points,
		delta,
		sigma,
		num_split_per_convex,      
		cumsum_of_subtriangles_per_convex,    
		num_points_per_convex,
		cumsum_of_points_per_convex,
		opacities,
		scaling,
		density_factor,
		shs,
		clamped,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		normals,
		offsets,
		num_points_per_convex_view,
		p_hom,
		p_w,
		p_proj,
		p_image,
		hull,
		indices,
		means2D,
		depths,
		transMats,                                           
		rgb,
		conic_opacity,
		cov3Ds,
		grid,
		tiles_touched,
		sub_triangle_father_index,    
		sub_points_world,             
		sub_triangle_visibility,     
		prefiltered
		);
}




void FORWARD::subpreprocess(
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
)
{
	subpreprocessCUDA<NUM_CHANNELS> << <(sub_triangle_total_num + 255) / 256, 256 >> > (
 		sub_triangle_total_num,
        p_world,                                
        sub_triangle_visibility,                                     
        depths,                                    
        p_image,                                       
        center_means2D,                             
		grid,
        tiles_touched,                                
        radii_2D,
		viewmatrix, 
        projmatrix,
	    W, H,
        focal_x, focal_y,
        tan_fovx, tan_fovy
		);
}