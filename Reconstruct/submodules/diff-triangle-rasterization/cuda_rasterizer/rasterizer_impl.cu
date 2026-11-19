#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"


            
void findMaxTilesTouched(const uint32_t* d_tiles_touched, size_t size) {
             
    uint32_t* h_tiles_touched = new uint32_t[size];

                  
    cudaMemcpy(h_tiles_touched, d_tiles_touched, size * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    uint32_t max_tiles_touched = *std::max_element(h_tiles_touched, h_tiles_touched + size);

    delete[] h_tiles_touched;
}



            
void findMinTilesTouched(const uint32_t* d_tiles_touched, size_t size) {
             
    uint32_t* h_tiles_touched = new uint32_t[size];     
    cudaMemcpy(h_tiles_touched, d_tiles_touched, size * sizeof(uint32_t), cudaMemcpyDeviceToHost);  
    uint32_t min_tiles_touched = *std::min_element(h_tiles_touched, h_tiles_touched + size);      
    delete[] h_tiles_touched;
}

uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Convexes that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Convex / tile overlaps. 
// Run once per Convex (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* Convex_keys_unsorted,
	uint32_t* Convex_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Convexes
	if (radii[idx] > 0)
	{
		// Find this Convex's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Convex. Sorting the values 
		// with this key yields Convex IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				
				
				                            
				if(key > grid.x * grid.y)
				{
					printf("[ERROR] duplicateWithKeysERROR idx = %d, key = %u, radii[idx] = %d, points_xy = %f, %f\n",idx, key, radii[idx], points_xy[idx].x, points_xy[idx].y);
				}
					
					

				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);                          
 
				
				Convex_keys_unsorted[off] = key;
				Convex_values_unsorted[off] = idx;
				off++;


				                                                                
			}
		}
	}
}




// for check subtriangle writing
__global__ void checkSubTriangleWrite(
	int P,
	int sub_triangle_total_num,
	const int*  num_split_per_convex,     
	const int* cumsum_of_subtriangles_per_convex,   
	int* radii, 
	const float* convex_points,
	bool*  sub_triangle_visibility,   //  subGeometryState.visibility,
	uint32_t* sub_triangle_father_index,  // subGeometryState.father_index,
	float3* sub_points_world   // subGeometryState.p_world
)
{
	for(int current_father_tringle_index = 0; current_father_tringle_index < P; current_father_tringle_index++)
	{
		// printf("CHECKING %d th Father Triangle ....\n", current_father_tringle_index);
		
		int curr_num_split = num_split_per_convex[current_father_tringle_index];

		int curr_start_subtriangle_index = cumsum_of_subtriangles_per_convex[current_father_tringle_index];

		int curr_sub_triangle_num = 1;               
		float curr_sub_seg_num = 1.0;                
		for(int i = 0; i < curr_num_split; i++)
		{
			curr_sub_triangle_num = curr_sub_triangle_num * 4;
			curr_sub_seg_num = curr_sub_seg_num * 2;       
		}

		bool current_father_visiblity = radii[current_father_tringle_index] > 0.0001; 

		                
		// printf("CHECK Visibility Start!\n");
		for(int i = 0; i < curr_sub_triangle_num; i++)
		{
			bool current_sub_visibility = sub_triangle_visibility[curr_start_subtriangle_index + i];
			if(current_sub_visibility != current_father_visiblity){
				printf("[ERROR] Visibility! curr_subtriangle_index = %d\n",curr_start_subtriangle_index + i);
			}
		}
		// printf("CHECK Visibility DONE!\n");


		if(!current_father_visiblity) continue;
		// printf("Current Triangle is Visibile CONTINUE CHECKING.......\n");

		

		// printf("CHECK Father Index Start!\n");
		for(int i = 0; i < curr_sub_triangle_num; i++)
		{
			uint32_t current_sub_father_index = sub_triangle_father_index[curr_start_subtriangle_index + i];
			if(current_sub_father_index != current_father_tringle_index){
				printf("[ERROR] Father Index! curr_subtriangle_index = %d\n",curr_start_subtriangle_index + i);
			}
		}
		// printf("CHECK Father Index DONE!\n");



		// printf("CHECK p_world Start!\n");
		const int curr_point_offset = 9 * current_father_tringle_index;                     


		          
		float3 father_p1 = {convex_points[curr_point_offset + 3 * 0], convex_points[curr_point_offset + 3 * 0 + 1], convex_points[curr_point_offset + 3 * 0 + 2]};
		float3 father_p2 = {convex_points[curr_point_offset + 3 * 1], convex_points[curr_point_offset + 3 * 1 + 1], convex_points[curr_point_offset + 3 * 1 + 2]};
		float3 father_p3 = {convex_points[curr_point_offset + 3 * 2], convex_points[curr_point_offset + 3 * 2 + 1], convex_points[curr_point_offset + 3 * 2 + 2]};

		if(curr_num_split == 0)               
		{
			// printf("CHECK NO SPLITTED First p_world START!\n");

			float3 sub_first_p1 = sub_points_world[3*curr_start_subtriangle_index + 0];
			float3 sub_first_p2 = sub_points_world[3*curr_start_subtriangle_index + 1];
			float3 sub_first_p3 = sub_points_world[3*curr_start_subtriangle_index + 2];

			float3 delta1 =  father_p1 - sub_first_p1;                 // {father_p1.x - sub_first_p1.x, father_p1.y - sub_first_p1.y, father_p1.z - sub_first_p1.z};
			float3 delta2 =  father_p2 - sub_first_p2;                 // {father_p2.x - sub_first_p2.x, father_p2.y - sub_first_p2.y, father_p2.z - sub_first_p2.z};  
			float3 delta3 =  father_p3 - sub_first_p3;                // {father_p3.x - sub_first_p3.x, father_p3.y - sub_first_p3.y, father_p3.z - sub_first_p3.z};   

			float r1 = sqrtf(delta1.x * delta1.x + delta1.y * delta1.y + delta1.z * delta1.z);
			float r2 = sqrtf(delta2.x * delta2.x + delta2.y * delta2.y + delta2.z * delta2.z);
			float r3 = sqrtf(delta3.x * delta3.x + delta3.y * delta3.y + delta3.z * delta3.z);

			bool ok_flag = (r1 < 0.0001) && (r2 < 0.0001) && (r3 <0.0001);

			if(!ok_flag) 
				printf("[ERROR] First P_world! curr_subtriangle_index = %d\n", curr_start_subtriangle_index);

			// printf("CHECK NO SPLITTED First p_world DONE!\n");

		}else{
			// printf("CHECK SPLITTED First p_world START!\n");

			            
			float3 sub_first_p1 = sub_points_world[3*curr_start_subtriangle_index + 0];
			float3 delta1 =  father_p1 - sub_first_p1; 
			float r1 = sqrtf(delta1.x * delta1.x + delta1.y * delta1.y + delta1.z * delta1.z);
			bool ok_flag = (r1 < 0.0001);


			if(!ok_flag) 
				printf("[ERROR] SPLITTED  First P_world! curr_subtriangle_index = %d\n", curr_start_subtriangle_index);

			// printf("CHECK SPLITTED First p_world DONE!\n");

			// printf("CHECK SPLITTED Last p_world START!\n");

			float3 p2_p1 = father_p2 - father_p1;  // a       // {father_p2.x - father_p1.x, father_p2.y - father_p1.y, father_p2.z - father_p1.z};
			float3 p3_p1 = father_p3 - father_p1;  // b       // {father_p3.x - father_p1.x, father_p3.y - father_p1.y, father_p3.z - father_p1.z};
			float3 p3_p2 = father_p3 - father_p2;  // c       // {father_p2.x - father_p1.x, father_p2.y - father_p1.y, father_p2.z - father_p1.z};
 	
			float3 a_step = {p2_p1.x / curr_sub_seg_num, p2_p1.y / curr_sub_seg_num, p2_p1.z / curr_sub_seg_num};
			float3 b_step = {p3_p1.x / curr_sub_seg_num, p3_p1.y / curr_sub_seg_num, p3_p1.z / curr_sub_seg_num};
			float3 c_step = {p3_p2.x / curr_sub_seg_num, p3_p2.y / curr_sub_seg_num, p3_p2.z / curr_sub_seg_num};
	
	 
			float3 sub_last_p1 = sub_points_world[3*curr_start_subtriangle_index + 3*curr_sub_triangle_num - 3 + 0];
			float3 sub_last_p2 = sub_points_world[3*curr_start_subtriangle_index + 3*curr_sub_triangle_num - 3 + 1];
			float3 sub_last_p3 = sub_points_world[3*curr_start_subtriangle_index + 3*curr_sub_triangle_num - 3 + 2];

			float3 check1_p3 = sub_last_p1 + c_step + b_step;    // {sub_last_p1.x + c_step.x + b_step.x, sub_last_p1.y + c_step.y + b_step.y, sub_last_p1.z + c_step.z + b_step.z}; 
			float3 check2_p3 = sub_last_p2 + c_step;             // {sub_last_p2.x + c_step.x, sub_last_p2.y + c_step.y, sub_last_p2.z + c_step.z};
			float3 check3_p3 = sub_last_p3 + b_step;             // {sub_last_p3.x + b_step.x, sub_last_p3.y + b_step.y, sub_last_p3.z + b_step.z};  

			float3 check_delta1 = father_p3 - check1_p3; // {father_p3.x - check1_p3.x, father_p3.y - check1_p3.y, father_p3.z - check1_p3.z};  
			float3 check_delta2 = father_p3 - check2_p3; // {father_p3.x - check2_p3.x, father_p3.y - check2_p3.y, father_p3.z - check2_p3.z}; 
			float3 check_delta3 = father_p3 - check3_p3; // {father_p3.x - check3_p3.x, father_p3.y - check3_p3.y, father_p3.z - check3_p3.z};  

			float check_r1 = sqrtf(check_delta1.x * check_delta1.x + check_delta1.y * check_delta1.y + check_delta1.z * check_delta1.z);
			float check_r2 = sqrtf(check_delta2.x * check_delta2.x + check_delta2.y * check_delta2.y + check_delta2.z * check_delta2.z);
			float check_r3 = sqrtf(check_delta3.x * check_delta3.x + check_delta3.y * check_delta3.y + check_delta3.z * check_delta3.z);
		
		
		
			bool ok_flag2 = (check_r1 < 0.0001) && (check_r2 < 0.0001) && (check_r3 <0.0001);

			if(!ok_flag2) 
				printf("[ERROR] SPLITTED Last P_world! curr_subtriangle_index = %d\n", curr_start_subtriangle_index);
			
			// printf("CHECK SPLITTED Last p_world DONE!\n");

		}
		// printf("CHECK p_world DONE!\n");

	}
	      
	           
	for(int current_sub_index = 0; current_sub_index < sub_triangle_total_num; current_sub_index++)
	{
		float3 p1 = sub_points_world[3 * current_sub_index + 0];
        float3 p2 = sub_points_world[3 * current_sub_index + 1];
        float3 p3 = sub_points_world[3 * current_sub_index + 2];


		float area = triangleArea(p1, p2, p3);
 

	}

}


// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Convex ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;

		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Convexes as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}




                                       
CudaRasterizer::SubGeometryState CudaRasterizer::SubGeometryState::fromChunk(char*& chunk, size_t sub_triangle_num)
{
	SubGeometryState subgeom;
	obtain(chunk, subgeom.depths, sub_triangle_num, 128);      
	obtain(chunk, subgeom.tiles_touched, sub_triangle_num, 128);
	cub::DeviceScan::InclusiveSum(nullptr, subgeom.scan_size, subgeom.tiles_touched, subgeom.tiles_touched, sub_triangle_num);
	obtain(chunk, subgeom.scanning_space, subgeom.scan_size, 128);
	obtain(chunk, subgeom.point_offsets, sub_triangle_num, 128);
	obtain(chunk, subgeom.p_world, sub_triangle_num*3, 128);                          
	obtain(chunk, subgeom.p_image, sub_triangle_num*3, 128);                          

	obtain(chunk, subgeom.father_index, sub_triangle_num, 128);                       
	obtain(chunk, subgeom.center_means2D, sub_triangle_num, 128);                            
	obtain(chunk, subgeom.visibility, sub_triangle_num, 128);
	
	obtain(chunk, subgeom.radii_2D, sub_triangle_num, 128);                     

	return subgeom;
}



CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P, size_t total_nb_points)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);                      
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	obtain(chunk, geom.p_image, total_nb_points, 128);
	obtain(chunk, geom.hull, total_nb_points * 2, 128);
	obtain(chunk, geom.indices, total_nb_points, 128);
	obtain(chunk, geom.num_points_per_convex_view, P, 128);
	obtain(chunk, geom.offsets, total_nb_points, 128);
	obtain(chunk, geom.normals, total_nb_points, 128);
	obtain(chunk, geom.p_hom, total_nb_points, 128);
	obtain(chunk, geom.p_w, total_nb_points, 128);
	obtain(chunk, geom.p_proj, total_nb_points, 128);

	obtain(chunk, geom.transMat, P * 9, 128);    // Add for 2DGS transMat   // 

	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N * 3, 128);
	obtain(chunk, img.n_contrib, N * 2, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Convexes.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> subGeometryBuffer,     // for sub triangle
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* convex_points,
	const float* delta,
	const float* sigma,
	const int* num_split_per_convex,     // for sub triangle
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
	int* radii,
	float* impact,                                            
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P, total_nb_points);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P, total_nb_points);


	/************ for sub triangle ***********/
	
	                      
	                                                                   
	size_t subgeometry_chunk_size = required<SubGeometryState>(sub_triangle_total_num);
	char* subgeometry_chunkptr = subGeometryBuffer(subgeometry_chunk_size);
	SubGeometryState subGeometryState = SubGeometryState::fromChunk(subgeometry_chunkptr, sub_triangle_total_num);
	                                          
	CHECK_CUDA(cudaMemset(subGeometryState.visibility, 0, sub_triangle_total_num * sizeof(bool)),debug);
	
	// printf("sub_triangle_total_num = %d\n", sub_triangle_total_num);
	// printf("subgeometry_chunk_size = %d\n", subgeometry_chunk_size);

	/*************************************************/

	
	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Convex colors!");
	}


	// Run preprocessing per-Convex (transformation, bounding, conversion of SHs to RGB)
	                                     
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		convex_points,
		delta,
		sigma,
		num_split_per_convex,     // for sub triangle
		cumsum_of_subtriangles_per_convex,    // for sub triangle
		num_points_per_convex,
		cumsum_of_points_per_convex,
		opacities,
		scaling,
		density_factor,
		shs,
		geomState.clamped,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.normals,
		geomState.offsets,
		geomState.num_points_per_convex_view,
		geomState.p_hom,
		geomState.p_w,
		geomState.p_proj,
		geomState.p_image,
		geomState.hull,
		geomState.indices,
		geomState.means2D,
		geomState.depths,
		geomState.transMat,                                        
		geomState.rgb,
		geomState.conic_opacity,
		geomState.cov3D,
		tile_grid,
		geomState.tiles_touched,
		subGeometryState.father_index,    // for preprocess sub triangle
		subGeometryState.p_world,          // for preprocess sub triangle
		subGeometryState.visibility,      // for preprocess sub triangle
		prefiltered
	), debug)

	           
	CHECK_CUDA(FORWARD::subpreprocess(
		sub_triangle_total_num,                         
		subGeometryState.p_world,                          
		subGeometryState.visibility,                                              
		subGeometryState.depths,                              
		subGeometryState.p_image,                                 
		subGeometryState.center_means2D,                       
		tile_grid,
		subGeometryState.tiles_touched,                          
		subGeometryState.radii_2D,                             
		viewmatrix, projmatrix,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy
	), debug)
                       
	// sub triangle
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(subGeometryState.scanning_space, subGeometryState.scan_size, subGeometryState.tiles_touched, subGeometryState.point_offsets, sub_triangle_total_num), debug)

	// Retrieve total number of Convex instances to launch and resize aux buffers
	int num_rendered;
	// sub triangle
	CHECK_CUDA(cudaMemcpy(&num_rendered, subGeometryState.point_offsets + sub_triangle_total_num - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);



	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Convex indices to be sorted
	// sub triangle
	duplicateWithKeys << <(sub_triangle_total_num + 255) / 256, 256 >> > (
		sub_triangle_total_num,
		subGeometryState.center_means2D,
		subGeometryState.depths,
		subGeometryState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		subGeometryState.radii_2D,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Convex indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

 
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);
 
	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Convexes independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	const float* transMat_ptr = geomState.transMat;    // for 2DGS   
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		sub_triangle_total_num,                      
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.normals,
		geomState.offsets,
		geomState.num_points_per_convex_view,
		geomState.p_image,                                
		delta,
		sigma,
		num_points_per_convex,
		cumsum_of_points_per_convex,
		feature_ptr,
		transMat_ptr,      
		geomState.conic_opacity,
		geomState.depths,
		imgState.accum_alpha,
		imgState.n_contrib,

		subGeometryState.depths, 
		subGeometryState.p_image,              
		subGeometryState.father_index,                   
		subGeometryState.visibility,                     

		background,
		out_color,
		out_others,
		impact), debug)                                              
	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
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
	const int sub_triangle_total_num,           // sub_triangle_total_num
	const float tan_fovx, float tan_fovy,
	const int* radii,                                                  
	char* geom_buffer,
	char* subGeom_buffer,      // for sub triangle
	char* binning_buffer,
	char* img_buffer,
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
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P, total_nb_points);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	// for sub triangle
	SubGeometryState subGeometryState = SubGeometryState::fromChunk(subGeom_buffer,sub_triangle_total_num);
	// printf("sub_triangle_total_num = %d\n", sub_triangle_total_num);
 
 
	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Convexes from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		sub_triangle_total_num,               
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		delta, 
		sigma,
		num_points_per_convex,
		cumsum_of_points_per_convex,
		geomState.normals,
		geomState.offsets,
		geomState.num_points_per_convex_view,
		geomState.transMat,             // from 2DGS  
		geomState.conic_opacity,
		geomState.depths,
		geomState.p_image,                                
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		subGeometryState.depths, 
		subGeometryState.p_image,              
		subGeometryState.father_index,                   
		subGeometryState.visibility,                     
		dL_dpix,
		dL_depths,                                    
		dL_ddelta,
		dL_dsigma,
        dL_dtransMat,                                 
        dL_da,                                        
        dL_dnormal3D,                                 
		(float3*)dL_dmeans2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		convex_points,
		width, height,
		radii,
		shs,
		geomState.clamped,
		viewmatrix,
		projmatrix,
		num_points_per_convex,
		cumsum_of_points_per_convex,
		geomState.p_hom,
		geomState.p_w,
		geomState.p_proj,
		geomState.p_image,
		geomState.hull,
		geomState.indices,
		geomState.num_points_per_convex_view,
		geomState.cov3D,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(glm::vec3*)dL_dconvex,
		dL_dtransMat,                                 
        dL_da,                                        
        dL_dnormal3D,                                 
		(glm::vec3*)dL_dmeans3D,
		(float3*)dL_dmeans2D,
		dL_dconic,
		dL_dcov3D,
		dL_dcolor,
		dL_dsh
		), debug)
}