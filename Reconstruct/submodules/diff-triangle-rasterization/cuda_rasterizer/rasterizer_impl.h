#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
		// org 3DGS	Need
		size_t scan_size;
		float* depths;
		char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* means2D;
		float4* conic_opacity;                                 
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		// Convex Spltting Need
		float* cov3D;

		int* num_points_per_convex_view;

		float4* p_hom;
		float* p_w;
		float3* p_proj;

		float2* p_image;                  
		int* hull;                        
		int* indices;                     

		float* offsets;
		float2* normals;

		// 2DGS Need
		
		float* transMat;

		static GeometryState fromChunk(char*& chunk, size_t P, size_t total_nb_points);
	};



	/********************* sub triangle ************************/
	// for sub triangle
	struct SubGeometryState
	{
		// org 3DGS	Need
		size_t scan_size;
		float* depths;                                          
		char* scanning_space;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;    // sub_triangle_num
		
		float3* p_world;                                                
		float2* p_image;                                                 
		uint32_t* father_index;                                            

		float2* center_means2D;                 
		
		bool* visibility;                             
		int* radii_2D;                        
		                               
		static SubGeometryState fromChunk(char*& chunk, size_t sub_triangle_num);
	};




	struct ImageState
	{
		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

    template <typename T>
    size_t required(size_t P, size_t total_nb_points = 0);

    // General case for states that don't need `total_nb_points`
	                         
    template<typename T> 
    size_t required(size_t P, size_t total_nb_points)
    {
        char* size = nullptr;
        T::fromChunk(size, P);
        return ((size_t)size) + 128;
    }

    // Specialization for GeometryState
    template<> 
    size_t required<GeometryState>(size_t P, size_t total_nb_points)
    {
        char* size = nullptr;
        GeometryState::fromChunk(size, P, total_nb_points);
        return ((size_t)size) + 128;
    }
};