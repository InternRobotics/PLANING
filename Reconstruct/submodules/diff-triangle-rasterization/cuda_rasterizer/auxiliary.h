#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)


 
/************************ ADD FOR 2DGS ***********************/

#define DEPTH_OFFSET 0
#define ALPHA_OFFSET 1
#define NORMAL_OFFSET 2 
#define MIDDEPTH_OFFSET 5
#define DISTORTION_OFFSET 6
// #define MEDIAN_WEIGHT_OFFSET 7

// distortion helper macros
#define BACKFACE_CULL 1
#define DUAL_VISIABLE 1
// #define NEAR_PLANE 0.2
// #define FAR_PLANE 100.0
#define DETACH_WEIGHT 0
/*************************************************************/


__device__ const float near_n = 0.2;
__device__ const float far_n = 100.0;
__device__ const float FilterSize = 0.707106; // sqrt(2) / 2
__device__ const float FilterInvSquare = 2.0f;

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	                                                         
	                                                                              
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}



/*********************************************************************/
// from 2DGS operators
__forceinline__ __device__ float3 cross3D(float3 a, float3 b){return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);}
__forceinline__ __device__ float dot3D(float3 a, float3 b){return a.x*b.x + a.y*b.y + a.z*b.z;}    




             
__forceinline__ __device__ float magnitude3D(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}


             
__forceinline__ __device__ float magnitude2D(float2 v) {
    return sqrtf(v.x * v.x + v.y * v.y);
}



                
__forceinline__ __device__ float triangleArea(float3 a, float3 b, float3 c) {
    float3 ab = make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
    float3 ac = make_float3(c.x - a.x, c.y - a.y, c.z - a.z);
    float3 cross = cross3D(ab, ac);
    return 0.5f * magnitude3D(cross);
}







             
__forceinline__ __device__ float crossProduct(float2 u, float2 v){return u.x * v.y - u.y * v.x;}




                    
      
__forceinline__ __device__ float triangleArea(float2 a, float2 b, float2 c) {
    float2 ab = make_float2(b.x - a.x, b.y - a.y);
    float2 ac = make_float2(c.x - a.x, c.y - a.y);
    float cross = crossProduct(ab, ac);
    return 0.5f * fabsf(cross);
}


__forceinline__ __device__ float3 operator*(float3 a, float3 b){return make_float3(a.x * b.x, a.y * b.y, a.z*b.z);}

__forceinline__ __device__ float2 operator*(float2 a, float2 b){return make_float2(a.x * b.x, a.y * b.y);}

__forceinline__ __device__ float3 operator*(float f, float3 a){return make_float3(f * a.x, f * a.y, f * a.z);}

__forceinline__ __device__ float2 operator*(float f, float2 a){return make_float2(f * a.x, f * a.y);}

__forceinline__ __device__ float3 operator-(float3 a, float3 b){return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);}


__forceinline__ __device__ float3 operator+(float3 a, float3 b){return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);}


__forceinline__ __device__ float2 operator-(float2 a, float2 b){return make_float2(a.x - b.x, a.y - b.y);}

__forceinline__ __device__ float sumf3(float3 a){return a.x + a.y + a.z;}

__forceinline__ __device__ float sumf2(float2 a){return a.x + a.y;}

__forceinline__ __device__ float3 sqrtf3(float3 a){return make_float3(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z));}

__forceinline__ __device__ float2 sqrtf2(float2 a){return make_float2(sqrtf(a.x), sqrtf(a.y));}

__forceinline__ __device__ float3 minf3(float f, float3 a){return make_float3(min(f, a.x), min(f, a.y), min(f, a.z));}

__forceinline__ __device__ float2 minf2(float f, float2 a){return make_float2(min(f, a.x), min(f, a.y));}

__forceinline__ __device__ float3 maxf3(float f, float3 a){return make_float3(max(f, a.x), max(f, a.y), max(f, a.z));}

__forceinline__ __device__ float2 maxf2(float f, float2 a){return make_float2(max(f, a.x), max(f, a.y));}

/*********************************************************************/
 
// Helper function to compute cross product of two vectors OA and OB
// A positive cross product indicates a counterclockwise turn, 
// a negative cross product indicates a clockwise turn,
// and a zero cross product indicates the points are collinear.
__forceinline__ __device__ float crossProduct(const float2& O, const float2& A, const float2& B) {
    return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}

__forceinline__ __device__ bool in_frustum_convex(int idx,
	const float3 p_orig,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	                                                                 
	                                                                              
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)
	{ 
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}


inline __device__ glm::mat3
scale_to_mat(const glm::vec2 scale, const float glob_scale) {
	glm::mat3 S = glm::mat3(1.f);
	S[0][0] = glob_scale * scale.x;
	S[1][1] = glob_scale * scale.y;
	return S;
}
                                                             
inline __device__ void grad_cross3D(const float3& dL_dc, const float3& a, const float3& b, float3& dL_da, float3& dL_db)
{
	dL_da = cross3D(b, dL_dc);             
    dL_db = cross3D(dL_dc, a);             
}
                   
inline __device__ void grad_normalize(const float3& dL_dn, const float& r, const float3& n, float3& dL_dn0)
{                                   
	const float proj = n.x * dL_dn.x + n.y * dL_dn.y + n.z * dL_dn.z;                               
	dL_dn0.x = (dL_dn.x - proj * n.x) * r;
	dL_dn0.y = (dL_dn.y - proj * n.y) * r;
	dL_dn0.z = (dL_dn.z - proj * n.z) * r;
}
                                                      
inline __device__ void grad_dot3D(const float& dL_dm, const float3& a, const float3& b, float3& dL_da, float3& dL_db)
{
	                  
	dL_da.x = dL_dm * b.x;
	dL_da.y = dL_dm * b.y;
	dL_da.z = dL_dm * b.z;

	                  
	dL_db.x = dL_dm * a.x;
	dL_db.y = dL_dm * a.y;
	dL_db.z = dL_dm * a.z;
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif