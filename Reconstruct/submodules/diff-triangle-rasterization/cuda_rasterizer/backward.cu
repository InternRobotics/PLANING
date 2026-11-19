#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3 means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dconvex, glm::vec3* dL_dshs, const int cumsum_for_convex, const int num_points_per_convex)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means;
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Convex to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });
	glm::vec3 scaled_dL_dmean = glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z) / static_cast<float>(num_points_per_convex);

	for (int i = 0; i < num_points_per_convex; i++)
	{
		dL_dconvex[cumsum_for_convex + i] += scaled_dL_dmean;
	}

}
	                           
__device__ void compute_transmat_gradient(
	int idx,
	const float3& center_convex,
	const float3& convex_p1,
	const float3& convex_p2,
	const float3& convex_p3,
	const float* projmatrix,
	const float* viewmatrix,
	const int W,
	const int H, 
	const float3* dL_dnormal3Ds,                          
	const float* dL_da,                    
	const float* dL_dtransMats,            
	float3& dL_dconvex1,                              
	float3& dL_dconvex2,                              
	float3& dL_dconvex3                               
)
{
	                  
	glm::mat3 T;                  
	float current_triangle_a;                

	glm::mat3x4 P;                                  
	float3 normal3D_camera;                         
                                
	// calculate p1p2 and p1p3
	// normal3D_world = p1_p2 x p1_p3
	const float3 p1_p2 = convex_p2 - convex_p1;
	const float3 p1_p3 = convex_p3 - convex_p1;
 
	const float3 normal3D_world = cross3D(p1_p2, p1_p3);            

	        
	const float s_normal3D = magnitude3D(normal3D_world) + 0.0000001f;              
	const float r_normal3D = 1.0 / s_normal3D;   
     
	// Rz = normal3D_world / ||normal3D_world||
	float3 Rz;
	Rz.x = normal3D_world.x * r_normal3D;
	Rz.y = normal3D_world.y * r_normal3D;
	Rz.z = normal3D_world.z * r_normal3D;
                                                          
	const float3 pc_p1 = convex_p1 - center_convex;
	const float3 pc_p2 = convex_p2 - center_convex;
	const float3 pc_p3 = convex_p3 - center_convex;

	const float3 sRx = pc_p1;            
	const float sx = magnitude3D(sRx) + 0.0000001f;                                
	const float rx = 1.0 / sx;

	                  
	float3 Rx = sRx;   
	Rx.x = Rx.x * rx;
	Rx.y = Rx.y * rx;
	Rx.z = Rx.z * rx;


	// Ry = Rz x Rx
	const float3 Ry = cross3D(Rz, Rx);
    const float sy = fabsf(dot3D(pc_p2, Ry));                         

	           
	// sRy = fabsf(pc_p2 · Ry) Ry
	float3 sRy = Ry;  
	sRy.x = Ry.x * sy;
	sRy.y = Ry.y * sy;
	sRy.z = Ry.z * sy;

	                   
	                         
	glm::mat3 L = glm::mat3(
		sRx.x, sRx.y, sRx.z, 
		sRy.x, sRy.y, sRy.z,  
		Rz.x, Rz.y, Rz.z
	);

	/**************** calculate a *****************/

	// a = (pc_p2 · Rx) / sx
	current_triangle_a = dot3D(pc_p2, Rx) * rx;                       

	/************************ calculate T **********************/
	// center of Gaussians in the camera coordinate
	glm::mat3x4 M = glm::mat3x4(
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

	P = world2ndc * ndc2pix;
	T = glm::transpose(M) * P;           
	normal3D_camera = transformVec4x3({Rz.x, Rz.y, Rz.z}, viewmatrix);                     
                                                        
	// Update gradients w.r.t. transformation matrix of the Gaussian
	glm::mat3 dL_dT = glm::mat3(
		dL_dtransMats[idx*9+0], dL_dtransMats[idx*9+1], dL_dtransMats[idx*9+2],
		dL_dtransMats[idx*9+3], dL_dtransMats[idx*9+4], dL_dtransMats[idx*9+5],
		dL_dtransMats[idx*9+6], dL_dtransMats[idx*9+7], dL_dtransMats[idx*9+8]
	);
 

	// Update gradients w.r.t. scaling, rotation, position of the Gaussian
	glm::mat3x4 dL_dM = P * glm::transpose(dL_dT);
	float3 dL_dtn = transformVec4x3Transpose(dL_dnormal3Ds[idx], viewmatrix);                       

	
	              
#if DUAL_VISIABLE
	float3 p_view = transformPoint4x3(center_convex, viewmatrix);
	float cos = -sumf3(p_view * normal3D_camera);
	float multiplier = cos > 0 ? 1: -1;                       
	dL_dtn = multiplier * dL_dtn;                                        
#endif
	                                                     
	float3 dL_dsRx  = {dL_dM[0][0], dL_dM[0][1], dL_dM[0][2]};
	float3 dL_dsRy  = {dL_dM[1][0], dL_dM[1][1], dL_dM[1][2]};
	float3 dL_dRz   = {dL_dtn.x,    dL_dtn.y,    dL_dtn.z};                  
	               
	// dL_dmeans[idx] = glm::vec3(dL_dM[2]);  // org
	float3 dL_dmean3D  = {dL_dM[2][0], dL_dM[2][1], dL_dM[2][2]};                    
	float dL_da_current = dL_da[idx];                                                            
	float3 dL_dRx   = {0.0, 0.0, 0.0};                     
	float3 dL_dpcp2   = {0.0, 0.0, 0.0};                

	const float dL_dsy = dot3D(dL_dsRy, Ry);
    float sign_sy = (dot3D(pc_p2, Ry) >= 0.0)? 1.0f : -1.0f;              

	               
	float3 dL_dpcp2_from_sy = {0.0};
	float3 dL_dRy_from_sy = {0.0};
	       
	grad_dot3D(dL_dsy, pc_p2, Ry, dL_dpcp2_from_sy, dL_dRy_from_sy);          
	dL_dpcp2_from_sy = sign_sy * dL_dpcp2_from_sy;
	dL_dRy_from_sy = sign_sy * dL_dRy_from_sy;

	                   
	float3 dL_dRy_from_sRy = dL_dsRy;
	dL_dRy_from_sRy = sy * dL_dRy_from_sRy;
 

	           
	float3 dL_dRy = dL_dRy_from_sy + dL_dRy_from_sRy;
                                                     
	float3 dL_dRz_from_sRy, dL_dRx_from_sRy;
	grad_cross3D(dL_dRy, Rz, Rx, dL_dRz_from_sRy, dL_dRx_from_sRy);


	                        
	dL_dRx = dL_dRx + dL_dRx_from_sRy;         // dL_dsRy --> dL_dRx
	dL_dpcp2 = dL_dpcp2 + dL_dpcp2_from_sy;      // dL_dsRy --> dL_dpcp2
	dL_dRz = dL_dRz + dL_dRz_from_sRy;         // dL_dsRy --> dL_dRz

	                                                                            
	// a = (pc_p2 · Rx) / sx
	// const float rx = 1.0 / sx;
	float dL_dsx_from_a = -(current_triangle_a * rx) * dL_da_current;
	float3 dsx_dsRx = Rx;
	float3 dL_dsRx_from_a = dL_dsx_from_a * dsx_dsRx;                         

	float3 dL_dRx_from_a, dL_dpcp2_from_a;
	float  dL_ddot = rx * dL_da_current;

	grad_dot3D(dL_ddot, pc_p2, Rx, dL_dpcp2_from_a, dL_dRx_from_a);

	                        
	dL_dRx = dL_dRx + dL_dRx_from_a;            // dL_da --> dL_dRx
	dL_dpcp2 = dL_dpcp2 + dL_dpcp2_from_a;      // dL_da --> dL_dpcp2
	dL_dsRx = dL_dsRx + dL_dsRx_from_a;         // dL_da --> dL_dsRx
	
	float3 dL_dsRx_from_Rx;
	grad_normalize(dL_dRx, rx, Rx, dL_dsRx_from_Rx);

	                              
	dL_dsRx = dL_dsRx + dL_dsRx_from_Rx;

	float3 dL_dpcp1 = dL_dsRx;                          
	float3 dL_dnormal3D_world;
	grad_normalize(dL_dRz, r_normal3D, Rz, dL_dnormal3D_world);

	float3 dL_dp1p2, dL_dp1p3;
	grad_cross3D(dL_dnormal3D_world, p1_p2, p1_p3, dL_dp1p2, dL_dp1p3);
              
	float3 dL_dpc = {
		-dL_dpcp1.x - dL_dpcp2.x + dL_dmean3D.x,
		-dL_dpcp1.y - dL_dpcp2.y + dL_dmean3D.y,
		-dL_dpcp1.z - dL_dpcp2.z + dL_dmean3D.z
	};
                 
	dL_dconvex1.x = -dL_dp1p2.x - dL_dp1p3.x + dL_dpcp1.x + dL_dpc.x/3.0;
	dL_dconvex1.y = -dL_dp1p2.y - dL_dp1p3.y + dL_dpcp1.y + dL_dpc.y/3.0;
	dL_dconvex1.z = -dL_dp1p2.z - dL_dp1p3.z + dL_dpcp1.z + dL_dpc.z/3.0;

	                                    
	dL_dconvex2.x = dL_dp1p2.x + dL_dpcp2.x + dL_dpc.x/3.0;
	dL_dconvex2.y = dL_dp1p2.y + dL_dpcp2.y + dL_dpc.y/3.0; 
	dL_dconvex2.z = dL_dp1p2.z + dL_dpcp2.z + dL_dpc.z/3.0;

	                         
	dL_dconvex3.x = dL_dp1p3.x + dL_dpc.x/3.0;
	dL_dconvex3.y = dL_dp1p3.y + dL_dpc.y/3.0;
	dL_dconvex3.z = dL_dp1p3.z + dL_dpc.z/3.0;

}

template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float* convex_points,
	int W, int H,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const float* viewmatrix,      
	const float* projmatrix,      
	const int* num_points_per_convex,
	const int* cumsum_of_points_per_convex,
	float4* p_hom,
	float* p_w,
	float3* p_proj,
	float2* p_image,
	int* hull,
	int* indices,
	int* num_points_per_convex_view,
	const glm::vec3* campos,
	glm::vec3* dL_dconvex,                                
	const float* dL_dtransMat,                                        
	const float* dL_da,                                               
	const float* dL_dnormal3D,                                        
	glm::vec3* dL_dmeans,      
	float3* dL_dmean2D,                                        
	float* dL_dcov3D,          
	float* dL_dcolor,  
	float* dL_dsh)                         
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;                                
 
	const int cumsum_for_convex = cumsum_of_points_per_convex[idx];    // 3 * idx
	const int offset = 3 * cumsum_for_convex;   // 9 * idx

	float3 convex_p1 = {convex_points[offset + 3 * 0], convex_points[offset + 3 * 0 + 1], convex_points[offset + 3 * 0 + 2]};
	float3 convex_p2 = {convex_points[offset + 3 * 1], convex_points[offset + 3 * 1 + 1], convex_points[offset + 3 * 1 + 2]};
	float3 convex_p3 = {convex_points[offset + 3 * 2], convex_points[offset + 3 * 2 + 1], convex_points[offset + 3 * 2 + 2]};
	float3 center_convex = {0.0f, 0.0f, 0.0f};
	center_convex.x = (convex_p1.x + convex_p2.x + convex_p3.x) / 3.0;
	center_convex.y = (convex_p1.y + convex_p2.y + convex_p3.y) / 3.0;
	center_convex.z = (convex_p1.z + convex_p2.z + convex_p3.z) / 3.0;
             
	float3 dL_dconvex1 = {0.0f, 0.0f, 0.0f};
	float3 dL_dconvex2 = {0.0f, 0.0f, 0.0f};
	float3 dL_dconvex3 = {0.0f, 0.0f, 0.0f};

	compute_transmat_gradient(
		idx,
		center_convex, 
		convex_p1, convex_p2, convex_p3,  
		projmatrix, viewmatrix, W, H, 
		(float3*)dL_dnormal3D,         
		dL_da,                         
		(dL_dtransMat),                
		dL_dconvex1,                   
		dL_dconvex2,                   
		dL_dconvex3                    
	);
	          
	// p1
	dL_dconvex[cumsum_for_convex + 0].x = dL_dconvex1.x;
	dL_dconvex[cumsum_for_convex + 0].y = dL_dconvex1.y;
	dL_dconvex[cumsum_for_convex + 0].z = dL_dconvex1.z;

	// p2
	dL_dconvex[cumsum_for_convex + 1].x = dL_dconvex2.x;
	dL_dconvex[cumsum_for_convex + 1].y = dL_dconvex2.y;
	dL_dconvex[cumsum_for_convex + 1].z = dL_dconvex2.z;

	// p3
	dL_dconvex[cumsum_for_convex + 2].x = dL_dconvex3.x;
	dL_dconvex[cumsum_for_convex + 2].y = dL_dconvex3.y;
	dL_dconvex[cumsum_for_convex + 2].z = dL_dconvex3.z;

	                                                                              
	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3)(center_convex.x, center_convex.y, center_convex.z), *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dconvex, (glm::vec3*)dL_dsh, cumsum_for_convex, num_points_per_convex[idx]);
	                          
	float3 p1p2 = convex_p2 - convex_p1; 
	float3 p2p3 = convex_p3 - convex_p2;
	float3 p3p1 = convex_p3 - convex_p2;

	         
	const float s_p1p2 = magnitude3D(p1p2) + 0.0000001f;                
	const float r_p1p2 = 1.0 / s_p1p2;
	p1p2 = r_p1p2 * p1p2;

	const float s_p2p3 = magnitude3D(p2p3) + 0.0000001f;                
	const float r_p2p3 = 1.0 / s_p2p3;
	p2p3 = r_p2p3 * p2p3;

	const float s_p3p1 = magnitude3D(p3p1) + 0.0000001f;                
	const float r_p3p1 = 1.0 / s_p3p1;
	p3p1 = r_p3p1 * p3p1;

	float3 delta_convex1 = -1.0 * dL_dconvex1;
	float3 delta_convex2 = -1.0 * dL_dconvex2;
	float3 delta_convex3 = -1.0 * dL_dconvex3;

	float delta_p1p2_from_p1 = dot3D(delta_convex1, p1p2);
	float delta_p1p2_from_p2 = dot3D(delta_convex2, p1p2);
	float delta_p1p2 = delta_p1p2_from_p2 - delta_p1p2_from_p1;

	float delta_p2p3_from_p2 = dot3D(delta_convex2, p2p3);
	float delta_p2p3_from_p3 = dot3D(delta_convex3, p2p3);	
	float delta_p2p3 = delta_p2p3_from_p3 - delta_p2p3_from_p2;

	float delta_p3p1_from_p3 = dot3D(delta_convex3, p3p1);
	float delta_p3p1_from_p1 = dot3D(delta_convex1, p3p1);
	float delta_p3p1 = delta_p3p1_from_p1 - delta_p3p1_from_p3;

	// Design means2D               
	dL_dmean2D[idx].x = delta_p1p2;                  
	dL_dmean2D[idx].y = delta_p2p3;                  
	dL_dmean2D[idx].z = delta_p3p1;                  

}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	int sub_triangle_total_num,               
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float* __restrict__ delta,
	const float* __restrict__ sigma,
	const int* __restrict__ num_points_per_convex,
	const int* __restrict__ cumsum_of_points_per_convex,
	const float2* __restrict__ normals,
	const float* __restrict__ offsets,
	const int* __restrict__ num_points_per_convex_view,
	const float* __restrict__ transMats,   
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ depths,
	const float2* __restrict__ p_tangent,                                                  
	const float* __restrict__ colors,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float*  __restrict__  sub_triangle_depths,                       
	const float2*  __restrict__ sub_triangle_p_image,                            
	const uint32_t*  __restrict__ sub_triangle_father_index,    
	const bool* __restrict__ sub_triangle_visibility,                       
	const float* __restrict__ dL_dpixels,                                                            
	const float* __restrict__ dL_depths,                                                                            
	float* __restrict__ dL_ddelta,
	float* __restrict__ dL_dsigma,
	float* __restrict__ dL_dtransMat,                                 
	float* __restrict__ dL_da,                                        
	float* __restrict__ dL_dnormal3D,                                 
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors)
{
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;


	const float distance_amplification_factor = 2000.0;                                     

	__shared__ int collected_sub_triangle_id[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];

	__shared__ float2 collected_normals[BLOCK_SIZE * MAX_NB_POINTS];
	__shared__ float collected_offsets[BLOCK_SIZE * MAX_NB_POINTS];
	__shared__ int collected_num_points_per_convex_view[BLOCK_SIZE];
	__shared__ int collected_cumsum_of_points_per_convex[BLOCK_SIZE];
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

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Convex is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];                                

	float dL_dreg;
	float dL_ddepth;
	float dL_daccum;
	float dL_dnormal2D[3];
	const int median_contributor = inside ? n_contrib[pix_id + H * W] : 0;                                                                                
	float dL_dmedian_depth;
	float dL_dmax_dweight;

	if (inside) {
		dL_ddepth = dL_depths[DEPTH_OFFSET * H * W + pix_id];                      
		dL_daccum = dL_depths[ALPHA_OFFSET * H * W + pix_id];                        
		dL_dreg = dL_depths[DISTORTION_OFFSET * H * W + pix_id];
		for (int i = 0; i < 3; i++) 
			dL_dnormal2D[i] = dL_depths[(NORMAL_OFFSET + i) * H * W + pix_id];

		dL_dmedian_depth = dL_depths[MIDDEPTH_OFFSET * H * W + pix_id];
	}

	// for compute gradient with respect to depth and normal
	float last_depth = 0;       
	float last_normal[3] = { 0 };
	float accum_depth_rec = 0;                   
	float accum_alpha_rec = 0;                   
	float accum_normal_rec[3] = {0};                  
	// for compute gradient with respect to the distortion map
	const float final_D = inside ? final_Ts[pix_id + H * W] : 0;                        
	const float final_D2 = inside ? final_Ts[pix_id + 2 * H * W] : 0;
	const float final_A = 1 - T_final;
	float last_dL_dT = 0;
                                                           
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

	float last_alpha = 0;
	float last_color[C] = { 0 };

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{

		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_sub_triangle_id = point_list[range.y - progress - 1];                                        

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
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_father_id * C + i];
			collected_num_points_per_convex_view[block.thread_rank()] = 3;   			                                          
			collected_cumsum_of_points_per_convex[block.thread_rank()] = cumsum_of_points_per_convex[coll_father_id];
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

		// Iterate over Convexes
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{

			contributor--;
			if (contributor >= last_contributor)
				continue;
              
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

			// Fisrt compute two homogeneous planes, See Eq. (8)
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
			// Transform the two planes into local u-v system. 
			                             
			                          
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;

			// Cross product of two planes is a line, Eq. (9)
			float3 p = cross3D(k, l);
			if (p.z == 0.0) continue;
			// Perspective division to get the intersection (u,v), Eq. (10)
			float2 s = {p.x / p.z, p.y / p.z};          

			// compute depth
			float depth = (s.x * Tw.x + s.y * Tw.y) + Tw.z;                    
			float c_d = depth;                          
			                                                           
	 
			if (depth < near_n) continue;

			float2 p1_tangent = collected_p_tangent[j * 3 + 0];
			float2 p2_tangent = collected_p_tangent[j * 3 + 1];
                                                             
			const float current_triangle_a =  p2_tangent.x;
			float2 p3_tangent = collected_p_tangent[j * 3 + 2];                                                                    

			float4 con_o = collected_conic_opacity[j];
			float normal[3] = {con_o.x, con_o.y, con_o.z};               

			float distances[MAX_NB_POINTS];
			float max_val = -INFINITY;

			                                                     
			for (int k = 0; k < 3; k++) {
				                                                                                                                                                                                                
				distances[k] = collected_normals[j * MAX_NB_POINTS + k].x * s.x + collected_normals[j * MAX_NB_POINTS + k].y * s.y + collected_offsets[j * MAX_NB_POINTS + k];                           
                                                 
				if(distances[k] > 0.0000001f) 
				{
					distances[k] = distances[k] - 0.025f;
					               
				}
 
				distances[k] =  distance_amplification_factor * distances[k];    // for test normalization

				if (distances[k] > max_val) {
					max_val = distances[k];
				}
			}
			
			// no use depths into weight
			float sum_exp = 0.0f;
			for (int k = 0; k < 3; k++) {
				sum_exp += expf(1.0 * collected_delta[j] * (distances[k]-max_val));
			}
			float phi_x = 1.0 * collected_delta[j]*max_val + logf(sum_exp);
			float Cx = 1.0f / (1.0f + expf(1.0 * collected_sigma[j] * phi_x));
			const float alpha = min(0.99f, con_o.w * Cx);

			if (alpha < 1.0f / 255.0f)
				continue;
            
			                               
			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			float dL_dalpha = 0.0f;
			const int global_father_id = collected_father_id[j];                             
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				atomicAdd(&(dL_dcolors[global_father_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			
			float dL_dz = 0.0f;                                                                     
			float dL_dweight = 0;
 
			               
			const float m_d = far_n / (far_n - near_n) * (1 - near_n / c_d);
			const float dmd_dd = (far_n * near_n) / ((far_n - near_n) * c_d * c_d);
                	
			if (contributor == median_contributor-1) {                                                        
				dL_dz += dL_dmedian_depth;                                      
				// dL_dweight += dL_dmax_dweight;
			}

			dL_dweight += (final_D2 + m_d * m_d * final_A - 2 * m_d * final_D) * dL_dreg;                    
			dL_dalpha += dL_dweight - last_dL_dT;                                 
			// propagate the current weight W_{i} to next weight W_{i-1}
			last_dL_dT = dL_dweight * alpha + (1 - alpha) * last_dL_dT;
			const float dL_dmd = 2.0f * (T * alpha) * (m_d * final_A - final_D) * dL_dreg;
			dL_dz += dL_dmd * dmd_dd;                            
			                                       
			// Propagate gradients w.r.t ray-splat depths
			accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;                                                        
			last_depth = c_d;
			dL_dalpha += (c_d - accum_depth_rec) * dL_ddepth;                                      
                              
			// Propagate gradients w.r.t. color ray-splat alphas
			accum_alpha_rec = last_alpha * 1.0 + (1.f - last_alpha) * accum_alpha_rec;                                     
			dL_dalpha += (1 - accum_alpha_rec) * dL_daccum;                                        

			                                                      
			// Propagate gradients to per-Gaussian normals
			for (int ch = 0; ch < 3; ch++) {
				accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];                                                             
				last_normal[ch] = normal[ch];
				dL_dalpha += (normal[ch] - accum_normal_rec[ch]) * dL_dnormal2D[ch];                                                      
				atomicAdd((&dL_dnormal3D[global_father_id * 3 + ch]), alpha * T * dL_dnormal2D[ch]);                 
			}        
			                            
			dL_dz += alpha * T * dL_ddepth;                                                                                         
			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;                      

			// Helpful reusable temporary variables
			const float dL_dC = con_o.w * dL_dalpha;

            float dL_dsigma_value = -1.0 * phi_x * Cx * (1.0f - Cx) * dL_dC;   
            atomicAdd(&(dL_dsigma[global_father_id]), dL_dsigma_value);
            float dL_dphi_x = -collected_sigma[j]  * 1.0 * Cx * (1.0f - Cx) * dL_dC;  

            // Calculate gradients with respect to distances
            float dL_ddistances[MAX_NB_POINTS];
            for (int k = 0; k < 3; k++) {
                float exp_val = expf(1.0 * collected_delta[j] * (distances[k]-max_val));
                dL_ddistances[k] = (exp_val / sum_exp) * dL_dphi_x * collected_delta[j] * 1.0;
            }
			float dL_ddelta_value = 0.0f;
			for (int k = 0; k < 3; k++) {
				float exp_val = expf(collected_delta[j] * 1.0 * (distances[k]-max_val));	
				dL_ddelta_value += 1.0 * (distances[k]-max_val) * exp_val / sum_exp;
			}

			float dL_ddelta_value_aux = (1.0 * max_val + dL_ddelta_value) * dL_dphi_x;     

			// Apply the gradient update to delta
			atomicAdd(&(dL_ddelta[global_father_id]), dL_ddelta_value_aux);
            atomicAdd(&(dL_da[global_father_id]), dL_ddistances[0] * (-1.0 * distance_amplification_factor * s.y));   // -k * s_y
            atomicAdd(&(dL_da[global_father_id]), dL_ddistances[1] * (2 * distance_amplification_factor * s.y));      // 2k * s_y
            atomicAdd(&(dL_da[global_father_id]), dL_ddistances[2] * (-1.0 * distance_amplification_factor * s.y));   // -k * s_y

			const float2 dL_ds = {
				distance_amplification_factor * (dL_ddistances[0] + dL_ddistances[1] * (-2.0) + dL_ddistances[2]) + dL_dz * Tw.x,
				distance_amplification_factor * (dL_ddistances[0] * (1.0 - current_triangle_a) + dL_ddistances[1] * (2.0 * current_triangle_a + 1.0) + dL_ddistances[2] * (-2.0 - current_triangle_a)) + dL_dz * Tw.y
			};
			const float3 dz_dTw = {s.x, s.y, 1.0};
			const float dsx_pz = dL_ds.x / p.z;
			const float dsy_pz = dL_ds.y / p.z;
			const float3 dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};  // p = cross3D(k, l);
			const float3 dL_dk = cross3D(l, dL_dp);
			const float3 dL_dl = cross3D(dL_dp, k);

			const float3 dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
			const float3 dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
			const float3 dL_dTw = {
				pixf.x * dL_dk.x + pixf.y * dL_dl.x + dL_dz * dz_dTw.x, 
				pixf.x * dL_dk.y + pixf.y * dL_dl.y + dL_dz * dz_dTw.y, 
				pixf.x * dL_dk.z + pixf.y * dL_dl.z + dL_dz * dz_dTw.z};


			// Update gradients w.r.t. 3D covariance (3x3 matrix)
			atomicAdd(&dL_dtransMat[global_father_id * 9 + 0],  dL_dTu.x);
			atomicAdd(&dL_dtransMat[global_father_id * 9 + 1],  dL_dTu.y);
			atomicAdd(&dL_dtransMat[global_father_id * 9 + 2],  dL_dTu.z);
			atomicAdd(&dL_dtransMat[global_father_id * 9 + 3],  dL_dTv.x);
			atomicAdd(&dL_dtransMat[global_father_id * 9 + 4],  dL_dTv.y);
			atomicAdd(&dL_dtransMat[global_father_id * 9 + 5],  dL_dTv.z);
			atomicAdd(&dL_dtransMat[global_father_id * 9 + 6],  dL_dTw.x);
			atomicAdd(&dL_dtransMat[global_father_id * 9 + 7],  dL_dTw.y);
			atomicAdd(&dL_dtransMat[global_father_id * 9 + 8],  dL_dTw.z);               
			atomicAdd(&(dL_dopacity[global_father_id]), dL_dalpha * Cx);
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float* convex_points,
	int W, int H,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const float* viewmatrix,
	const float* projmatrix,
	const int* num_points_per_convex,
	const int* cumsum_of_points_per_convex,
	float4* p_hom,
	float* p_w,
	float3* p_proj,
	float2* p_image,
	int* hull,
	int* indices,
	int* num_points_per_convex_view,
	const float* cov3Ds,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	glm::vec3* dL_dconvex,
	float* dL_dtransMat,                                 
	float* dL_da,                                        
	float* dL_dnormal3D,                                 
	glm::vec3* dL_dmean3D,
	float3* dL_dmean2D,
	const float* dL_dconic,
	float* dL_dcov3D,
	float* dL_dcolor,
	float* dL_dsh
	)
{
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		convex_points,
		W, H,
		radii,
		shs,
		clamped,
		viewmatrix,    
		projmatrix,
		num_points_per_convex,
		cumsum_of_points_per_convex,
		p_hom,
		p_w,
		p_proj,
		p_image,
		hull,
		indices,
		num_points_per_convex_view,
		campos,
		(glm::vec3*)dL_dconvex,
		dL_dtransMat,                                 
		dL_da,                                        
		dL_dnormal3D,                                 
		(glm::vec3*)dL_dmean3D,
		(float3*)dL_dmean2D,
		dL_dcov3D,
		dL_dcolor,
		dL_dsh);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	int sub_triangle_total_num,               
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float* delta,
	const float* sigma,
	const int* num_points_per_convex,
	const int* cumsum_of_points_per_convex,
	const float2* normals,
	const float* offsets,
	const int* num_points_per_convex_view,
	float* transMats,     
	const float4* conic_opacity,
	const float* depths,
	const float2* p_tangent,                                      
	const float* colors,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* sub_triangle_depths,                        
	const float2* sub_triangle_p_image,                             
	const uint32_t* sub_triangle_father_index,       
	const bool* sub_triangle_visibility,                      
	const float* dL_dpixels,
	const float* dL_depths,                       
	float* dL_ddelta,
	float* dL_dsigma,
	float* dL_dtransMat,                                 
	float* dL_da,                                        
	float* dL_dnormal3D,                                 
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		sub_triangle_total_num,               
		ranges,
		point_list,
		W, H,
		bg_color,
		delta,
		sigma,
		num_points_per_convex,
		cumsum_of_points_per_convex,
		normals,
		offsets,
		num_points_per_convex_view,
		transMats,                       
		conic_opacity,
		depths,
		p_tangent,                                                  
		colors,
		final_Ts,
		n_contrib,
		sub_triangle_depths,                        
		sub_triangle_p_image,                             
		sub_triangle_father_index,       
		sub_triangle_visibility,                      
		dL_dpixels,
		dL_depths,                                                                                                
		dL_ddelta,
		dL_dsigma,
        dL_dtransMat,                                 
		dL_da,                                        
		dL_dnormal3D,                                 
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors
		);
}