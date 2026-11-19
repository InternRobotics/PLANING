from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_convexes(
    convex_points,
    delta, 
    sigma,
    num_split_per_convex,                  
    cumsum_of_subtriangles_per_convex,     
    num_points_per_convex,
    cumsum_of_points_per_convex,
    number_of_points,
    sh,
    colors_precomp,
    opacities,
    means2D,
    scaling,
    density_factor,
    raster_settings,
):
    return _RasterizeConvexes.apply(
        convex_points,
        delta, 
        sigma,
        num_split_per_convex,          
        cumsum_of_subtriangles_per_convex,    
        num_points_per_convex,
        cumsum_of_points_per_convex,
        number_of_points,
        sh,
        colors_precomp,
        opacities,
        means2D,
        scaling,
        density_factor,
        raster_settings,
    )

class _RasterizeConvexes(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        convex_points,
        delta, 
        sigma,
        num_split_per_convex,     
        cumsum_of_subtriangles_per_convex,    
        num_points_per_convex,
        cumsum_of_points_per_convex,
        number_of_points,
        sh,
        colors_precomp,
        opacities,
        means2D,                                 
        scaling,
        density_factor,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            convex_points,
            delta, 
            sigma,
            num_split_per_convex,     
            cumsum_of_subtriangles_per_convex,   
            num_points_per_convex,
            cumsum_of_points_per_convex,
            colors_precomp,
            opacities,
            scaling,
            density_factor,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            number_of_points,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        
                                        
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, num_sub_triangle, color, depth, radii, impact, geomBuffer, subGeometryBuffer, binningBuffer, imgBuffer, scaling, density_factor = _C.rasterize_convexes(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, num_sub_triangle, color, depth, radii, impact, geomBuffer, subGeometryBuffer, binningBuffer, imgBuffer, scaling, density_factor = _C.rasterize_convexes(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.number_of_points = number_of_points
        
        ctx.num_sub_triangle = num_sub_triangle       
        ctx.save_for_backward(convex_points, delta, sigma, num_points_per_convex, cumsum_of_points_per_convex, colors_precomp, radii, sh, geomBuffer, subGeometryBuffer, binningBuffer, imgBuffer)
        return color, radii, scaling, density_factor, impact, depth                                        

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_scaling, grad_density_factor, grad_impact, grad_depth):
    # grad_radii, grad_scaling, grad_density_factor, grad_depth for geometry Loss
 
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        number_of_points = ctx.number_of_points
        
        num_sub_triangle = ctx.num_sub_triangle                    
        # subGeometryBuffer for sub triangle
        convex_points, delta, sigma, num_points_per_convex, cumsum_of_points_per_convex, colors_precomp, radii, sh, geomBuffer, subGeometryBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                convex_points,
                delta,
                sigma,
                num_points_per_convex,
                cumsum_of_points_per_convex,
                radii, 
                colors_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                number_of_points,
                num_sub_triangle,     
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                grad_depth,            # grad_depth for geometry Loss
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                subGeometryBuffer,           # subGeometryBuffer for sub triangle
                binningBuffer,
                imgBuffer,
                raster_settings.debug)
           
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_convex, grad_delta, grad_sigma, grad_colors_precomp, grad_opacities, grad_sh, grad_means2D = _C.rasterize_convexes_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_convex, grad_delta, grad_sigma, grad_colors_precomp, grad_opacities, grad_sh, grad_means2D = _C.rasterize_convexes_backward(*args)

        grad_convex = grad_convex.flatten(0)
        grad_delta = grad_delta.view(-1, 1) 
        grad_sigma = grad_sigma.view(-1, 1)

                              
        grads = (
            grad_convex,                   # convex_points
            grad_delta,                    # delta
            grad_sigma,                    # sigma
            None,                          # num_split_per_convex    
            None,                          # cumsum_of_subtriangles_per_convex       
            None,                          # num_points_per_convex
            None,                          # cumsum_of_points_per_convex
            None,                          # number_of_points
            grad_sh,                       # sh
            grad_colors_precomp,           # colors_precomp
            grad_opacities,                # opacities
            grad_means2D,                  # means2D    
            None,                          # scaling
            None,                          # density_factor
            None                           # raster_settings
        )

        return grads

class ConvexRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class ConvexRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

                                                                             
    def forward(self, convex_points, delta, sigma, num_split_per_convex, cumsum_of_subtriangles_per_convex, num_points_per_convex, cumsum_of_points_per_convex, number_of_points, opacities, means2D, scaling, density_factor,  shs = None, colors_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if shs is None:
            shs = torch.Tensor([]).cuda()
        if colors_precomp is None:
            colors_precomp = torch.Tensor([]).cuda()

        # Invoke C++/CUDA rasterization routine
        return rasterize_convexes(
            convex_points,
            delta,
            sigma,
            num_split_per_convex,
            cumsum_of_subtriangles_per_convex,
            num_points_per_convex,
            cumsum_of_points_per_convex,
            number_of_points,
            shs,
            colors_precomp,
            opacities,
            means2D,
            scaling,
            density_factor,
            raster_settings, 
        )

