# Main.jl
include("../NvDiffRast.jl")
using .NvDiffRast
using CUDA
using Images
using FileIO
using Statistics

function render_mesh(
    mesh_path::String;
    resolution::Tuple{Int,Int}=(512, 512),
    azimuth::Float32=0.0f0,
    elevation::Float32=0.3f0,
    distance::Float32=2.5f0,
    fov::Float32=Float32(π / 4)
)
    # Load mesh
    vertices_cpu, faces_cpu = load_mesh(mesh_path)

    # Center and normalize
    center = mean(vertices_cpu, dims=1)
    vertices_cpu .-= center
    scale = maximum(sqrt.(sum(vertices_cpu .^ 2, dims=2)))
    vertices_cpu ./= scale

    # Upload to GPU
    vertices = CuArray(vertices_cpu)
    faces = CuArray(faces_cpu)

    # Setup camera
    eye = spherical_to_cartesian(azimuth, elevation, distance)
    view = look_at(eye, Float32[0, 0, 0], Float32[0, 0, 1])
    proj = perspective_projection(fov, Float32(resolution[2] / resolution[1]), 0.1f0, 10.0f0)
    mvp = proj * view

    # Transform vertices to clip space
    vertices_homo = hcat(vertices, CUDA.ones(Float32, size(vertices, 1), 1))
    mvp_gpu = CuArray(mvp)
    vertices_clip = CuArray{Float32}((mvp_gpu * vertices_homo')')  # Keep on GPU

    # Create context and rasterize
    ctx = RasterizeContext()
    rast, rast_db = rasterize(ctx, vertices_clip, faces; resolution=resolution)

    # Create vertex colors (simple gray)
    colors = CUDA.fill(0.7f0, size(vertices, 1), 3)

    # Interpolate colors
    rgb = interpolate(ctx, colors, rast, rast_db, faces)

    # Apply alpha
    alpha = rast[:, :, 4:4]
    rgb_final = rgb .* alpha

    # Antialias
    rgb_aa = antialias(ctx, rgb_final, rast, vertices_clip, faces)

    # Download and convert to image
    img_data = Array(rgb_aa)
    alpha_data = Array(alpha)

    # Create RGBA image
    rgba = cat(img_data, alpha_data, dims=3)
    rgba = permutedims(rgba, (2, 1, 3))  # Transpose for image format
    rgba = clamp.(rgba, 0.0f0, 1.0f0)

    return colorview(RGBA, rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3], rgba[:, :, 4])
end

# Render from multiple viewpoints
function render_turntable(mesh_path::String, output_dir::String; num_views::Int=36)
    mkpath(output_dir)

    for i in 1:num_views
        azimuth = Float32(2π * (i - 1) / num_views)
        img = render_mesh(mesh_path; azimuth=azimuth)
        save(joinpath(output_dir, "view_$(lpad(i, 3, '0')).png"), img)
        println("Rendered view $i/$num_views")
    end
end

# Usage
render_turntable("NvDiffRast/test/obj_000001.ply", "NvDiffRast/test/output_renders"; num_views=36)
