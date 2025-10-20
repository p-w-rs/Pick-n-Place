# NvDiffrast.jl
module NvDiffRast

using CUDA
using LinearAlgebra
using FileIO
using MeshIO
using GeometryBasics
using Statistics

const libnvdr = joinpath(@__DIR__, "build", "libnvdiffrast.so")

# Context management
mutable struct RasterizeContext
    ptr::Ptr{Cvoid}

    function RasterizeContext()
        ptr = @ccall libnvdr.nvdr_create_context()::Ptr{Cvoid}
        ctx = new(ptr)
        finalizer(ctx) do c
            @ccall libnvdr.nvdr_destroy_context(c.ptr::Ptr{Cvoid})::Cvoid
        end
        return ctx
    end
end

# Helper to get device pointer
cu_pointer(x::CuArray) = reinterpret(Ptr{eltype(x)}, pointer(x))

# Rasterize
function rasterize(
    ctx::RasterizeContext,
    pos::CuArray{Float32, 2},      # [N, 4] clip space positions
    tri::CuArray{Int32, 2};         # [T, 3] triangle indices
    resolution::Tuple{Int, Int}=(512, 512)
)
    N = size(pos, 1)
    T = size(tri, 1)
    H, W = resolution

    out = CUDA.zeros(Float32, H, W, 4)
    out_db = CUDA.zeros(Int32, H, W, 4)

    ret = @ccall libnvdr.nvdr_rasterize(
        ctx.ptr::Ptr{Cvoid},
        cu_pointer(pos)::Ptr{Float32},
        cu_pointer(tri)::Ptr{Int32},
        N::Cint,
        T::Cint,
        H::Cint,
        W::Cint,
        cu_pointer(out)::Ptr{Float32},
        cu_pointer(out_db)::Ptr{Int32}
    )::Cint

    ret != 0 && error("Rasterization failed")
    return out, out_db
end

# Interpolate
function interpolate(
    ctx::RasterizeContext,
    attr::CuArray{Float32, 2},     # [N, C] vertex attributes
    rast::CuArray{Float32, 3},     # [H, W, 4] rasterization output
    rast_db::CuArray{Int32, 3},    # [H, W, 4] rasterization derivatives
    tri::CuArray{Int32, 2}         # [T, 3] triangle indices
)
    N, C = size(attr)
    T = size(tri, 1)
    H, W, _ = size(rast)

    out = CUDA.zeros(Float32, H, W, C)

    ret = @ccall libnvdr.nvdr_interpolate(
        ctx.ptr::Ptr{Cvoid},
        cu_pointer(attr)::Ptr{Float32},
        cu_pointer(rast)::Ptr{Float32},
        cu_pointer(rast_db)::Ptr{Int32},
        cu_pointer(tri)::Ptr{Int32},
        N::Cint,
        T::Cint,
        C::Cint,
        H::Cint,
        W::Cint,
        cu_pointer(out)::Ptr{Float32}
    )::Cint

    ret != 0 && error("Interpolation failed")
    return out
end

# Texture sampling
function texture(
    ctx::RasterizeContext,
    tex::CuArray{Float32, 3},      # [TH, TW, C] texture
    uv::CuArray{Float32, 3};       # [H, W, 2] UV coordinates
    filter_mode::Int=1,             # 1=linear
    boundary_mode::Int=0            # 0=wrap
)
    TH, TW, C = size(tex)
    H, W, _ = size(uv)

    out = CUDA.zeros(Float32, H, W, C)

    ret = @ccall libnvdr.nvdr_texture(
        ctx.ptr::Ptr{Cvoid},
        cu_pointer(tex)::Ptr{Float32},
        cu_pointer(uv)::Ptr{Float32},
        TH::Cint,
        TW::Cint,
        C::Cint,
        H::Cint,
        W::Cint,
        filter_mode::Cint,
        boundary_mode::Cint,
        cu_pointer(out)::Ptr{Float32}
    )::Cint

    ret != 0 && error("Texture sampling failed")
    return out
end

# Antialias
function antialias(
    ctx::RasterizeContext,
    color::CuArray{Float32, 3},    # [H, W, C] input color
    rast::CuArray{Float32, 3},     # [H, W, 4] rasterization output
    pos::CuArray{Float32, 2},      # [N, 4] vertex positions
    tri::CuArray{Int32, 2}         # [T, 3] triangle indices
)
    H, W, C = size(color)
    N = size(pos, 1)
    T = size(tri, 1)

    out = CUDA.zeros(Float32, H, W, C)

    ret = @ccall libnvdr.nvdr_antialias(
        ctx.ptr::Ptr{Cvoid},
        cu_pointer(color)::Ptr{Float32},
        cu_pointer(rast)::Ptr{Float32},
        cu_pointer(pos)::Ptr{Float32},
        cu_pointer(tri)::Ptr{Int32},
        N::Cint,
        T::Cint,
        C::Cint,
        H::Cint,
        W::Cint,
        cu_pointer(out)::Ptr{Float32}
    )::Cint

    ret != 0 && error("Antialiasing failed")
    return out
end

# Utility functions
function load_mesh(path::String)
    mesh = load(path)

    # Extract vertices
    coords = GeometryBasics.coordinates(mesh)
    vertices = zeros(Float32, length(coords), 3)
    for (i, v) in enumerate(coords)
        vertices[i, 1] = Float32(v[1])
        vertices[i, 2] = Float32(v[2])
        vertices[i, 3] = Float32(v[3])
    end

    # Extract faces (convert to 0-indexed)
    mesh_faces = GeometryBasics.faces(mesh)
    face_array = zeros(Int32, length(mesh_faces), 3)
    for (i, f) in enumerate(mesh_faces)
        face_array[i, 1] = Int32(f[1] - 1)
        face_array[i, 2] = Int32(f[2] - 1)
        face_array[i, 3] = Int32(f[3] - 1)
    end

    return vertices, face_array
end

function perspective_projection(fov_y::Float32, aspect::Float32, near::Float32, far::Float32)
    f = 1.0f0 / tan(fov_y / 2.0f0)
    return Float32[
        f/aspect 0 0 0;
        0 f 0 0;
        0 0 (far+near)/(near-far) (2*far*near)/(near-far);
        0 0 -1 0
    ]
end

function look_at(eye::Vector{Float32}, center::Vector{Float32}, up::Vector{Float32})
    f = normalize(center - eye)
    s = normalize(cross(f, up))
    u = cross(s, f)

    view = Float32[
        s[1] s[2] s[3] -dot(s, eye);
        u[1] u[2] u[3] -dot(u, eye);
        -f[1] -f[2] -f[3] dot(f, eye);
        0 0 0 1
    ]
    return view
end

function spherical_to_cartesian(azimuth::Float32, elevation::Float32, distance::Float32)
    x = distance * cos(elevation) * cos(azimuth)
    y = distance * cos(elevation) * sin(azimuth)
    z = distance * sin(elevation)
    return Float32[x, y, z]
end

export RasterizeContext, rasterize, interpolate, texture, antialias
export load_mesh, perspective_projection, look_at, spherical_to_cartesian

end # module
