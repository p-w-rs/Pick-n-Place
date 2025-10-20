module TensorRT

using CUDA

const LIB = joinpath(@__DIR__, "build", "libtrt.so")

mutable struct TRTModel
    ptr::Ptr{Cvoid}
    n_inputs::Int
    n_outputs::Int
    input_names::Vector{String}
    output_names::Vector{String}
    input_shapes::Vector{Vector{Int}}
    output_shapes::Vector{Vector{Int}}
    output_buffers::Union{Nothing,Vector{CuArray}}
    cached_batch_size::Int
end

function auto_workspace_gb()
    try
        free_gpu_mem = CUDA.available_memory()
        workspace_gb = Int(free_gpu_mem ÷ 1024^3)
        println("Auto-detected workspace: $workspace_gb GB")
        return workspace_gb
    catch
        println("GPU not detected, using default 4 GB workspace")
        return 4
    end
end

function compile_trt(
    onnx_path::String,
    engine_path::String;
    opt_level::Int=3,
    fp16::Bool=false,
    workspace_gb::Union{Nothing,Int}=nothing,
    min_batch::Int=1,
    opt_batch::Int=1,
    max_batch::Int=1
)
    # Auto-detect workspace if not provided
    ws_gb = workspace_gb === nothing ? auto_workspace_gb() : workspace_gb
    workspace_bytes = UInt(ws_gb * 1024^3)

    error_msg = zeros(UInt8, 1024)

    serialized = @ccall LIB.trt_build_engine(
        onnx_path::Cstring,
        opt_level::Cint,
        fp16::Bool,
        workspace_bytes::Csize_t,
        min_batch::Cint,
        opt_batch::Cint,
        max_batch::Cint,
        error_msg::Ptr{UInt8},
        1024::Cint
    )::Ptr{Cvoid}

    serialized == C_NULL && error(unsafe_string(pointer(error_msg)))

    success = @ccall LIB.trt_save_engine(
        serialized::Ptr{Cvoid},
        engine_path::Cstring
    )::Bool

    success || error("Failed to save engine")
    println("Engine saved to $engine_path")
end

function TRTModel(engine_path::String)
    error_msg = zeros(UInt8, 1024)

    ptr = @ccall LIB.trt_load_engine(
        engine_path::Cstring,
        error_msg::Ptr{UInt8},
        1024::Cint
    )::Ptr{Cvoid}

    ptr == C_NULL && error(unsafe_string(pointer(error_msg)))

    n_inputs = @ccall LIB.trt_get_n_inputs(ptr::Ptr{Cvoid})::Cint
    n_outputs = @ccall LIB.trt_get_n_outputs(ptr::Ptr{Cvoid})::Cint

    input_names = [unsafe_string(@ccall LIB.trt_get_input_name(ptr::Ptr{Cvoid}, i::Cint)::Cstring)
                   for i in 1:n_inputs]
    output_names = [unsafe_string(@ccall LIB.trt_get_output_name(ptr::Ptr{Cvoid}, i::Cint)::Cstring)
                    for i in 1:n_outputs]

    input_shapes = [get_shape(ptr, name) for name in input_names]
    output_shapes = [get_shape(ptr, name) for name in output_names]

    TRTModel(ptr, n_inputs, n_outputs, input_names, output_names,
        input_shapes, output_shapes, nothing, 0)
end

function get_shape(ptr::Ptr{Cvoid}, name::String)
    dims = zeros(Int32, 8)
    n_dims = @ccall LIB.trt_get_tensor_shape(
        ptr::Ptr{Cvoid},
        name::Cstring,
        dims::Ptr{Cint}
    )::Cint
    return Int[dims[i] for i in 1:n_dims]
end

n_inputs(m::TRTModel) = m.n_inputs
n_outputs(m::TRTModel) = m.n_outputs
input_name(m::TRTModel, i::Int) = m.input_names[i]
output_name(m::TRTModel, i::Int) = m.output_names[i]
input_size(m::TRTModel, i::Int) = m.input_shapes[i]
output_size(m::TRTModel, i::Int) = m.output_shapes[i]

function set_input_shape!(m::TRTModel, name::String, dims::Vector{Int})
    dims_i32 = Int32.(dims)
    @ccall LIB.trt_set_input_shape(
        m.ptr::Ptr{Cvoid},
        name::Cstring,
        dims_i32::Ptr{Cint},
        length(dims)::Cint
    )::Bool
end

function allocate_outputs!(m::TRTModel, batch_size::Int)
    outputs = CuArray[]
    for i in 1:m.n_outputs
        dims = zeros(Int32, 8)
        n_dims = @ccall LIB.trt_get_tensor_shape(
            m.ptr::Ptr{Cvoid},
            m.output_names[i]::Cstring,
            dims::Ptr{Cint}
        )::Cint

        shape = Int[dims[j] for j in 1:n_dims]

        for j in 1:length(shape)
            if shape[j] == -1
                shape[j] = batch_size
            end
        end

        push!(outputs, CUDA.zeros(Float32, shape...))
    end

    m.output_buffers = outputs
    m.cached_batch_size = batch_size
end

function (m::TRTModel)(inputs::CuArray...)
    length(inputs) == m.n_inputs || error("Expected $(m.n_inputs) inputs, got $(length(inputs))")

    batch_size = size(inputs[1], 1)

    for (i, inp) in enumerate(inputs)
        actual_shape = collect(size(inp))
        expected_shape = m.input_shapes[i]
        if any(expected_shape .== -1)
            set_input_shape!(m, m.input_names[i], actual_shape)
        end
    end

    if m.output_buffers === nothing || m.cached_batch_size != batch_size
        allocate_outputs!(m, batch_size)
    end

    input_ptrs = Ptr{Cvoid}[reinterpret(Ptr{Cvoid}, pointer(inp)) for inp in inputs]
    output_ptrs = Ptr{Cvoid}[reinterpret(Ptr{Cvoid}, pointer(out)) for out in m.output_buffers]

    stream_ptr = reinterpret(Ptr{Cvoid}, CUDA.stream().handle)

    success = @ccall LIB.trt_infer(
        m.ptr::Ptr{Cvoid},
        input_ptrs::Ptr{Ptr{Cvoid}},
        output_ptrs::Ptr{Ptr{Cvoid}},
        stream_ptr::Ptr{Cvoid}
    )::Bool

    success || error("Inference failed")

    return length(m.output_buffers) == 1 ? m.output_buffers[1] : tuple(m.output_buffers...)
end

function Base.close(m::TRTModel)
    m.output_buffers = nothing
    @ccall LIB.trt_destroy(m.ptr::Ptr{Cvoid})::Cvoid
end

export TRTModel, compile_trt, n_inputs, n_outputs, input_name, output_name, input_size, output_size

end # module
