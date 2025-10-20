include("../TensorRT.jl")
using .TensorRT
using CUDA

# Compile ONNX to TRT
compile_trt("TensorRT/test/refine_model.onnx", "TensorRT/test/refine_model.engine";
    opt_level=5, fp16=true, workspace_gb=nothing,
    min_batch=1, opt_batch=32, max_batch=256)
compile_trt("TensorRT/test/score_model.onnx", "TensorRT/test/score_model.engine";
    opt_level=5, fp16=true, workspace_gb=16,
    min_batch=1, opt_batch=32, max_batch=256)

# Load and use
model = TRTModel("TensorRT/test/refine_model.engine")
n = n_inputs(model)
println("Inputs: ", n)
for i in 1:n
    println("Input $(i): ", input_name(model, i), " ", input_size(model, i))
end

n = n_outputs(model)
println("Outputs: ", n)
for i in 1:n
    println("Output $(i): ", output_name(model, i), " ", output_size(model, i))
end

# Inference
input1 = CUDA.randn(Float32, 256, 160, 160, 6)
input2 = CUDA.randn(Float32, 256, 160, 160, 6)
output = model(input1, input2)
close(model)

# Multiple models in parallel
model1 = TRTModel("TensorRT/test/refine_model.engine")
model2 = TRTModel("TensorRT/test/score_model.engine")
@sync begin
    @async result1 = model1(input1, input2)
    @async result2 = model2(input1, input2)
end

@sync begin
    @async close(model1)
    @async close(model2)
end
