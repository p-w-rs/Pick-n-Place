#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <memory>
#include <cstring>
#include <mutex>

using namespace nvinfer1;

template<typename T>
struct TRTDeleter {
    void operator()(T* obj) const { delete obj; }
};

template<>
struct TRTDeleter<IRuntime> {
    void operator()(IRuntime* obj) const { delete obj; }
};

template<>
struct TRTDeleter<ICudaEngine> {
    void operator()(ICudaEngine* obj) const { delete obj; }
};

template<>
struct TRTDeleter<IExecutionContext> {
    void operator()(IExecutionContext* obj) const { delete obj; }
};

template<>
struct TRTDeleter<IBuilder> {
    void operator()(IBuilder* obj) const { delete obj; }
};

template<>
struct TRTDeleter<INetworkDefinition> {
    void operator()(INetworkDefinition* obj) const { delete obj; }
};

template<>
struct TRTDeleter<IBuilderConfig> {
    void operator()(IBuilderConfig* obj) const { delete obj; }
};

template<>
struct TRTDeleter<nvonnxparser::IParser> {
    void operator()(nvonnxparser::IParser* obj) const { delete obj; }
};

template<>
struct TRTDeleter<IHostMemory> {
    void operator()(IHostMemory* obj) const { delete obj; }
};

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            fprintf(stderr, "[TRT] %s\n", msg);
    }
};

// Global logger instance
static Logger g_logger;
static std::mutex g_logger_mutex;

struct TRTEngine {
    std::unique_ptr<IRuntime, TRTDeleter<IRuntime>> runtime;
    std::unique_ptr<ICudaEngine, TRTDeleter<ICudaEngine>> engine;
    std::unique_ptr<IExecutionContext, TRTDeleter<IExecutionContext>> context;
    int n_bindings;
};

extern "C" {

void* trt_build_engine(
    const char* onnx_path,
    int opt_level,
    bool fp16,
    size_t workspace_bytes,
    int min_batch,
    int opt_batch,
    int max_batch,
    char* error_msg,
    int error_msg_size
) {
    std::lock_guard<std::mutex> lock(g_logger_mutex);

    auto builder = std::unique_ptr<IBuilder, TRTDeleter<IBuilder>>(createInferBuilder(g_logger));
    if (!builder) {
        snprintf(error_msg, error_msg_size, "Failed to create builder");
        return nullptr;
    }

    auto network = std::unique_ptr<INetworkDefinition, TRTDeleter<INetworkDefinition>>(
        builder->createNetworkV2(0U));

    auto parser = std::unique_ptr<nvonnxparser::IParser, TRTDeleter<nvonnxparser::IParser>>(
        nvonnxparser::createParser(*network, g_logger));

    if (!parser->parseFromFile(onnx_path, static_cast<int>(ILogger::Severity::kWARNING))) {
        snprintf(error_msg, error_msg_size, "Failed to parse ONNX");
        return nullptr;
    }

    auto config = std::unique_ptr<IBuilderConfig, TRTDeleter<IBuilderConfig>>(
        builder->createBuilderConfig());

    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, workspace_bytes);

    if (opt_level >= 0 && opt_level <= 5)
        config->setBuilderOptimizationLevel(opt_level);

    if (fp16) {
        config->setFlag(BuilderFlag::kFP16);
        config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    }

    // Dynamic batch size via optimization profile
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    for (int i = 0; i < network->getNbInputs(); i++) {
        auto* input = network->getInput(i);
        Dims dims = input->getDimensions();

        Dims min_dims = dims, opt_dims = dims, max_dims = dims;
        if (dims.d[0] == -1) {
            min_dims.d[0] = min_batch;
            opt_dims.d[0] = opt_batch;
            max_dims.d[0] = max_batch;
        }
        profile->setDimensions(input->getName(), OptProfileSelector::kMIN, min_dims);
        profile->setDimensions(input->getName(), OptProfileSelector::kOPT, opt_dims);
        profile->setDimensions(input->getName(), OptProfileSelector::kMAX, max_dims);
    }
    config->addOptimizationProfile(profile);

    auto serialized = std::unique_ptr<IHostMemory, TRTDeleter<IHostMemory>>(
        builder->buildSerializedNetwork(*network, *config));

    if (!serialized) {
        snprintf(error_msg, error_msg_size, "Failed to build engine");
        return nullptr;
    }

    return serialized.release();
}

bool trt_save_engine(void* serialized_engine, const char* path) {
    auto mem = std::unique_ptr<IHostMemory, TRTDeleter<IHostMemory>>(
        static_cast<IHostMemory*>(serialized_engine));

    std::ofstream file(path, std::ios::binary);
    if (!file) return false;

    file.write(static_cast<const char*>(mem->data()), mem->size());
    return true;
}

void* trt_load_engine(const char* path, char* error_msg, int error_msg_size) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        snprintf(error_msg, error_msg_size, "Failed to open engine file");
        return nullptr;
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);

    TRTEngine* trt = new TRTEngine();
    trt->runtime.reset(createInferRuntime(g_logger));
    trt->engine.reset(trt->runtime->deserializeCudaEngine(buffer.data(), size));

    if (!trt->engine) {
        snprintf(error_msg, error_msg_size, "Failed to deserialize engine");
        delete trt;
        return nullptr;
    }

    trt->context.reset(trt->engine->createExecutionContext());
    trt->n_bindings = trt->engine->getNbIOTensors();

    return trt;
}

int trt_get_n_inputs(void* engine_ptr) {
    TRTEngine* trt = static_cast<TRTEngine*>(engine_ptr);
    int count = 0;
    for (int i = 0; i < trt->n_bindings; i++) {
        const char* name = trt->engine->getIOTensorName(i);
        if (trt->engine->getTensorIOMode(name) == TensorIOMode::kINPUT)
            count++;
    }
    return count;
}

int trt_get_n_outputs(void* engine_ptr) {
    TRTEngine* trt = static_cast<TRTEngine*>(engine_ptr);
    int count = 0;
    for (int i = 0; i < trt->n_bindings; i++) {
        const char* name = trt->engine->getIOTensorName(i);
        if (trt->engine->getTensorIOMode(name) == TensorIOMode::kOUTPUT)
            count++;
    }
    return count;
}

const char* trt_get_input_name(void* engine_ptr, int idx) {
    TRTEngine* trt = static_cast<TRTEngine*>(engine_ptr);
    int count = 0;
    for (int i = 0; i < trt->n_bindings; i++) {
        const char* name = trt->engine->getIOTensorName(i);
        if (trt->engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
            if (++count == idx) return name;
        }
    }
    return nullptr;
}

const char* trt_get_output_name(void* engine_ptr, int idx) {
    TRTEngine* trt = static_cast<TRTEngine*>(engine_ptr);
    int count = 0;
    for (int i = 0; i < trt->n_bindings; i++) {
        const char* name = trt->engine->getIOTensorName(i);
        if (trt->engine->getTensorIOMode(name) == TensorIOMode::kOUTPUT) {
            if (++count == idx) return name;
        }
    }
    return nullptr;
}

int trt_get_tensor_shape(void* engine_ptr, const char* name, int* dims_out) {
    TRTEngine* trt = static_cast<TRTEngine*>(engine_ptr);
    Dims dims = trt->engine->getTensorShape(name);
    for (int i = 0; i < dims.nbDims; i++)
        dims_out[i] = dims.d[i];
    return dims.nbDims;
}

bool trt_set_input_shape(void* engine_ptr, const char* name, const int* dims, int n_dims) {
    TRTEngine* trt = static_cast<TRTEngine*>(engine_ptr);
    Dims d;
    d.nbDims = n_dims;
    for (int i = 0; i < n_dims; i++)
        d.d[i] = dims[i];
    return trt->context->setInputShape(name, d);
}

bool trt_infer(void* engine_ptr, void** input_ptrs, void** output_ptrs, void* stream) {
    TRTEngine* trt = static_cast<TRTEngine*>(engine_ptr);

    int in_idx = 0, out_idx = 0;
    for (int i = 0; i < trt->n_bindings; i++) {
        const char* name = trt->engine->getIOTensorName(i);
        if (trt->engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
            trt->context->setTensorAddress(name, input_ptrs[in_idx++]);
        } else {
            trt->context->setTensorAddress(name, output_ptrs[out_idx++]);
        }
    }

    return trt->context->enqueueV3(static_cast<cudaStream_t>(stream));
}

void trt_destroy(void* engine_ptr) {
    TRTEngine* trt = static_cast<TRTEngine*>(engine_ptr);
    delete trt;
}

} // extern "C"
