#include "tensorrt.hpp"
#include <dlfcn.h>

namespace dnn_engine {

tensorrt::tensorrt() : m_logger{std::make_unique<TensorrtLogger>()}{}

tensorrt::~tensorrt() {
    // if(m_params.m_outputs.size() > 0) {
    //     rknn_outputs_release(m_params.m_rknnCtx, m_params.m_io_num.n_output, m_params.m_outputs.data());
    // }
    // if(m_params.m_rknnCtx) {
    //     rknn_destroy(m_params.m_rknnCtx);
    // }
    m_params.m_model_data.reset();

    // delete this->context;
    // delete this->engine;
    // delete this->runtime;
    m_params.m_context.reset();
    m_params.m_cuda_engine.reset();
    m_params.m_runtime.reset();
    
    // cudaStreamDestroy(this->stream);
    // for (auto& ptr : this->device_ptrs) {
    //     CHECK(cudaFree(ptr));
    // }

    // for (auto& ptr : this->host_ptrs) {
    //     CHECK(cudaFreeHost(ptr));
    // }
}

std::shared_ptr<unsigned char> tensorrt::loadModelFile(const std::string& modelPath) {
    FILE* fp;

    fp = fopen(modelPath.c_str(), "rb");
    if (NULL == fp) {
        m_logger->printStdoutLog(Logger::LogLevel::Error, "Open file %s failed.", modelPath);
        return nullptr;
    }

    fseek(fp, 0, SEEK_END);
    m_params.m_model_size = ftell(fp);
    auto data = loadModelData(fp, 0, m_params.m_model_size);
    fclose(fp);
    return data;
}

std::shared_ptr<unsigned char> tensorrt::loadModelData(FILE* fp, size_t offset, size_t size) {
    int ret;

    if (NULL == fp) {
        return nullptr;
    }

    ret = fseek(fp, offset, SEEK_SET);

    if (ret != 0) {
        m_logger->printStdoutLog(Logger::LogLevel::Error, "blob seek failure.");
        return nullptr;
    }

    std::shared_ptr<unsigned char> data(new unsigned char[size], std::default_delete<unsigned char[]>());
    if (nullptr == data) {
        m_logger->printStdoutLog(Logger::LogLevel::Error, "blob malloc failure.");
        return nullptr;
    }

    ret = fread(data.get(), 1, size, fp);
    return data;
}


void tensorrt::loadModel(const std::string& modelPath) {
    if (modelPath.empty()) {
        throw std::runtime_error("modelPath is empty.");
    }
    m_params.m_model_data = loadModelFile(modelPath);

    initLibNvInferPlugins(&m_logger, "");

    m_params.m_runtime = nvinfer1::createInferRuntime(m_logger);
    if(m_params.m_runtime == nullptr) {
        m_logger->printStdoutLog(Logger::LogLevel::Error, "create inference runtime failure.");
        // throw std::runtime_error("create inference runtime failure.");
    }

    m_params.m_engine = m_params.m_runtime->deserializeCudaEngine(m_params.m_model_data, m_params.m_model_size);
    if(m_params.m_engine == nullptr) {
        m_logger->printStdoutLog(Logger::LogLevel::Error, "create cuda engine failure.");
        // throw std::runtime_error("create cuda engine failure.");
    }

    m_params.m_context = m_params.m_engine->createExecutionContext();
    if(m_params.m_context == nullptr) {
        m_logger->printStdoutLog(Logger::LogLevel::Error, "create inference context failure.");
        // throw std::runtime_error("create inference context failure.");
    }

    cudaStreamCreate(&m_params.m_stream);

#ifdef TRT_10
    m_params.m_num_bindings = m_params.m_engine->getNbIOTensors();
#else
    m_params.m_num_bindings = m_params.m_num_bindings = m_params.m_engine->getNbBindings();
#endif

    for (int i = 0; i < m_params.m_num_bindings; ++i) {
        TensorrtBinding binding;
        nvinfer1::Dims dims;

#ifdef TRT_10
        std::string        name  = m_params.m_engine->getIOTensorName(i);
        nvinfer1::DataType dtype = m_params.m_engine->getTensorDataType(name.c_str());
#else
        nvinfer1::DataType dtype = m_params.m_engine->getBindingDataType(i);
        std::string        name  = m_params.m_engine->getBindingName(i);
#endif

        binding.fmt = m_params.m_engine->getBindingFormat(i);

        binding.name  = name;
        binding.dsize = trt_datatype2size(dtype);

#ifdef TRT_10
        bool IsInput = m_params.m_engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
#else
        bool IsInput = m_params.m_engine->bindingIsInput(i);
#endif

        if (IsInput) {
#ifdef TRT_10
            dims = m_params.m_engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
            // set max opt shape
            m_params.m_context->setInputShape(name.c_str(), dims);
#else
            dims = m_params.m_engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            // set max opt shape
            m_params.m_context->setBindingDimensions(i, dims);
#endif

            binding.size = trt_get_size_by_dims(dims);
            binding.dims = dims;
            m_params.m_input_bindings.push_back(binding);

            m_params.m_num_inputs += 1;
        }
        else {
#ifdef TRT_10
            dims = m_params.m_context->getTensorShape(name.c_str());
#else
            dims = m_params.m_context->getBindingDimensions(i);
#endif

            binding.size = trt_get_size_by_dims(dims);
            binding.dims = dims;
            m_params.m_output_bindings.push_back(binding);
            m_params.m_num_outputs += 1;
        }
    }

}

// if kCHW4 is right?
int tensorrt::getInputShape(dnnInputShape& shape) {

    // format refer to : https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/namespacenvinfer1.html#ac3e115b1a2b1e578e8221ef99d27cd45
    if (m_params.m_input_bindings[0].fmt == kCHW4) {
        shape.height = m_params.m_input_bindings[0].dims.d[2];
        shape.width  = m_params.m_input_bindings[0].dims.d[3];
        shape.channel = m_params.m_input_bindings[0].dims.d[1];
    }
    else {
        shape.height = m_params.m_input_bindings[0].dims.d[1];
        shape.width  = m_params.m_input_bindings[0].dims.d[2];
        shape.channel = m_params.m_input_bindings[0].dims.d[3];
    }
    return 0;
}


void tensorrt::trt_make_pipe(bool warmup) {
    for (auto& bindings : m_params.m_inputs_bingdins) {
        void* d_ptr;
        auto error_code = cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, m_params.m_stream);
        if (error_code != cudaSuccess) {
            m_logger->printStdoutLog(Logger::LogLevel::Error, "cudaMallocAsync failure, error text: %s", cudaGetErrorString(error_code));
        }
        m_params.device_ptrs.push_back(d_ptr);

#ifdef TRT_10
        auto name = bindings.name.c_str();
        m_params.m_context->setInputShape(name, bindings.dims);
        m_params.m_context->setTensorAddress(name, d_ptr);
#endif
    }

    for (auto& bindings : this->output_bindings) {
        void *d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;

        auto error_code = cudaMallocAsync(&d_ptr, size, m_params.m_stream);
        if (error_code != cudaSuccess) {
            m_logger->printStdoutLog(Logger::LogLevel::Error, "cudaMallocAsync failure, error text: %s", cudaGetErrorString(error_code));
        }

        error_code = cudaHostAlloc(&h_ptr, size, 0);
        if (error_code != cudaSuccess) {
            m_logger->printStdoutLog(Logger::LogLevel::Error, "cudaMallocAsync failure, error text: %s", cudaGetErrorString(error_code));
        }

        m_params.m_device_ptrs.push_back(d_ptr);
        m_params.m_host_ptrs.push_back(h_ptr);

#ifdef TRT_10
        auto name = bindings.name.c_str();
        m_params.m_context->setTensorAddress(name, d_ptr);
#endif
    }

    if (warmup) {
        for (int i = 0; i < MODEL_WARM_UP_TIMES; i++) {
            for (auto& bindings : m_params.m_input_bindings) {
                size_t size  = bindings.size * bindings.dsize;
                void*  h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                CHECK(cudaMemcpyAsync(m_params.m_device_ptrs[0], h_ptr, size, cudaMemcpyHostToDevice, m_params.m_stream));
                free(h_ptr);
            }
            runInference();
        }
        m_logger->printStdoutLog(Logger::LogLevel::Debug, "model warmup %d times", MODEL_WARM_UP_TIMES);
    }
}


// Currently only supports floating-point types
int getOutputQuantParams(std::vector<int32_t>& zeroPoints, std::vector<float>& scales) {
    return -1;
}

int pushInputData(dnnInput& inputData) {
    if (inputData.size == 0) {
        m_logger->printStdoutLog(BspLogger::LogLevel::Error, "inputData.buf is empty.");
        return -1;
    }

    size_t inputSize = inputData.size;
    auto error_code = cudaMemcpyAsync(m_params.m_device_ptrs[0], inputData.buf.data(), inputSize, cudaMemcpyHostToDevice, m_params.m_stream));
    if (error_code != cudaSuccess) {
        m_logger->printStdoutLog(Logger::LogLevel::Error, "cudaMemcpyAsync failure, error text: %s", cudaGetErrorString(error_code));
    }

#ifdef TRT_10
    auto name = m_params.m_input_bindings[0].name.c_str();
    m_params.m_context->setInputShape(name, nvinfer1::Dims{4, {1, inputData.shape.channel, inputData.shape.height, inputData.shape.width}});
    m_params.m_context->setTensorAddress(name, m_params.m_device_ptrs[0]);
#else
    m_params.m_context->setBindingDimensions(0, nvinfer1::Dims{4, {1, inputData.shape.channel, inputData.shape.height, inputData.shape.width}});
#endif

    return 0;
}

int popOutputData(std::vector<dnnOutput>& outputVector) {
    // if (m_params.m_outputs.size() != m_params.m_io_num.n_output) {
    //     m_params.m_outputs.resize(m_params.m_io_num.n_output);
    //     for (int i = 0; i < m_params.m_io_num.n_output; i++) {
    //         std::memset(&m_params.m_outputs[i], 0, sizeof(rknn_output));
    //         m_params.m_outputs[i].index = i;
    //         m_params.m_outputs[i].want_float = 0;
    //     }
    // }

    // if (outputVector.size() != m_params.m_io_num.n_output) {
    //     outputVector.resize(m_params.m_io_num.n_output);
    // }

    // // Get Output
    // int ret = rknn_outputs_get(m_params.m_rknnCtx, m_params.m_io_num.n_output,
    //                 m_params.m_outputs.data(), nullptr);

    // for (int i = 0; i < m_params.m_io_num.n_output; i++) {
    //     outputVector[i].index = m_params.m_outputs[i].index;
    //     outputVector[i].buf = m_params.m_outputs[i].buf;
    //     outputVector[i].size = m_params.m_outputs[i].size;
    //     outputVector[i].dataType = m_params.m_outputs[i].want_float ? "float32" : "int8";
    // }
    
    // return ret;


    for (int i = 0; i < m_params.m_num_outputs; i++) {
        outputVector[i].index = i;
        outputVector[i].buf = m_params.m_host_ptrs[i];
        outputVector[i].size = m_params.m_output_bindings[i].size;
        outputVector[i].dataType = "float32";
    }

    return 0;
}


int tensorrt::runInference() {
#ifdef TRT_10
    m_params.m_context->enqueueV3(m_params.m_stream);
#else
    m_params.m_context->enqueueV2(m_params.m_device_ptrs.data(), m_params.m_stream, nullptr);
#endif
    for (int i = 0; i < m_params.m_num_outputs; i++) {
        size_t osize = m_params.m_output_bindings[i].size * m_params.m_output_bindings[i].dsize;
        auto error_code = cudaMemcpyAsync(m_params.m_host_ptrs[i], m_params.m_device_ptrs[i + m_params.m_num_inputs], osize, cudaMemcpyDeviceToHost, m_params.m_stream);
        if (error_code != cudaSuccess) {
            m_logger->printStdoutLog(Logger::LogLevel::Error, "cudaMemcpyAsync failure, error text: %s", cudaGetErrorString(error_code));
        }
    }
    cudaStreamSynchronize(m_params.m_stream);

    return 0;
}


} // namespace dnn_engine
