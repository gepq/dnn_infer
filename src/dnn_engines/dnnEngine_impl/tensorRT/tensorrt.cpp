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
            m_params.m_num_inputs += 1;

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




} // namespace dnn_engine
