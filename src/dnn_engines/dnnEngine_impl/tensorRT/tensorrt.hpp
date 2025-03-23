#ifndef __TENSORRT_HPP__
#define __TENSORRT_HPP__

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <NvInferPlugin.h>
#include <NvInfer.h>
#include "dnn_engines/IDnnEngine.hpp"
#include "common/Logger.hpp"

namespace dnn_engine {

using namespace common;

#define MODEL_WARM_UP_TIMES 10

struct TensorrtBinding {
    size_t         size  = 1;
    size_t         dsize = 1;
    nvinfer1::TensorFormat fmt;
    nvinfer1::Dims dims;
    std::string    name;
};

struct TensorrtParams {
    std::shared_ptr<char> m_model_data{nullptr}; // if unsigned char is ok?
    // rknn_context m_rknnCtx;
    int32_t m_model_size;

    int m_num_bindings;
    int m_num_inputs  = 0; // = m_input_bindings.size()
    int m_num_outputs = 0; // = m_output_bindings.size()
    std::vector<TensorrtBinding> m_input_bindings;
    std::vector<TensorrtBinding> m_output_bindings;

    // store inference result vector
    std::vector<void*>   m_host_ptrs;   // point to CPU memory
    std::vector<void*>   m_device_ptrs; // point to GPU memory

    std::shared_ptr<nvinfer1::ICudaEngine> m_cuda_engine;
    std::shared_ptr<nvinfer1::IRuntime> m_runtime;
    std::shared_ptr<nvinfer1::IExecutionContext> m_context;
    cudaStream_t m_stream;
};

// unify nvinfer1::ILogger and common::Logger
class TensorrtLogger : public nvinfer1::ILogger {
public:
    explicit TensorrtLogger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kVERBOSE):
            m_reportableSeverity(severity), 
            m_logger{std::make_unique<Logger>("tensorrt")}{}
    ~TensorrtLogger() = default;

    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        if (severity > m_reportableSeverity) {
            return;
        }
        switch (severity) {
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            case nvinfer1::ILogger::Severity::kERROR:
                m_logger->printStdoutLog(Logger::LogLevel::Error, "%s", msg);
                break;
            case nvinfer1::ILogger::Severity::kWARNING:
                m_logger->printStdoutLog(Logger::LogLevel::Warn, "%s", msg);
                break;
            case nvinfer1::ILogger::Severity::kINFO:
                m_logger->printStdoutLog(Logger::LogLevel::Info, "%s", msg);
                break;
            default:
                // std::cerr << "VERBOSE: ";
                m_logger->printStdoutLog(Logger::LogLevel::Debug, "%s", msg);
                break;
        }
    }

    template<typename... Args>
    void printStdoutLog(LogLevel level, std::string_view sv, const Args &... args){
        m_logger->printStdoutLog(level, sv, args...);
    }

    template<typename... Args>
    void printFileLog(LogLevel level, std::string_view sv, const Args &... args){
        m_logger->printStdoutLog(level, sv, args...);
    }


    template<typename... Args>
    void printAsyncFileLog(LogLevel level, std::string_view sv, const Args &... args){
        m_logger->printStdoutLog(level, sv, args...);
    }

private:
    std::unique_ptr<Logger> m_logger;
    nvinfer1::ILogger::Severity m_reportableSeverity;
}

class tensorrt : public IDnnEngine {
public:
    explicit tensorrt();
    tensorrt(const tensorrt&) = delete;
    tensorrt& operator=(const tensorrt&) = delete;
    tensorrt(tensorrt&&) = delete;
    tensorrt& operator=(tensorrt&&) = delete;
    ~tensorrt();

    void loadModel(const std::string& modelPath) override;

    int getInputShape(dnnInputShape& shape) override;

    int getOutputQuantParams(std::vector<int32_t>& zeroPoints, std::vector<float>& scales) override;

    int pushInputData(dnnInput& inputData) override;

    int popOutputData(std::vector<dnnOutput>& outputVector) override;

    int runInference() override;

private:
    std::shared_ptr<unsigned char> loadModelFile(const std::string& modelPath);
    std::shared_ptr<unsigned char> loadModelData(FILE* fp, size_t offset, size_t size);

    void trt_make_pipe(bool warmup);

    inline int trt_datatype2size(const nvinfer1::DataType& dataType) {
        switch (dataType) {
            case nvinfer1::DataType::kFLOAT:
                return 4;
            case nvinfer1::DataType::kHALF:
                return 2;
            case nvinfer1::DataType::kINT32:
                return 4;
            case nvinfer1::DataType::kINT8:
                return 1;
            case nvinfer1::DataType::kBOOL:
                return 1;
            default:
                return 4;
        }
    }

    inline int trt_get_size_by_dims(const nvinfer1::Dims& dims) {
        int size = 1;
        for (int i = 0; i < dims.nbDims; i++) {
            size *= dims.d[i];
        }
        return size;
    }



private:
    TensorrtParams m_params{};
    std::unique_ptr<TensorrtLogger> m_logger;

};


} // namespace dnn_engine


#endif // __TENSORRT_HPP__
