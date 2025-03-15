#include "dnnObjDetector.hpp"
#include "IDnnObjDetectorPlugin.hpp"
#include <dlfcn.h>

namespace dnn_algorithm {


dnnObjDetector::dnnObjDetector(const std::string& dnnType, const std::string& pluginPath, const std::string& labelTextPath):
                                m_logger{std::make_unique<Logger>("dnnObjDetector")},
                                m_dnnEngine{IDnnEngine::create(dnnType)},
                                m_labelTextPath{labelTextPath} {
    if (!m_dnnEngine) {
        m_logger->printStdoutLog(Logger::LogLevel::Error, "Failed to create DNN engine.");
        throw std::runtime_error("Failed to create DNN engine.");
    }

    if (pluginPath.empty()) {
        m_logger->printStdoutLog(Logger::LogLevel::Error, "pluginPath is empty.");
        return;
    }

    // Dynamically load the algorithm plugin library
    m_pluginLibraryHandle = std::shared_ptr<void>(dlopen(pluginPath.c_str(), RTLD_LAZY), dlclose);

    if (m_pluginLibraryHandle == nullptr) {
        m_logger->printStdoutLog(Logger::LogLevel::Error, "Failed to open plugin library: {}", dlerror());
        throw std::runtime_error(dlerror());
    }

    // Retrieve the create interface from the plugin library
    auto create = reinterpret_cast<IDnnObjDetectorPlugin* (*)()>(dlsym(m_pluginLibraryHandle.get(), "create"));
    if (create == nullptr) {
        m_logger->printStdoutLog(Logger::LogLevel::Error, "Failed to load symbol create: {}", dlerror());
        throw std::runtime_error(dlerror());
    }

    // Call the create interface to obtain the algorithm object from the plugin library
    m_dnnPluginHandle.reset(create());

    if (m_dnnPluginHandle == nullptr) {
        m_logger->printStdoutLog(Logger::LogLevel::Error, "Failed to create plugin instance.");
        throw std::runtime_error("Failed to create plugin instance.");
    }

}


dnnObjDetector::~dnnObjDetector() {
    if (m_dnnPluginHandle != nullptr) {
        auto destroy = reinterpret_cast<void (*)(IDnnObjDetectorPlugin*)>(dlsym(m_pluginLibraryHandle.get(), "destroy"));
        if (destroy == nullptr) {
            m_logger->printStdoutLog(Logger::LogLevel::Error, "Failed to load symbol destroy: {}", dlerror());
            // throw std::runtime_error(dlerror());
        }
        /* When calling m_dnnPluginHandle.reset(), the current resources will be implicitly destroyed. */
        // else {
        //     destroy(m_dnnPluginHandle.get()); // call the destory interface
        // }
        m_dnnPluginHandle.reset();
    }

    if (m_pluginLibraryHandle != nullptr) {
        dlclose(m_pluginLibraryHandle.get());
        m_pluginLibraryHandle.reset();
    }
}

void dnnObjDetector::loadModel(const std::string& modelPath) {
    m_dnnEngine->loadModel(modelPath);
}

void dnnObjDetector::pushInputData(std::shared_ptr<ObjDetectInput> dataInput) {
    m_dataInput = dataInput;
}

std::vector<ObjDetectOutput>& dnnObjDetector::popOutputData() {
    return m_dataOutputVector;
}

int dnnObjDetector::defaultPreProcess(ObjDetectInput& inputData, IDnnEngine::dnnInput& outputData) {
    return 0;
}

int dnnObjDetector::defaultPostProcess(const std::string& labelTextPath, const ObjDetectParams& params,
        std::vector<IDnnEngine::dnnOutput>& inputData, std::vector<ObjDetectOutput>& outputData) {
    return 0;
}

int dnnObjDetector::runObjDetect(ObjDetectParams& params) {
    // In case the algorithm plugin is not provided
    if ((m_pluginLibraryHandle == nullptr) || (m_dnnPluginHandle == nullptr)) {
        m_logger->printStdoutLog(Logger::LogLevel::Error, "pluginLibraryHandle is nullptr.");
        IDnnEngine::dnnInput dnn_input_tensor{};
        defaultPreProcess(*m_dataInput, dnn_input_tensor);
        m_dnnEngine->pushInputData(dnn_input_tensor);
        m_dnnEngine->runInference();
        std::vector<IDnnEngine::dnnOutput> dnn_output_vector{};
        m_dnnEngine->popOutputData(dnn_output_vector);
        return defaultPostProcess(m_labelTextPath, params,
                    dnn_output_vector, m_dataOutputVector);
    }

    IDnnEngine::dnnInput dnn_input_tensor{};
    m_dnnPluginHandle->preProcess(params, *m_dataInput, dnn_input_tensor);
    m_dnnEngine->pushInputData(dnn_input_tensor);
    m_dnnEngine->runInference();
    std::vector<IDnnEngine::dnnOutput> dnn_output_vector{};
    m_dnnEngine->popOutputData(dnn_output_vector);
    m_logger->printStdoutLog(Logger::LogLevel::Info, "dnn_output_vector.size(): {}", dnn_output_vector.size());
    for (const auto& dnn_output : dnn_output_vector) {
        m_logger->printStdoutLog(Logger::LogLevel::Info, "dnn_output.index: {}, dnn_output.size: {}", dnn_output.index, dnn_output.size);
        m_logger->printStdoutLog(Logger::LogLevel::Info, "dnn_output.dataType: {}", dnn_output.dataType);
    }
    m_dataOutputVector.clear();
    return m_dnnPluginHandle->postProcess(m_labelTextPath, params,
                dnn_output_vector, m_dataOutputVector);
}


} // namespace dnn_algorithm
