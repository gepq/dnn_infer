#ifndef __DNN_OBJDETECTOR_HPP__
#define __DNN_OBJDETECTOR_HPP__

#include "IDnnObjDetectorPlugin.hpp"
#include "dnn_engines/IDnnEngine.hpp"
#include "common/Logger.hpp"
#include <memory>
#include <string>
#include <vector>


namespace dnn_algorithm {

using namespace common;
using namespace dnn_engine;

class dnnObjDetector {
public:
    dnnObjDetector(const std::string& dnnType, const std::string& pluginPath, const std::string& labelTextPath);

    virtual ~dnnObjDetector();

    void loadModel(const std::string& modelPath);

    int getInputShape(IDnnEngine::dnnInputShape& shape) {
        return m_dnnEngine->getInputShape(shape);
    }

    int getOutputQuantParams(std::vector<int32_t>& zeroPoints, std::vector<float>& scales) {
        return m_dnnEngine->getOutputQuantParams(zeroPoints, scales);
    }

    void pushInputData(std::shared_ptr<ObjDetectInput> dataInput);

    std::vector<ObjDetectOutput>& popOutputData();

    int runObjDetect(ObjDetectParams& params);

private:
    int defaultPreProcess(ObjDetectInput& inputData, IDnnEngine::dnnInput& outputData);
    int defaultPostProcess(const std::string& labelTextPath, const ObjDetectParams& params,
            std::vector<IDnnEngine::dnnOutput>& inputData, std::vector<ObjDetectOutput>& outputData);

private:
    std::shared_ptr<void> m_pluginLibraryHandle{nullptr};
    std::shared_ptr<IDnnObjDetectorPlugin> m_dnnPluginHandle{nullptr};
    std::unique_ptr<IDnnEngine> m_dnnEngine{nullptr};
    std::unique_ptr<Logger> m_logger{nullptr};
    std::shared_ptr<ObjDetectInput> m_dataInput{nullptr};
    std::vector<ObjDetectOutput> m_dataOutputVector;
    std::string m_labelTextPath;

};


} // namespace dnn_algorithm


#endif // __DNN_OBJDETECTOR_HPP__
