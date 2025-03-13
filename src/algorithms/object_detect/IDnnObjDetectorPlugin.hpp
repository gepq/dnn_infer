#ifndef __IDNN_OBJDETECTOR_PLUGIN_HPP__
#define __IDNN_OBJDETECTOR_PLUGIN_HPP__

#include "dnn_engines/IDnnEngine.hpp"
#include <memory>
#include <string>
#include <vector>
#include <any>

namespace dnn_algorithm {

using namespace dnn_engine;

struct ObjDetectInput {
    std::string handleType{"opencv4"};
    std::any imageHandle;
};

template <typename T>
// keep the parameter name consistent with OpenCv4
// (corresponding to x, y, w, h in YOLO)
struct bboxRect {
    T left;
    T right;
    T top;
    T bottom;
};

struct ObjDetectOutput {
    bboxRect<int> bbox{};
    float score{0.0};
    std::string label{};
};

struct ObjDetectParams {
    size_t model_input_width;
    size_t model_input_height;
    size_t model_input_channel;
    float conf_threshold;
    float nms_threshold;
    float scale_width;
    float scale_height;
    bboxRect<int> pads;
    // quantization params
    std::vector<int32_t> quantize_zero_points;
    std::vector<float> quantize_scales;
};

class IDnnObjDetectorPlugin {
public:
    virtual int preProcess(ObjDetectParams& params, ObjDetectInput& inputData, IDnnEngine::dnnInput& outputData) = 0;
    virtual int postProcess(const std::string& labelTextPath, const ObjDetectParams& params,
                    std::vector<IDnnEngine::dnnOutput>& inputData, std::vector<ObjDetectOutput>& outputData) = 0;
    
    IDnnObjDetectorPlugin() = default; // reserved for plugin interface
    virtual ~IDnnObjDetectorPlugin() = default;
};


} // namespace dnn_algorithm


#define CREATE_PLUGIN_INSTANCE(PLUGIN_CLASS) \
    extern "C" dnn_algorithm::IDnnObjDetectorPlugin* create() { \
        return new PLUGIN_CLASS(); \
    } \
    extern "C" void destroy(dnn_algorithm::IDnnObjDetectorPlugin* plugin) { \
        delete plugin; \
    }


#endif // __IDNN_OBJDETECTOR_PLUGIN_HPP__
