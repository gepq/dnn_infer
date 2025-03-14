#ifndef __YOLOV5_HPP__
#define __YOLOV5_HPP__

#include "algorithms/object_detect/IDnnObjDetectorPlugin.hpp"
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <cmath>


namespace dnn_algorithm {

/**
 * TUDO: 
 * 1. Change some of the parameters of the post process functions to member variables.
 * 2. Extract the series of static member functions(such as nms), from the yolov5 class and refactor them into a separate utility class.
 */
class yolov5 : public IDnnObjDetectorPlugin
{
public:
    static constexpr int YOLOV5_OUTPUT_BATCH = 3; // The number of output tensors of the YOLOv5 model
    static constexpr int YOLOV5_ANCHORS_NUM = 3; // The number of anchors per grid

    static constexpr int BASIC_STRIDE = 8;
    static constexpr int MAX_OBJ_NUM = 64;
    static constexpr int OBJ_CLASS_NUM = 80;
    static constexpr int PROP_BOX_SIZE = 5 + OBJ_CLASS_NUM;

    yolov5() = default;
    ~yolov5() = default;

    int preProcess(ObjDetectParams& params, ObjDetectInput& inputData, IDnnEngine::dnnInput& outputData) override;
    int postProcess(const std::string& labelTextPath, const ObjDetectParams& params,
            std::vector<IDnnEngine::dnnOutput>& inputData, std::vector<ObjDetectOutput>& outputData)  override;

private:
    int initLabelMap(const std::string& labelMapPath);
    int runPostProcess(const ObjDetectParams& params, std::vector<IDnnEngine::dnnOutput>& inputData,
            std::vector<ObjDetectOutput>& outputData);

    int doProcess(const int idx, const ObjDetectParams& params, int stride, IDnnEngine::dnnOutput& inputData,
        std::vector<float>& bboxes, std::vector<float>& objScores, std::vector<int>& classId);

    static void inverseSortWithIndices(std::vector<float> &input, std::vector<int> &indices);

    static int nms(int validCount, std::vector<float> &filterBoxes, std::vector<int> classIds,
            std::vector<int> &order, int filterId, float threshold);

    static inline int clamp(float val, int min, int max) {
        return val > min ? (val < max ? val : max) : min;
    }

    template <typename T>
    static float calculateOverlap(const bboxRect<T>& bbox1, const bboxRect<T>& bbox2) {
        float w = fmax(0.f, fmin(bbox1.right, bbox2.right) - fmax(bbox1.left, bbox2.left) + 1.0);
        float h = fmax(0.f, fmin(bbox1.bottom, bbox2.bottom) - fmax(bbox1.top, bbox2.top) + 1.0);
        float i = w * h;
        float u = (bbox1.right - bbox1.left + 1.0) * (bbox1.bottom - bbox1.top + 1.0)
                + (bbox2.right - bbox2.left + 1.0) * (bbox2.bottom - bbox2.top + 1.0) - i;

        return u <= 0.f ? 0.f : (i / u);
    }


    static int8_t qauntFP32ToAffine(float fp32, int8_t zp, float scale);

    static float deqauntAffineToFP32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

    static inline int32_t clip(int32_t val, int32_t min, int32_t max) {
        float f = val <= min ? min : (val >= max ? max : val);
        return f;
    }

private:
    // YoloPostProcess m_yoloPostProcess;
    std::vector<std::string> m_labelMap;
    bool m_labelMapInited{false};
    std::vector<std::array<const int, 6>> m_anchorVec = { // yolov5 anchors
        {10, 13, 16, 30, 33, 23},
        {30, 61, 62, 45, 59, 119},
        {116, 90, 156, 198, 373, 326}
    };
};



} // namespace dnn_algorithm

CREATE_PLUGIN_INSTANCE(dnn_algorithm::yolov5)

#endif // __YOLOV5_HPP__
