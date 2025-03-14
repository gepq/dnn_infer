#include "yolov5.hpp"
#include <opencv2/opencv.hpp>
#include <memory>
#include <any>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <set>
#include <fstream>

namespace dnn_algorithm {

int yolov5::preProcess(ObjDetectParams& params, ObjDetectInput& inputData, IDnnEngine::dnnInput& outputData) {
    // This plugin only processes input in OpenCV 4 format
    if (inputData.handleType.compare("opencv4") != 0) {
        throw std::invalid_argument("Only opencv4 is supported.");
    }

    auto orig_image_ptr = std::any_cast<std::shared_ptr<cv::Mat>>(inputData.imageHandle);
    if (orig_image_ptr == nullptr) {
        throw std::invalid_argument("inputData.imageHandle is nullptr.");
    }
    cv::Mat orig_image = *orig_image_ptr;
    cv::Mat rgb_image;
    cv::cvtColor(orig_image, rgb_image, cv::COLOR_BGR2RGB);

    // Resize the original image to match the model input dimensions
    cv::Size target_size(params.model_input_width, params.model_input_height);
    cv::Mat padded_image(target_size.height, target_size.width, CV_8UC3);
    float min_scale = std::min(params.scale_width, params.scale_height); // params.scale_width = model_input_width/orig_image_width
    params.scale_width = min_scale;
    params.scale_height = min_scale;
    cv::Mat resized_image;
    cv::resize(rgb_image, resized_image, cv::Size(), min_scale, min_scale);

    // Pad the resized image with gray to match the model input size along the insufficient dimension
    int pad_width = target_size.width - resized_image.cols;
    int pad_height = target_size.height - resized_image.rows;
    params.pads.left = pad_width / 2;
    params.pads.right = pad_width - params.pads.left;
    params.pads.top = pad_height / 2;
    params.pads.bottom = pad_height - params.pads.top;
    cv::copyMakeBorder(resized_image, padded_image, params.pads.top, params.pads.bottom, params.pads.left, params.pads.right, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));

    // The output of the preprocessing is the image resized and padded to match the model input size
    outputData.index = 0;
    outputData.shape.width = params.model_input_width;
    outputData.shape.height = params.model_input_height;
    outputData.shape.channel = params.model_input_channel;
    outputData.size = padded_image.total() * padded_image.elemSize();
    if (outputData.buf.size() != outputData.size) {
        outputData.buf.resize(outputData.size);
    }
    outputData.dataType = "UINT8";
    std::memcpy(outputData.buf.data(), padded_image.data, outputData.size);
    return 0;
}

int yolov5::postProcess(const std::string& labelTextPath, const ObjDetectParams& params,
        std::vector<IDnnEngine::dnnOutput>& inputData, std::vector<ObjDetectOutput>& outputData) {
    if (inputData.size() != YOLOV5_OUTPUT_BATCH || labelTextPath.empty()) {
        throw std::invalid_argument("The size of inputData is not equal to RKNN_YOLOV5_OUTPUT_BATCH or labelTextPath is empty.");
    }

    int ret = initLabelMap(labelTextPath);
    if (ret != 0) {
        return ret;
    }
    return runPostProcess(params, inputData, outputData);
}


// Load the label list file
int yolov5::initLabelMap(const std::string& labelMapPath) {
    if (m_labelMapInited) {
        return 0;
    }

    if (labelMapPath.empty()) {
        return -1;
    }

    std::cout << "loading label path: " << labelMapPath << std::endl;

    std::ifstream labelFile(labelMapPath);
    if (!labelFile.is_open()) {
        std::cerr << "Failed to open label map file: " << labelMapPath << std::endl;
        return -1;
    }

    std::string line;
    while (std::getline(labelFile, line)) {
        m_labelMap.push_back(line);
    }

    labelFile.close();
    m_labelMapInited = true;
    return 0;
}

/**
 * Sort the floating-point vector <input> in descending order 
 * and simultaneously update the corresponding index vector <indices>
 */
void yolov5::inverseSortWithIndices(std::vector<float> &input, std::vector<int> &indices) {
    // Create a vector of pairs (value, index)
    std::vector<std::pair<float, int>> value_index_pairs;
    for (size_t i = 0; i < input.size(); ++i) {
        value_index_pairs.emplace_back(input[i], indices[i]);
    }

    // Sort the pairs based on the value in descending order
    std::sort(value_index_pairs.begin(), value_index_pairs.end(), [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
        return a.first >= b.first;
    });

    // Update the input and indices vectors based on the sorted pairs
    for (size_t i = 0; i < value_index_pairs.size(); ++i) {
        input[i] = value_index_pairs[i].first;
        indices[i] = value_index_pairs[i].second;
    }
}

// non-maximum suppression
int yolov5::nms(int validCount, std::vector<float> &filterBoxes, std::vector<int> classIds,
        std::vector<int> &order, int filterId, float threshold) {
    for (int i = 0; i < validCount; ++i) {
        int n = order[i];
        if (n == -1 || classIds[n] != filterId) {
            continue;
        }

        for (int j = i + 1; j < validCount; ++j) {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId) {
                continue;
            }

            bboxRect<float> bbox1;
            bbox1.left = filterBoxes[n * 4 + 0];
            bbox1.top = filterBoxes[n * 4 + 1];
            bbox1.right = filterBoxes[n * 4 + 0] + filterBoxes[n * 4 + 2];
            bbox1.bottom = filterBoxes[n * 4 + 1] + filterBoxes[n * 4 + 3];
            bboxRect<float> bbox2;
            bbox2.left = filterBoxes[m * 4 + 0];
            bbox2.top = filterBoxes[m * 4 + 1];
            bbox2.right = filterBoxes[m * 4 + 0] + filterBoxes[m * 4 + 2];
            bbox2.bottom = filterBoxes[m * 4 + 1] + filterBoxes[m * 4 + 3];
            float iou = calculateOverlap(bbox1, bbox2);

            if (iou > threshold) {
                order[j] = -1;
            }
        }
  }
  return 0;
}

int yolov5::runPostProcess(const ObjDetectParams& params, std::vector<IDnnEngine::dnnOutput>& inputData, std::vector<ObjDetectOutput>& outputData) {
    std::vector<float> filterBoxes;
    std::vector<float> objScores;
    std::vector<int> classId;
    int validBoxNum = 0;

    for (int i = 0; i < inputData.size(); i++) {
        int stride = BASIC_STRIDE * (1 << i);
        validBoxNum += doProcess(i, params, stride, inputData[i], filterBoxes, objScores, classId);
    }

    if (validBoxNum <= 0) {
        return 0;
    }

    std::vector<int> indexArray(validBoxNum);
    std::iota(indexArray.begin(), indexArray.end(), 0);
    inverseSortWithIndices(objScores, indexArray);
    std::set<int> class_set(std::begin(classId), std::end(classId));

    for(auto filterId : class_set) {
        nms(validBoxNum, filterBoxes, classId, indexArray, filterId, params.nms_threshold);
    }

    int detectObjCount = 0;
    for (int i = 0; i < validBoxNum; i++) {
        ObjDetectOutput outputBox;
        if (indexArray[i] == -1 || detectObjCount >= MAX_OBJ_NUM) {
            continue;
        }

        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0] - params.pads.left;
        float y1 = filterBoxes[n * 4 + 1] - params.pads.top;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        outputBox.bbox.left = (int)(clamp(x1, 0, params.model_input_width) / params.scale_width);
        outputBox.bbox.top = (int)(clamp(y1, 0, params.model_input_height) / params.scale_height);
        outputBox.bbox.right = (int)(clamp(x2, 0, params.model_input_width) / params.scale_width);
        outputBox.bbox.bottom = (int)(clamp(y2, 0, params.model_input_height) / params.scale_height);
        outputBox.score = objScores[i];
        int id = classId[n];
        outputBox.label = m_labelMap[id];
        outputData.push_back(outputBox);

        detectObjCount++;
    }

    return 0;
}

int8_t yolov5::qauntFP32ToAffine(float fp32, int8_t zp, float scale) {
    float dst_val = (fp32 / scale) + zp;
    int8_t res = (int8_t)clip(dst_val, -128, 127);
    return res;
}

/**
 * The current plugin only supports quantized networks
 * TUDO: Floating-point model inference
 */
int yolov5::doProcess(const int idx, const ObjDetectParams& params, int stride, IDnnEngine::dnnOutput& inputData,
            std::vector<float>& bboxes, std::vector<float>& objScores, std::vector<int>& classId) {
    int validCount = 0;
    int grid_h = params.model_input_height / stride;
    int grid_w = params.model_input_width / stride;
    int grid_len = grid_h * grid_w;

    // Convert the floating-point confidence threshold to a quantized int8_t value for direct comparison with the model's int8 output
    int8_t thres_i8 = qauntFP32ToAffine(params.conf_threshold, params.quantize_zero_points[idx],
                        params.quantize_scales[idx]);

    int8_t *input_buf = (int8_t *)inputData.buf;

    for (int a = 0; a < YOLOV5_ANCHORS_NUM; a++) {
        for (int i = 0; i < grid_h; i++) {
            for (int j = 0; j < grid_w; j++) {
                int8_t box_confidence = input_buf[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
                if (box_confidence >= thres_i8) {
                    int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
                    int8_t *in_ptr = input_buf + offset;
                    float box_x = (deqauntAffineToFP32(*in_ptr, params.quantize_zero_points[idx], params.quantize_scales[idx])) * 2.0 - 0.5;
                    float box_y = (deqauntAffineToFP32(in_ptr[grid_len], params.quantize_zero_points[idx], params.quantize_scales[idx])) * 2.0 - 0.5;
                    float box_w = (deqauntAffineToFP32(in_ptr[2 * grid_len], params.quantize_zero_points[idx], params.quantize_scales[idx])) * 2.0;
                    float box_h = (deqauntAffineToFP32(in_ptr[3 * grid_len], params.quantize_zero_points[idx], params.quantize_scales[idx])) * 2.0;
                    box_x = (box_x + j) * (float) stride;
                    box_y = (box_y + i) * (float) stride;
                    box_w = box_w * box_w * (float) m_anchorVec[idx][a * 2];
                    box_h = box_h * box_h * (float) m_anchorVec[idx][a * 2 + 1];
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    int8_t maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < OBJ_CLASS_NUM; ++k) {
                        int8_t prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs) {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }

                    if (maxClassProbs > thres_i8) {
                        objScores.push_back((deqauntAffineToFP32(maxClassProbs, params.quantize_zero_points[idx], params.quantize_scales[idx]))
                                    * (deqauntAffineToFP32(box_confidence, params.quantize_zero_points[idx], params.quantize_scales[idx])));
                        classId.push_back(maxClassId);
                        validCount++;
                        bboxes.push_back(box_x);
                        bboxes.push_back(box_y);
                        bboxes.push_back(box_w);
                        bboxes.push_back(box_h);
                    }
                }
            }
        }
    }
    return validCount;
}


} // namespace dnn_algorithm
