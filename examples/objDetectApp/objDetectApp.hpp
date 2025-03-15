#ifndef __OBJDETECTAPP_HPP__
#define __OBJDETECTAPP_HPP__

#include "common/Logger.hpp"
#include "common/ArgParser.hpp"
#include "algorithms/object_detect/dnnObjDetector.hpp"
#include <memory>
#include <string>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>



namespace example {

using namespace dnn_algorithm;

class ObjDetectApp {
public:
    static constexpr char LOG_TAG[] {"[ObjDetectApp]: "};

    ObjDetectApp(common::ArgParser&& args) : 
        m_args(std::move(args)),
        m_logger{std::make_unique<common::Logger>("ObjDetectApp")},
        m_colors{
            cv::Scalar(255, 0, 0),    // Blue
            cv::Scalar(0, 255, 0),    // Green
            cv::Scalar(0, 0, 255),    // Red
            cv::Scalar(255, 255, 0),  // Cyan
            cv::Scalar(255, 0, 255),  // Magenta
            cv::Scalar(0, 255, 255),  // Yellow
            cv::Scalar(128, 0, 0),    // Maroon
            cv::Scalar(0, 128, 0),    // Olive
            cv::Scalar(0, 0, 128),    // Navy
            cv::Scalar(128, 128, 0),  // Teal
            cv::Scalar(128, 0, 128),  // Purple
            cv::Scalar(0, 128, 128)   // Aqua
        } {
        
        m_logger->setPattern();

        std::string dnnType;
        std::string pluginPath;
        std::string labelTextPath;
        m_args.getOptionVal("--dnnType", dnnType);
        m_args.getOptionVal("--pluginPath", pluginPath);
        m_args.getOptionVal("--labelTextPath", labelTextPath);
        m_dnnObjDetector = std::make_unique<dnn_algorithm::dnnObjDetector>(dnnType, pluginPath, labelTextPath);

        std::string modelPath;
        m_args.getOptionVal("--modelPath", modelPath);
        m_dnnObjDetector->loadModel(modelPath);
        std::string imagePath;
        m_args.getOptionVal("--imagePath", imagePath);
        m_orig_image_ptr = std::make_shared<cv::Mat>(cv::imread(imagePath, cv::IMREAD_COLOR));
    }

    ObjDetectApp(const ObjDetectApp&) = delete;
    ObjDetectApp& operator=(const ObjDetectApp&) = delete;
    ObjDetectApp(ObjDetectApp&&) = delete;
    ObjDetectApp& operator=(ObjDetectApp&&) = delete;

    ~ObjDetectApp() {
        m_dnnObjDetector.reset();

        m_logger->printStdoutLog(common::Logger::LogLevel::Debug, "{} ObjDetectApp::~ObjDetectApp()", LOG_TAG);
        m_logger->printFileLog(common::Logger::LogLevel::Debug, "{} ObjDetectApp::~ObjDetectApp()", LOG_TAG);
        m_logger->printAsyncFileLog(common::Logger::LogLevel::Debug, "{} ObjDetectApp::~ObjDetectApp()", LOG_TAG);
    }


    void inference_once() {
        dnn_algorithm::ObjDetectInput objDetectInput = {
            .handleType = "opencv4",
            .imageHandle = m_orig_image_ptr,
        };
        m_dnnObjDetector->pushInputData(std::make_shared<dnn_algorithm::ObjDetectInput>(objDetectInput));
        setObjDetectParams(m_objDetectParams);
        m_dnnObjDetector->runObjDetect(m_objDetectParams);
        auto& objDetectOutput = m_dnnObjDetector->popOutputData();

        for (const auto& item : objDetectOutput) {
            m_logger->printStdoutLog(common::Logger::LogLevel::Debug,
                "Detected object: label={}, score={}, x={}, y={}, width={}, height={}",
                item.label, item.score, item.bbox.left, item.bbox.right, item.bbox.top, item.bbox.bottom);
        }

        std::map<std::string, cv::Scalar> labelColorMap;
        for (size_t i = 0; i < objDetectOutput.size(); ++i) {
            const auto& obj = objDetectOutput[i];
            if (labelColorMap.find(obj.label) == labelColorMap.end()) {
                labelColorMap[obj.label] = m_colors[i % m_colors.size()];
            }
        }

        for (const auto& obj : objDetectOutput) {
            m_logger->printStdoutLog(common::Logger::LogLevel::Info, "{} ObjDetectApp::onProcess() objDetectOutput: bbox: [{}, {}, {}, {}], score: {}, label: {}",
                LOG_TAG, obj.bbox.left, obj.bbox.top, obj.bbox.right, obj.bbox.bottom, obj.score, obj.label);
            cv::rectangle(*m_orig_image_ptr, cv::Point(obj.bbox.left, obj.bbox.top), cv::Point(obj.bbox.right, obj.bbox.bottom),
                    labelColorMap[obj.label], 2);
            cv::putText(*m_orig_image_ptr, obj.label, cv::Point(obj.bbox.left, obj.bbox.top + 12), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(256, 255, 255));
        }

        cv::imwrite("output.jpg", *m_orig_image_ptr);
    }


private:
    void setObjDetectParams(ObjDetectParams& objDetectParams) {
        IDnnEngine::dnnInputShape shape;
        m_dnnObjDetector->getInputShape(shape);

        objDetectParams.model_input_width = shape.width;
        objDetectParams.model_input_height = shape.height;
        objDetectParams.model_input_channel = shape.channel;
        m_args.getSubOptionVal("objDetectParams", "--conf_threshold", objDetectParams.conf_threshold);
        m_args.getSubOptionVal("objDetectParams", "--nms_threshold", objDetectParams.nms_threshold);
        objDetectParams.scale_width = static_cast<float>(shape.width) / static_cast<float>(m_orig_image_ptr->cols);
        objDetectParams.scale_height = static_cast<float>(shape.height) / static_cast<float>(m_orig_image_ptr->rows);
        m_args.getSubOptionVal("objDetectParams", "--pads_left", objDetectParams.pads.left);
        m_args.getSubOptionVal("objDetectParams", "--pads_right", objDetectParams.pads.right);
        m_args.getSubOptionVal("objDetectParams", "--pads_top", objDetectParams.pads.top);
        m_args.getSubOptionVal("objDetectParams", "--pads_bottom", objDetectParams.pads.bottom);
        m_dnnObjDetector->getOutputQuantParams(objDetectParams.quantize_zero_points, objDetectParams.quantize_scales);
    }

private:
    common::ArgParser m_args;
    std::unique_ptr<common::Logger> m_logger{nullptr};
    std::unique_ptr<dnn_algorithm::dnnObjDetector> m_dnnObjDetector{nullptr};
    dnn_algorithm::ObjDetectParams m_objDetectParams{};
    std::shared_ptr<cv::Mat> m_orig_image_ptr{nullptr};
    std::vector<cv::Scalar> m_colors; // for bounding box
};


} // namespace example


#endif // __OBJDETECTAPP_HPP__
