#include <iostream>
#include <string>
#include "objDetectApp.hpp"

using namespace example;
using namespace common;


int main(int argc, char* argv[])
{
    ArgParser parser("ObjDetectApp");
    parser.addOption("--dnn, --dnnType", std::string("rknn"), "DNN type: trt or rknn");
    parser.addOption("--plugin, --pluginPath", std::string(""), "Path to the plugin library");
    parser.addOption("--label, --labelTextPath", std::string(""), "Path to the label text file");
    parser.addOption("--model, --modelPath", std::string(""), "Path to the model file");
    parser.addOption("--image, --imagePath", std::string(""), "Path to the input image file");

    parser.addSubOption("objDetectParams", "--conf_threshold", float(0.25), "objDetectParams conf_threshold");
    parser.addSubOption("objDetectParams", "--nms_threshold", float(0.45), "objDetectParams nms_threshold");
    parser.addSubOption("objDetectParams", "--pads_left", int(0), "objDetectParams pads_left");
    parser.addSubOption("objDetectParams", "--pads_right", int(0), "objDetectParams pads_right");
    parser.addSubOption("objDetectParams", "--pads_top", int(0), "objDetectParams pads_top");
    parser.addSubOption("objDetectParams", "--pads_bottom", int(0), "objDetectParams pads_bottom");

    parser.parseArgs(argc, argv);


    ObjDetectApp app(std::move(parser));
    app.inference_once();

    return 0;
}

