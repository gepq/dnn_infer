#include "dnn_engines/IDnnEngine.hpp"
#include "rknn/rknn.hpp"

namespace dnn_engine {

std::unique_ptr<IDnnEngine> IDnnEngine::create(const std::string& dnnType) {
    if(dnnType.compare("TensorRT") == 0) {
        // return std::make_unique<TensorRTDnn>();
        throw std::invalid_argument("TensorRT is not implemented.");
    }
    else if(dnnType.compare("rknn") == 0) {
        return std::make_unique<rknn>();
    }
    else {
        throw std::invalid_argument("Invalid DNN type specified.");
    }
}

} // namespace dnn_engine
