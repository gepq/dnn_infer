#ifndef __IDNN_ENGINE_HPP__
#define __IDNN_ENGINE_HPP__

#include <memory>
#include <string>
#include <vector>

namespace dnn_engine {

class IDnnEngine {
public:

    // Used to standardize custom data types from different dnn engines.
    struct dnnInputShape {
        size_t width{0};
        size_t height{0};
        size_t channel{0};
    };

    struct dnnInput {
        size_t index{0};
        std::vector<uint8_t> buf{};
        size_t size{0};
        dnnInputShape shape{};
        // dataType can be "UINT8", "float32"
        std::string dataType{"UINT8"};
    };

    struct dnnOutput {
        size_t index{0};
        void* buf{nullptr};
        size_t size{0};
        // dataType can be "UINT8", "INT8", "float32"
        std::string dataType{"INT8"};
    };

    static std::unique_ptr<IDnnEngine> create(const std::string& dnnType);

    virtual void loadModel(const std::string& modelPath) = 0;

    virtual int getInputShape(dnnInputShape& shape) = 0;

    /* for networks using quantitative models
     * when using a quantization model, the post-processing process requires inverse quantization to 
     * floating-point data based on the model's scale array and zero_point array
     */
    virtual int getOutputQuantParams(std::vector<int32_t>& zeroPoints, std::vector<float>& scales) = 0;

    virtual int pushInputData(dnnInput& inputData) = 0;

    virtual int popOutputData(std::vector<dnnOutput>& outputVector) = 0;

    virtual int runInference() = 0;

    virtual ~IDnnEngine() = default;

private:
    IDnnEngine() = default;
};

} // namespace dnn_engine

#endif // __IDNN_ENGINE_HPP__
