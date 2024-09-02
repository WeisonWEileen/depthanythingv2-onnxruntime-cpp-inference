#include <functional> // for std::multiplies
#include <numeric>    // for std::accumulate
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/core.hpp>
#include <vector>

// colorful output
#define BLACK "\033[30m" /* Black */
#define RED "\033[31m"   /* Red */
#define GREEN "\033[32m" /* Green */

class DepthAnyV2Onnx
{
  public:
    DepthAnyV2Onnx(const char* onnx_model_path);
    cv::Mat predict(const cv::Mat& input_tensor);

    // @TODO do we really need index?
    cv::Mat preprocessImg(const cv::Mat& mat);

  private:
    // Ort::Env env_;
    // Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    // viewed from vitb.onnx
    std::vector<const char*> input_node_names_ = {"l_x_"};
    std::vector<const char*> output_node_names_ = {"select_36"};
    std::vector<int64_t> input_node_dims_;
    std::vector<int64_t> output_node_dims_;
    const char* onnx_model_path_;
    void printInputOutputInfo(Ort::Session& session);
};

template <typename T>
T product(const std::vector<T>& v)
{
    return std ::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}