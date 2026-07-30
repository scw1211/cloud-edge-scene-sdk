#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <curl/curl.h>
#include <yaml-cpp/yaml.h>

#include "patchcore_cuda.h"

namespace {
constexpr uint32_t kBankVersion = 1;
constexpr char kBankMagic[] = "PCBNK01";
constexpr float kBlurRadius = 0.0f;

#pragma pack(push, 1)
struct BankHeader {
  char magic[8];
  uint32_t version;
  uint32_t rows;
  uint32_t cols;
  float mean;
  float stddev;
};
#pragma pack(pop)

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    // 只打印 warning 及以上，避免 TensorRT info 日志影响部署端输出。
    if (severity <= Severity::kWARNING) std::cerr << "TensorRT: " << message << std::endl;
  }
};

// TensorRT 旧版对象用 destroy() 释放，用 unique_ptr 包一层避免异常路径泄漏。
template <typename T> struct TrtDeleter { void operator()(T* value) const { if (value) value->destroy(); } };
template <typename T> using TrtPtr = std::unique_ptr<T, TrtDeleter<T>>;

void check(cudaError_t status, const char* action) {
  if (status != cudaSuccess) throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
}

std::vector<char> read_file(const std::string& path) {
  // 小文件使用一次性读入；.pcbank 需要完整解析并上传 GPU。
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("Cannot open: " + path);
  const std::streamsize size = input.tellg();
  input.seekg(0);
  std::vector<char> bytes(static_cast<size_t>(size));
  input.read(bytes.data(), size);
  return bytes;
}

class MappedFile {
 public:
  explicit MappedFile(const std::string& path) {
    // TensorRT plan 文件可能较大，mmap 可避免先读入 vector 再反序列化造成额外匿名内存峰值。
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("Cannot open: " + path);
    struct stat info {};
    if (fstat(fd_, &info) != 0 || info.st_size <= 0) {
      close(fd_);
      fd_ = -1;
      throw std::runtime_error("Cannot stat: " + path);
    }
    size_ = static_cast<size_t>(info.st_size);
    data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (data_ == MAP_FAILED) {
      close(fd_);
      fd_ = -1;
      data_ = nullptr;
      throw std::runtime_error("Cannot map: " + path);
    }
  }

  ~MappedFile() {
    if (data_) munmap(data_, size_);
    if (fd_ >= 0) close(fd_);
  }

  const void* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  int fd_ = -1;
  size_t size_ = 0;
  void* data_ = nullptr;
};

struct Bank {
  // host 保存一份 FP16 bit pattern，供 image_score 在 CPU 端做 PatchCore reweight。
  // device 保存同一份记忆库，供 CUDA kernel 查每个 patch 的最近邻。
  BankHeader header{};
  std::vector<uint16_t> host;
  __half* device = nullptr;
  ~Bank() { if (device) cudaFree(device); }
};

Bank load_bank(const std::string& path) {
  // 读取并校验自定义记忆库文件，防止模型输出通道数和记忆库列数不一致时静默出错。
  const auto bytes = read_file(path);
  if (bytes.size() < sizeof(BankHeader)) throw std::runtime_error("Invalid bank file");
  Bank bank;
  std::memcpy(&bank.header, bytes.data(), sizeof(BankHeader));
  if (std::memcmp(bank.header.magic, kBankMagic, 7) != 0 || bank.header.version != kBankVersion)
    throw std::runtime_error("Unsupported bank format");
  const size_t count = static_cast<size_t>(bank.header.rows) * bank.header.cols;
  const size_t expected = sizeof(BankHeader) + count * sizeof(uint16_t);
  if (bytes.size() != expected) throw std::runtime_error("Bank size mismatch");
  bank.host.resize(count);
  std::memcpy(bank.host.data(), bytes.data() + sizeof(BankHeader), count * sizeof(uint16_t));
  check(cudaMalloc(reinterpret_cast<void**>(&bank.device), count * sizeof(uint16_t)), "Allocate GPU memory bank");
  check(cudaMemcpy(bank.device, bank.host.data(), count * sizeof(uint16_t), cudaMemcpyHostToDevice), "Upload memory bank");
  return bank;
}

void collect_pngs(const std::string& path, std::vector<std::string>* files) {
  // 原始图片路径：递归收集 RGB/test 下所有 png，保持 defect 子目录结构。
  DIR* dir = opendir(path.c_str());
  if (!dir) return;
  while (dirent* item = readdir(dir)) {
    const std::string name(item->d_name);
    if (name == "." || name == "..") continue;
    const std::string child = path + "/" + name;
    if (item->d_type == DT_DIR) collect_pngs(child, files);
    else if (name.size() >= 4 && name.substr(name.size() - 4) == ".png") files->push_back(child);
  }
  closedir(dir);
}

void collect_tensors(const std::string& path, std::vector<std::string>* files) {
  // 预处理路径：prepare_inputs.py 生成的 NCHW float32 张量，后缀固定为 .nchw.f32。
  DIR* dir = opendir(path.c_str());
  if (!dir) return;
  while (dirent* item = readdir(dir)) {
    const std::string name(item->d_name);
    if (name == "." || name == "..") continue;
    const std::string child = path + "/" + name;
    if (item->d_type == DT_DIR) collect_tensors(child, files);
    else if (name.size() >= 9 && name.substr(name.size() - 9) == ".nchw.f32") files->push_back(child);
  }
  closedir(dir);
}

std::vector<float> read_tensor(const std::string& path, size_t count) {
  // 延迟关键路径优先使用预处理张量，避免把图片解码和 resize 计入实时推理。
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input || static_cast<size_t>(input.tellg()) != count * sizeof(float))
    throw std::runtime_error("Invalid preprocessed tensor: " + path);
  std::vector<float> tensor(count);
  input.seekg(0);
  input.read(reinterpret_cast<char*>(tensor.data()), tensor.size() * sizeof(float));
  return tensor;
}

std::string original_path(const std::string& tensor_path, const std::string& tensor_root, const std::string& source_root) {
  // 预处理输入仍需要在 predictions.csv 中回写原始 PNG 路径，方便 verifier 找 GT mask。
  const std::string prefix = tensor_root + "/";
  if (tensor_path.compare(0, prefix.size(), prefix) != 0)
    throw std::runtime_error("Tensor path is outside preprocessed input directory");
  std::string relative = tensor_path.substr(prefix.size());
  relative.resize(relative.size() - std::string(".nchw.f32").size());
  return source_root + "/" + relative + ".png";
}

std::vector<float> preprocess(const std::string& path, int width, int height) {
  // 非预处理模式直接在 C++ 内完成 OpenCV 解码、BGR->RGB、bicubic resize 和 ImageNet normalize。
  // 注意这里使用 OpenCV resize；正式低延迟验收推荐 prepare_inputs.py 的 PIL 路径。
  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
  if (image.empty()) throw std::runtime_error("Cannot decode image: " + path);
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  cv::resize(image, image, cv::Size(width, height), 0, 0, cv::INTER_CUBIC);
  constexpr float mean[] = {0.485f, 0.456f, 0.406f};
  constexpr float stddev[] = {0.229f, 0.224f, 0.225f};
  std::vector<float> output(static_cast<size_t>(3) * width * height);
  for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
    const cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
    for (int c = 0; c < 3; ++c) output[c * width * height + y * width + x] = (pixel[c] / 255.0f - mean[c]) / stddev[c];
  }
  return output;
}

float half_to_float(uint16_t value) { __half h; std::memcpy(&h, &value, sizeof(h)); return __half2float(h); }

std::string json_escape(const std::string& value) {
  // 手写最小 JSON 转义，保证文件路径里出现引号、反斜杠或控制字符时 JSON 仍合法。
  std::ostringstream escaped;
  for (const char ch : value) {
    switch (ch) {
      case '\\': escaped << "\\\\"; break;
      case '"': escaped << "\\\""; break;
      case '\b': escaped << "\\b"; break;
      case '\f': escaped << "\\f"; break;
      case '\n': escaped << "\\n"; break;
      case '\r': escaped << "\\r"; break;
      case '\t': escaped << "\\t"; break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20) {
          escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch);
        } else {
          escaped << ch;
        }
    }
  }
  return escaped.str();
}

std::string utc_timestamp() {
  // 事件时间使用 UTC ISO-8601，格式对齐 rgb_infra_module 里的 CloudEvents 风格 JSON。
  std::time_t now = std::time(nullptr);
  std::tm utc{};
  gmtime_r(&now, &utc);
  std::ostringstream output;
  output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return output.str();
}

std::string file_stem(const std::string& path) {
  // 从原图路径提取不带扩展名的文件名，用于生成可读 subject。
  const size_t slash = path.find_last_of('/');
  const std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
  const size_t dot = name.find_last_of('.');
  return dot == std::string::npos ? name : name.substr(0, dot);
}

std::string score_text(float score) {
  // 文件名和 JSON score 共用同一种短格式，避免默认 ostream 输出过长。
  std::ostringstream output;
  output << std::setprecision(9) << score;
  return output.str();
}

std::string index_text(size_t index) {
  std::ostringstream output;
  output << std::setw(6) << std::setfill('0') << index;
  return output.str();
}

std::string get_local_ip() {
  // 通过 UDP connect 探测本机出口 IP；不会真正向 8.8.8.8 发送业务数据。
  // 如果设备无网络或探测失败，保持原链路可用，source 写 unknown。
  const int fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0) return "unknown";
  sockaddr_in remote{};
  remote.sin_family = AF_INET;
  remote.sin_port = htons(80);
  if (inet_pton(AF_INET, "8.8.8.8", &remote.sin_addr) != 1 ||
      connect(fd, reinterpret_cast<sockaddr*>(&remote), sizeof(remote)) != 0) {
    close(fd);
    return "unknown";
  }
  sockaddr_in local{};
  socklen_t length = sizeof(local);
  if (getsockname(fd, reinterpret_cast<sockaddr*>(&local), &length) != 0) {
    close(fd);
    return "unknown";
  }
  close(fd);
  char buffer[INET_ADDRSTRLEN] = {};
  return inet_ntop(AF_INET, &local.sin_addr, buffer, sizeof(buffer)) ? std::string(buffer) : "unknown";
}

bool post_json(const std::string& url,
               const std::string& json_body) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        return false;
    }

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers,
                                "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body.c_str());

    CURLcode res = curl_easy_perform(curl);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    return res == CURLE_OK;
}

void write_event_json(const std::string& path, const std::string& event_id, const std::string& event_source,
                      const std::string& subject, const std::string& sample_id,
                      const std::string& source_path, const std::string& heatmap_path,
                      float score, double inference_ms) {
  // 写出贴近 rgb_infra_module 的最小事件 JSON：
  // raw_uri 指向原始图片，heatmap_uri 指向实时生成并计入端到端时延的 map_*.f32。
  std::ofstream output(path);
  std::stringstream ss;
  ss << "{\n"
         << "    \"specversion\": \"1.0\",\n"
         << "    \"id\": \"" << json_escape(event_id) << "\",\n"
         << "    \"source\": \"" << json_escape(event_source) << "\",\n"
         << "    \"type\": \"com.example.industrial.anomaly-map.v1\",\n"
         << "    \"scene\": \"industrial_anomaly\",\n"
         << "    \"edgeid\": \"industrial-rgb-edge\",\n"
         << "    \"subject\": \"" << json_escape(subject) << "\",\n"
         << "    \"time\": \"" << utc_timestamp() << "\",\n"
         << "    \"datacontenttype\": \"application/json\",\n"
         << "    \"dataschema\": \"https://cloud-edge.local/schemas/examples/industrial-anomaly-map-v1.json\",\n"
         << "    \"data\": {\n"
         << "        \"asset_id\": \"192.168.31.100\",\n"// 新增，保留字段
         << "        \"sample_id\": \"" << json_escape(sample_id) << "\",\n"
         << "        \"modality\": \"rgb\",\n"
         << "        \"region_id\": \"1\",\n"// 新增，保留字段
         << "        \"threshold\": \"0.5\",\n"// 新增，保留字段
         << "        \"proposed_limit_percent\": \"50\",\n"// 新增，保留字段
         << "        \"shared_resource\": [],\n"// 新增，保留字段
         << "        \"score\": " << score_text(score) << ",\n"
         << "        \"raw_uri\": \"file://" << json_escape(source_path) << "\",\n"
         << "        \"heatmap_uri\": \"file://" << json_escape(heatmap_path) << "\",\n"
         << "        \"inference_ms\": " << inference_ms << "\n"
         << "    }\n"
         << "}\n";
  std::string json_body = ss.str();
  output << json_body;

  std::stringstream request_ss;
  request_ss << "{"
           << "\"event\":"
           << json_body
           << "}";
  output.close();

  YAML::Node config = YAML::LoadFile("config.yaml");

  std::string host = config["host"].as<std::string>();
  int port = config["port"].as<int>();

  std::string url =
    "http://" + host + ":" +
    std::to_string(port) +
    "/api/v1/collaboration/decide";
  

  std::string request_body = request_ss.str();
  post_json(url, request_body);
}

float image_score(const std::vector<uint16_t>& features, const std::vector<float>& distances,
                  const std::vector<int>& indices, const Bank& bank) {
  // PatchCore 图像级分数：
  // 1. 找到距离最大的异常 patch；
  // 2. 找该 patch 最近记忆库向量 m* 的邻居；
  // 3. 用原论文中的 reweight 公式削弱孤立噪声点。
  const int patch = static_cast<int>(std::max_element(distances.begin(), distances.end()) - distances.begin());
  const float s_star = distances[patch] / 1000.0f;
  std::vector<std::pair<float, int>> nearest;
  nearest.reserve(bank.header.rows);
  const int reference = indices[patch];
  for (uint32_t row = 0; row < bank.header.rows; ++row) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < bank.header.cols; ++col) {
      const float a = half_to_float(bank.host[reference * bank.header.cols + col]);
      const float b = half_to_float(bank.host[row * bank.header.cols + col]);
      const float d = a - b; sum += d * d;
    }
    nearest.push_back(std::make_pair(std::sqrt(sum), static_cast<int>(row)));
  }
  std::partial_sort(nearest.begin(), nearest.begin() + std::min<size_t>(3, nearest.size()), nearest.end());
  float denominator = 0.0f;
  const float dim = std::sqrt(static_cast<float>(bank.header.cols));
  for (size_t i = 1; i < std::min<size_t>(3, nearest.size()); ++i) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < bank.header.cols; ++col) {
      const float feature = (half_to_float(features[col * distances.size() + patch]) - bank.header.mean) / bank.header.stddev;
      const float reference_value = half_to_float(bank.host[nearest[i].second * bank.header.cols + col]);
      const float d = feature - reference_value; sum += d * d;
    }
    denominator += std::exp((std::sqrt(sum) / 1000.0f) / dim);
  }
  const float numerator = std::exp(s_star / dim);
  const float weight = denominator > 0.0f ? 1.0f - numerator / denominator : 0.0f;
  return weight * s_star;
}

void write_raw_map(const std::string& path, const std::vector<float>& source, int grid, int size) {
  // TensorRT 输出 patch 网格通常小于输入图，这里用双线性插值恢复到 img_size x img_size 原始得分图。
  cv::Mat map(size, size, CV_32FC1);
  for (int y = 0; y < size; ++y) for (int x = 0; x < size; ++x) {
    const float fy = static_cast<float>(y) * (grid - 1) / (size - 1);
    const float fx = static_cast<float>(x) * (grid - 1) / (size - 1);
    const int y0 = static_cast<int>(fy), x0 = static_cast<int>(fx);
    const int y1 = std::min(y0 + 1, grid - 1), x1 = std::min(x0 + 1, grid - 1);
    const float wy = fy - y0, wx = fx - x0;
    map.at<float>(y, x) = (1 - wy) * ((1 - wx) * source[y0 * grid + x0] + wx * source[y0 * grid + x1]) + wy * ((1 - wx) * source[y1 * grid + x0] + wx * source[y1 * grid + x1]);
  }
  if (kBlurRadius > 0.0f) {
    const int kernel_size = 2 * static_cast<int>(std::ceil(3.0f * kBlurRadius)) + 1;
    cv::GaussianBlur(map, map, cv::Size(kernel_size, kernel_size), kBlurRadius, kBlurRadius, cv::BORDER_REFLECT_101);
  }
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(map.ptr<float>()), static_cast<std::streamsize>(size) * size * sizeof(float));
}

void write_half_tensor(const std::string& path, const std::vector<__half>& values) {
  // 调试开关 DUMP_TRT_FEATURES 使用：导出 TensorRT 特征以便和 PyTorch backbone 对齐。
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(__half)));
}

void write_float_tensor(const std::string& path, const std::vector<float>& values) {
  // 调试开关 DUMP_TRT_DISTANCES 使用：导出 CUDA 最近邻距离以便定位误差来源。
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
}
} 

int main(int argc, char** argv) {
  if (argc != 5 && argc != 7) {
    std::cerr << "Usage: rgb_patchcore_trt ENGINE BANK INPUT_DIR OUTPUT_DIR [--preprocessed SOURCE_PNG_DIR]\n";
    return 2;
  }
  try {
    // 主流程：加载 TensorRT engine -> 加载记忆库 -> 预热 -> 批量推理 -> 写预测、得分图和延迟记录。
    const auto process_started = std::chrono::steady_clock::now();
    const std::string engine_path(argv[1]), bank_path(argv[2]), input_dir(argv[3]), output_dir(argv[4]);
    const bool preprocessed = argc == 7 && std::string(argv[5]) == "--preprocessed";
    if (argc == 7 && !preprocessed) throw std::runtime_error("Expected --preprocessed SOURCE_PNG_DIR");
    const std::string source_dir = preprocessed ? argv[6] : "";
    Logger logger;

    // 先创建 runtime，再反序列化 engine；整体内存峰值由 acceptance.sh 的 time.txt 记录。
    TrtPtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    TrtPtr<nvinfer1::ICudaEngine> engine;
    {
      MappedFile engine_file(engine_path);
      engine.reset(runtime->deserializeCudaEngine(engine_file.data(), engine_file.size(), nullptr));
    }
    TrtPtr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    if (!engine || !context) throw std::runtime_error("Cannot deserialize TensorRT engine");
    const int input = engine->getBindingIndex("image"), output = engine->getBindingIndex("patch_features");
    if (input < 0 || output < 0) throw std::runtime_error("Engine bindings image/patch_features are required");
    const auto input_dims = engine->getBindingDimensions(input), output_dims = engine->getBindingDimensions(output);

    // 当前部署只支持静态方形输入和方形 patch grid，和导出的 ViT-small 160x160 模型匹配。
    const int height = input_dims.d[2], width = input_dims.d[3], channels = output_dims.d[1], grid = output_dims.d[2];
    if (height != width || grid != output_dims.d[3]) throw std::runtime_error("Only square static engines are supported");
    Bank bank = load_bank(bank_path);
    if (bank.header.cols != static_cast<uint32_t>(channels)) throw std::runtime_error("Memory bank channel mismatch");
    if (engine->getBindingDataType(output) != nvinfer1::DataType::kHALF) throw std::runtime_error("Build the engine with FP16 output");
    
    // GPU 缓冲区：
    // input 为 NCHW float32；features 为 TensorRT FP16 输出；distances/indices 为 PatchCore CUDA 后处理结果。
    const int patches = grid * grid; std::vector<void*> bindings(engine->getNbBindings(), nullptr); float* gpu_input = nullptr; __half* gpu_features = nullptr; float* gpu_distances = nullptr; int* gpu_indices = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&gpu_input), 3 * height * width * sizeof(float)), "Allocate input");
    check(cudaMalloc(reinterpret_cast<void**>(&gpu_features), channels * patches * sizeof(__half)), "Allocate features");
    check(cudaMalloc(reinterpret_cast<void**>(&gpu_distances), patches * sizeof(float)), "Allocate distances");
    check(cudaMalloc(reinterpret_cast<void**>(&gpu_indices), patches * sizeof(int)), "Allocate indices");
    bindings[input] = gpu_input; bindings[output] = gpu_features;
    PatchDistanceWorkspace* distance_workspace = create_patch_distance_workspace(channels, patches, bank.header.rows, bank.device);
    std::vector<std::string> files;
    if (preprocessed) collect_tensors(input_dir, &files); else collect_pngs(input_dir, &files);
    std::sort(files.begin(), files.end());

    // 触发 TensorRT 的 lazy setup，避免第一张图延迟混入 engine 内部初始化。
    if (!context->enqueueV2(bindings.data(), 0, nullptr)) throw std::runtime_error("TensorRT warm-up failed");
    check(cudaDeviceSynchronize(), "Synchronize TensorRT warm-up");
    const std::string event_source = get_local_ip();
    const std::string f32_dir = output_dir + "/f32";
    const std::string json_dir = output_dir + "/json";
    std::system(("mkdir -p '" + output_dir + "' '" + f32_dir + "' '" + json_dir + "'").c_str()); std::ofstream csv(output_dir + "/predictions.csv"); csv << "path,image_score,map_file\n";
    std::ofstream latency_csv(output_dir + "/latency.csv");
    latency_csv << "path,inference_ms,end_to_end_ms,cold_start_to_result_ms\n";
    for (size_t i = 0; i < files.size(); ++i) {
      // end_to_end_ms 从读预处理张量/图片开始；inference_ms 拆成 H2D、TRT、CUDA PatchCore、D2H 和 CPU score。
      const auto end_to_end_started = std::chrono::steady_clock::now();
      const auto host_input = preprocessed ? read_tensor(files[i], static_cast<size_t>(3) * height * width) : preprocess(files[i], width, height);
      const auto started = std::chrono::steady_clock::now();
      const auto h2d_started = std::chrono::steady_clock::now();
      check(cudaMemcpy(gpu_input, host_input.data(), host_input.size() * sizeof(float), cudaMemcpyHostToDevice), "Upload input");
      const double input_h2d_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - h2d_started).count();
      const auto trt_started = std::chrono::steady_clock::now();
      if (!context->enqueueV2(bindings.data(), 0, nullptr)) throw std::runtime_error("TensorRT enqueue failed");
      check(cudaDeviceSynchronize(), "Synchronize TensorRT inference");
      const double trt_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - trt_started).count();
      
      // TensorRT 只负责 backbone 特征；PatchCore 最近邻搜索由自定义 CUDA kernel 完成。
      const auto patchcore_started = std::chrono::steady_clock::now();
      patch_min_distances(gpu_features, channels, patches, bank.device, bank.header.rows, bank.header.mean, bank.header.stddev, gpu_distances, gpu_indices, distance_workspace);
      check(cudaDeviceSynchronize(), "Synchronize PatchCore CUDA");
      const double patchcore_cuda_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - patchcore_started).count();
      std::vector<__half> features(channels * patches); std::vector<float> distances(patches); std::vector<int> indices(patches);
      const auto d2h_started = std::chrono::steady_clock::now();
      check(cudaMemcpy(features.data(), gpu_features, features.size() * sizeof(__half), cudaMemcpyDeviceToHost), "Download features"); check(cudaMemcpy(distances.data(), gpu_distances, distances.size() * sizeof(float), cudaMemcpyDeviceToHost), "Download distances"); check(cudaMemcpy(indices.data(), gpu_indices, indices.size() * sizeof(int), cudaMemcpyDeviceToHost), "Download indices");
      const double d2h_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - d2h_started).count();
      if (i == 0 && std::getenv("DUMP_TRT_FEATURES"))
        write_half_tensor(output_dir + "/trt_features_0.f16", features);
      if (i == 0 && std::getenv("DUMP_TRT_DISTANCES"))
        write_float_tensor(output_dir + "/trt_distances_0.f32", distances);
      const auto score_started = std::chrono::steady_clock::now();
      std::vector<uint16_t> feature_bits(features.size()); std::memcpy(feature_bits.data(), features.data(), features.size() * sizeof(uint16_t));
      const float score = image_score(feature_bits, distances, indices, bank);
      const double score_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - score_started).count();
      const double inference_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
      const std::string source_path = preprocessed ? original_path(files[i], input_dir, source_dir) : files[i];
      const std::string sample_index = index_text(i);
      const std::string event_id = "industrial_rgb_event_" + sample_index;
      const std::string subject = "rgb_image_" + sample_index + "_" + file_stem(source_path);
      const std::string sample_id = "sample_" + sample_index;
      const std::string map_name = "map_" + std::to_string(i) + ".f32";
      const std::string map_file = "f32/" + map_name;
      const std::string map_path = output_dir + "/" + map_file;
      write_raw_map(map_path, distances, grid, height);
      const std::string json_name = score_text(score) + "_event_" + sample_index + ".json";
      write_event_json(json_dir + "/" + json_name, event_id, event_source, subject, sample_id,
                       source_path, map_path, score, inference_ms);
      csv << source_path << ',' << score << ',' << map_file << "\n";
      csv.flush();
      const double end_to_end_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - end_to_end_started).count();
      const double cold_start_to_result_ms = i == 0
          ? std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - process_started).count()
          : 0.0;
      latency_csv << source_path << ',' << inference_ms << ',' << end_to_end_ms << ','
                  << cold_start_to_result_ms << "\n";
      latency_csv.flush();
    }
    destroy_patch_distance_workspace(distance_workspace);
    cudaFree(gpu_input); cudaFree(gpu_features); cudaFree(gpu_distances); cudaFree(gpu_indices); std::cout << "Saved " << files.size() << " predictions to " << output_dir << std::endl;
  } catch (const std::exception& error) { std::cerr << "Error: " << error.what() << std::endl; return 1; }
  return 0;
}
