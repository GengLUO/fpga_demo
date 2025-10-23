#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_uuid.h>

#include <iostream>
#include <vector>
#include <random>
#include <cstring>
#include <cassert>

#include <fstream>   // NEW for loading bins
#include <string>
#include <iomanip>   // for std::setprecision
#include <algorithm>

static bool load_bin(const std::string& path, void* dst, size_t bytes_expected) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "[load_bin] open failed: " << path << "\n";
    return false;
  }
  // (可选) 检查文件大小
  f.seekg(0, std::ios::end);
  std::streamsize sz = f.tellg();
  f.seekg(0, std::ios::beg);
  if (sz != static_cast<std::streamsize>(bytes_expected)) {
    std::cerr << "[load_bin] size mismatch for " << path
              << " (got " << sz << ", expect " << bytes_expected << ")\n";
    return false;
  }
  f.read(reinterpret_cast<char*>(dst), bytes_expected);
  if (!f) {
    std::cerr << "[load_bin] read failed: " << path << "\n";
    return false;
  }
  return true;
}

static void compare_and_report(const float* a, const float* b,
                               size_t N, int H, int W, int C,
                               const char* tag = "FINAL") {
  double mse = 0.0, max_abs = 0.0;
  size_t max_pos = 0;
  for (size_t i = 0; i < N; ++i) {
    double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    double ad = (d < 0) ? -d : d;
    mse += d * d;
    if (ad > max_abs) { max_abs = ad; max_pos = i; }
  }
  mse /= (N ? N : 1);
  int yy = (max_pos / C) / W;
  int xx = (max_pos / C) % W;
  int cc = static_cast<int>(max_pos % C);
  std::cout << "[CHECK-" << tag << "] "
            << "N=" << N << "  MSE=" << std::scientific << mse
            << "  max_abs=" << max_abs
            << " at (y=" << yy << ", x=" << xx << ", c=" << cc << ")\n"
            << std::defaultfloat;
}


static void dump_bin(const std::string& path, const void* src, size_t bytes) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(src), bytes);
}

static void dump_csv(const std::string& path, const float* buf,
                     int H, int W, int C, long long max_rows = -1) {
  std::ofstream f(path);
  f.setf(std::ios::fixed);
  f << std::setprecision(17);
  f << "y,x,c,val\n";
  long long count = 0;
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const size_t base = (static_cast<size_t>(y) * W + x) * C;
      for (int c = 0; c < C; ++c) {
        if (max_rows >= 0 && count >= max_rows) return;
        f << y << ',' << x << ',' << c << ',' << buf[base + c] << '\n';
        ++count;
      }
    }
  }
}

// ---- 封装：按层装载权重的小函数 ----
static bool load_layer_weights(
    int layer_idx,
    const std::string& weights_dir,
    float* w1_ptr, float* b1_ptr, float* w2_ptr, float* b2_ptr,
    size_t bytes_w, size_t bytes_b,
    xrt::bo& bo_w1, xrt::bo& bo_b1, xrt::bo& bo_w2, xrt::bo& bo_b2)
{
  // 子目录
  char buf[32];
  std::snprintf(buf, sizeof(buf), "block%02d", layer_idx);
  std::string base = weights_dir + "/" + buf + "/";

  std::string w1p = base + "w1.bin";
  std::string b1p = base + "b1.bin";
  std::string w2p = base + "w2.bin";
  std::string b2p = base + "b2.bin";

  if (!load_bin(w1p, w1_ptr, bytes_w)) return false;
  if (!load_bin(b1p, b1_ptr, bytes_b)) return false;
  if (!load_bin(w2p, w2_ptr, bytes_w)) return false;
  if (!load_bin(b2p, b2_ptr, bytes_b)) return false;

  bo_w1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_w2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  return true;
};

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " <xclbin> <H> <W> <C> [weights_dir] [input_bin]\n";
    return 1;
  }
  const char* xclbin_path = argv[1];
  int H = std::stoi(argv[2]);
  int W = std::stoi(argv[3]);
  int C = std::stoi(argv[4]);
  std::string weights_dir = (argc >= 6) ? argv[5] : std::string("weights");
  std::string input_path  = (argc >= 7) ? argv[6] : std::string();  // NEW
  std::string ref_out_path= (argc >= 8) ? argv[7] : std::string();   // NEW
  int L = 16;
  if (argc >= 9) {
    L = std::stoi(argv[8]);
    if (L <= 0) { std::cerr << "Error: num_layers must be > 0\n"; return 1; }
  }
  std::cout << "[INFO] num_layers = " << L << "\n";

  if ((C & 7) != 0) {
    std::cerr << "Error: C must be multiple of 8 for Shift8.\n";
    return 1;
  }

  try {
    // 1) Load device & xclbin
    xrt::device device(0);
    auto xclbin = xrt::xclbin{xclbin_path};
    auto uuid = device.load_xclbin(xclbin);

    // 2) Create kernels
    xrt::kernel k_conv1 = xrt::kernel{device, uuid, "conv1x1_ddr"};
    xrt::kernel k_conv2 = xrt::kernel{device, uuid, "conv1x1_ddr"};
    xrt::kernel k_shift8 = xrt::kernel{device, uuid, "shift8_ddr"};
    xrt::kernel k_relu   = xrt::kernel{device, uuid, "relu_ddr"};
    xrt::kernel k_add    = xrt::kernel{device, uuid, "add_residual_ddr"};

    // 3) Sizes (bytes)
    size_t bytes_inout = static_cast<size_t>(H) * W * C * sizeof(float);
    size_t bytes_w     = static_cast<size_t>(C) * C * sizeof(float); // Cin=Cout=C
    size_t bytes_b     = static_cast<size_t>(C) * sizeof(float);

    // 4) Allocate BOs on appropriate memory groups
    // conv1 args: in(0), weight(1), bias(2), out(3), H,W,Cin,Cout (scalars)
    // auto bo_in   = xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_conv1.group_id(0));
    // 把原来的 bo_in 改为 ping-pong：bo_io[0], bo_io[1]
    auto bo_io0 = xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_conv1.group_id(0));
    auto bo_io1 = xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_conv1.group_id(0));
    int cur = 0; // 当前输入在 bo_io[cur]，下一层切换

    auto bo_c1   = xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_conv1.group_id(3)); // conv1 out
    auto bo_w1   = xrt::bo(device, bytes_w,     xrt::bo::flags::normal, k_conv1.group_id(1));
    auto bo_b1   = xrt::bo(device, bytes_b,     xrt::bo::flags::normal, k_conv1.group_id(2));

    // shift8 args: in(0), out(1), H,W,C
    auto bo_s8   = xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_shift8.group_id(1)); // shift8 out

    // relu args: in(0), out(1), H,W,C
    auto bo_relu = xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_relu.group_id(1));   // relu out

    // conv2: reuse same kernel symbol, separate BOs
    auto bo_c2   = xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_conv2.group_id(3)); // conv2 out
    auto bo_w2   = xrt::bo(device, bytes_w,     xrt::bo::flags::normal, k_conv2.group_id(1));
    auto bo_b2   = xrt::bo(device, bytes_b,     xrt::bo::flags::normal, k_conv2.group_id(2));

    // add_residual: x(0), y(1), out(2), res_scale, H,W,C
    auto bo_out  = xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_add.group_id(2));   // final out

    // 5) Map & initialize host-side data
    // auto* in_ptr  = bo_in.map<float*>();
    auto* in0_ptr = bo_io0.map<float*>();   // 初始输入写到 io0
    auto* w1_ptr  = bo_w1.map<float*>();
    auto* b1_ptr  = bo_b1.map<float*>();
    auto* w2_ptr  = bo_w2.map<float*>();
    auto* b2_ptr  = bo_b2.map<float*>();

    // 5.1) 输入：优先从文件读（NHWC/float32），否则回退到随机
    if (!input_path.empty()) {
      std::cout << "[INFO] Loading input from " << input_path << " ...\n";
      if (!load_bin(input_path, in0_ptr, bytes_inout)) return 2;
    } else {
      std::cout << "[INFO] No input_bin provided, using random input.\n";
      std::mt19937 rng(123);
      std::uniform_real_distribution<float> dist(-1.f, 1.f);
      for (size_t i = 0; i < bytes_inout/sizeof(float); ++i) in0_ptr[i] = dist(rng);
    }
    dump_bin("01_input.bin", in0_ptr, bytes_inout);
    dump_csv("01_input.csv", in0_ptr, H, W, C);

    // 初始输入 sync -> device
    bo_io0.sync(XCL_BO_SYNC_BO_TO_DEVICE);

// ---- 6) 主循环：依次跑 L 层 ----
for (int l = 0; l < L; ++l) {
  std::cout << "[LAYER " << l << "] loading weights...\n";
  if (!load_layer_weights(
        l, weights_dir,
        w1_ptr, b1_ptr, w2_ptr, b2_ptr,
        bytes_w, bytes_b,
        bo_w1, bo_b1, bo_w2, bo_b2))
  {
    std::cerr << "[ERROR] failed to load weights for layer " << l << "\n";
    return 2;
  }

  // 当前层输入 BO / 残差基线
  auto& bo_in  = (cur == 0 ? bo_io0 : bo_io1);
  // 下一层输入会用到的 BO（ping-pong 另一端）
  auto& bo_next = (cur == 0 ? bo_io1 : bo_io0);

  // --- conv1 ---
  auto run_c1 = k_conv1(bo_in, bo_w1, bo_b1, bo_c1, H, W, C, C);
  run_c1.wait();

  // --- shift8 ---
  auto run_s8 = k_shift8(bo_c1, bo_s8, H, W, C);
  run_s8.wait();

  // --- relu ---
  auto run_relu = k_relu(bo_s8, bo_relu, H, W, C);
  run_relu.wait();

  // --- conv2 ---
  auto run_c2 = k_conv2(bo_relu, bo_w2, bo_b2, bo_c2, H, W, C, C);
  run_c2.wait();

  // --- add residual: x=bo_in, y=bo_c2, out=bo_out ---
  float res_scale = 1.0f;
  auto run_add = k_add(bo_in, bo_c2, bo_out, res_scale, H, W, C);
  run_add.wait();

  // 结果 bo_out 作为下一层输入：拷到 bo_next，或直接用 device 侧 ping-pong
  // 这里为了避免 device->host->device，我们直接做 device 侧 copy：
  // 简洁做法：host 不介入，直接让下一层把 bo_out 当输入。
  // 但我们为了保持 "bo_in 总是 conv1 的 group_id(0)"，把 bo_out copy 到 bo_next。
  // 如果你的平台支持 device to device copy，可用 xrt::bo::copy(bo_next, bo_out)
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE); // 若要 dump/校验，先取回
  {
    // dump 每层输出（可选）
    char name_bin[64], name_csv[64];
    std::snprintf(name_bin, sizeof(name_bin), "L%02d_out.bin", l);
    std::snprintf(name_csv, sizeof(name_csv), "L%02d_out.csv", l);
    dump_bin(name_bin, bo_out.map<void*>(), bytes_inout);
    dump_csv(name_csv, bo_out.map<float*>(), H, W, C);
  }

  // 将 host 侧数据再写入 bo_next（简单安全但多了一次 H2D；如需更快，可用 D2D 拷贝）
  std::memcpy(bo_next.map<void*>(), bo_out.map<const void*>(), bytes_inout);
  bo_next.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // 交换 ping-pong
  cur ^= 1;
}
// ---- 主循环结束 ----
    

    // // Sync host->device for all inputs
    // bo_in0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // bo_w1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // bo_b1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // bo_w2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // bo_b2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

// ---- 7) 最终输出（在 bo_io[cur] 里） ----
auto& bo_final = (cur == 0 ? bo_io0 : bo_io1);
bo_final.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
dump_bin("final_out.bin", bo_final.map<void*>(), bytes_inout);
dump_csv("final_out.csv", bo_final.map<float*>(), H, W, C);

// (可选) 和参考输出比较
if (!ref_out_path.empty()) {
  std::vector<float> ref(H * W * C);
  if (!load_bin(ref_out_path, ref.data(), bytes_inout)) {
    std::cerr << "[WARN] failed to load ref_out_bin: " << ref_out_path << "\n";
  } else {
    auto* out_ptr = bo_final.map<const float*>();
    compare_and_report(out_ptr, ref.data(), static_cast<size_t>(H) * W * C, H, W, C, "FINAL");
    const double MAX_ABS_THR = 5e-6, MSE_THR = 1e-12;
    double mse = 0.0, max_abs = 0.0;
    for (size_t i=0;i<ref.size();++i) {
      double d = static_cast<double>(out_ptr[i]) - static_cast<double>(ref[i]);
      mse += d*d; 
      if (std::abs(d) > max_abs) max_abs = std::abs(d);
    }
    mse /= (ref.empty()?1:ref.size());
    std::cout << ((max_abs <= MAX_ABS_THR && mse <= MSE_THR) ? "PASS\n" : "FAIL\n");
  }
}


  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 2;
  }
  return 0;
}
