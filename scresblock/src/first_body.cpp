#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_uuid.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <cassert>
#include <chrono>

// ========================= 开关 =========================
// #define DEBUG_MODE 1
// #define TIME 1
// =======================================================

#ifdef TIME
  #define TSTART() auto __t0 = std::chrono::high_resolution_clock::now()
  #define TEND(msg) do { \
    auto __t1 = std::chrono::high_resolution_clock::now(); \
    std::cout << msg << std::chrono::duration<double, std::milli>(__t1 - __t0).count() << " ms\n"; \
  } while (0)
#else
  #define TSTART()    do {} while (0)
  #define TEND(msg)   do {} while (0)
#endif

// ========================= 工具函数 =========================
static bool load_bin(const std::string& path, void* dst, size_t bytes_expected) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "[load_bin] open failed: " << path << "\n";
    return false;
  }
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

static void dump_bin(const std::string& path, const void* src, size_t bytes) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(src), bytes);
}

static void dump_csv(const std::string& path, const float* buf,
                     int H, int W, int C, long long max_rows = -1) {
#ifdef DEBUG_MODE
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
#else
  (void)path; (void)buf; (void)H; (void)W; (void)C; (void)max_rows;
#endif
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

// ========================= 权重加载（body block） =========================
static bool load_block_weights(
    int layer_idx,
    const std::string& weights_dir,
    float* w1_ptr, float* b1_ptr, float* w2_ptr, float* b2_ptr,
    size_t bytes_w, size_t bytes_b,
    xrt::bo& bo_w1, xrt::bo& bo_b1, xrt::bo& bo_w2, xrt::bo& bo_b2)
{
  char buf[32];
  std::snprintf(buf, sizeof(buf), "block%02d", layer_idx);
  std::string base = weights_dir + "/" + buf + "/";

  if (!load_bin(base + "w1.bin", w1_ptr, bytes_w)) return false;
  if (!load_bin(base + "b1.bin", b1_ptr, bytes_b)) return false;
  if (!load_bin(base + "w2.bin", w2_ptr, bytes_w)) return false;
  if (!load_bin(base + "b2.bin", b2_ptr, bytes_b)) return false;

  bo_w1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_w2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  return true;
}

// ========================= 主程序 =========================
int main(int argc, char** argv) {
  // 用法: <xclbin> <H> <W> <C> [Cin0] [weights_dir] [input_bin] [ref_out] [num_layers]
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0]
              << " <xclbin> <H> <W> <C> [Cin0] [weights_dir] [input_bin] [ref_out] [num_layers]\n";
    return 1;
  }

  const char* xclbin_path = argv[1];
  const int H    = std::stoi(argv[2]);
  const int W    = std::stoi(argv[3]);
  const int C    = std::stoi(argv[4]);               // body 通道（Shift8 需要 C%8==0）
  const int Cin0 = (argc >= 6) ? std::stoi(argv[5]) : C;  // 输入通道（例如 3）
  const std::string weights_dir = (argc >= 7) ? argv[6] : std::string("weights");
  const std::string input_path  = (argc >= 8) ? argv[7] : std::string();
  const std::string ref_out_path= (argc >= 9) ? argv[8] : std::string();
  int L = 16;
  if (argc >= 10) {
    L = std::stoi(argv[9]);
    if (L <= 0) {
      std::cerr << "Error: num_layers must be > 0\n";
      return 1;
    }
  }

  std::cout << "[INFO] H=" << H << " W=" << W << " C=" << C
            << " Cin0=" << Cin0 << " L=" << L << "\n";

  if ((C & 7) != 0) {
    std::cerr << "Error: C must be multiple of 8 for Shift8.\n";
    return 1;
  }

  try {
    // 1) 设备 & xclbin
    xrt::device device(0);
    auto xclbin = xrt::xclbin{xclbin_path};
    auto uuid = device.load_xclbin(xclbin);

    // 2) kernel 句柄
    xrt::kernel k_conv1  = xrt::kernel{device, uuid, "conv1x1_ddr"};     // conv_first & body.conv1
    xrt::kernel k_conv2  = xrt::kernel{device, uuid, "conv1x1_ddr"};     // body.conv2
    xrt::kernel k_shift8 = xrt::kernel{device, uuid, "shift8_ddr"};
    xrt::kernel k_relu   = xrt::kernel{device, uuid, "leaky_relu_ddr"};  // alpha=0.1/0.0
    xrt::kernel k_add    = xrt::kernel{device, uuid, "add_residual_ddr"};

    // 3) 尺寸
    const size_t bytes_inout = static_cast<size_t>(H) * W * C    * sizeof(float);
    const size_t bytes_in0   = static_cast<size_t>(H) * W * Cin0 * sizeof(float);
    const size_t bytes_w0    = static_cast<size_t>(C) * Cin0 * sizeof(float); // conv_first [C x Cin0]
    const size_t bytes_b0    = static_cast<size_t>(C) * sizeof(float);        // conv_first [C]
    const size_t bytes_w     = static_cast<size_t>(C) * C    * sizeof(float); // body [C x C]
    const size_t bytes_b     = static_cast<size_t>(C) * sizeof(float);

    // 4) 预处理层 BO（Cin0 -> C）
    auto bo_prein  = xrt::bo(device, bytes_in0,   xrt::bo::flags::normal, k_conv1.group_id(0));
    auto bo_preout = xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_conv1.group_id(3));
    auto bo_w0     = xrt::bo(device, bytes_w0,    xrt::bo::flags::normal, k_conv1.group_id(1));
    auto bo_b0     = xrt::bo(device, bytes_b0,    xrt::bo::flags::normal, k_conv1.group_id(2));

    // 5) 主体 ping-pong + 中间 BO（全部 C 通道）
    auto bo_io0 = xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_conv1.group_id(0));
    auto bo_io1 = xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_conv1.group_id(0));
    int cur = 0;

    auto bo_c1   = xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_conv1.group_id(3));
    auto bo_w1   = xrt::bo(device, bytes_w,     xrt::bo::flags::normal, k_conv1.group_id(1));
    auto bo_b1   = xrt::bo(device, bytes_b,     xrt::bo::flags::normal, k_conv1.group_id(2));

    auto bo_s8   = xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_shift8.group_id(1));
    auto bo_relu = xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_relu.group_id(1));

    auto bo_c2   = xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_conv2.group_id(3));
    auto bo_w2   = xrt::bo(device, bytes_w,     xrt::bo::flags::normal, k_conv2.group_id(1));
    auto bo_b2   = xrt::bo(device, bytes_b,     xrt::bo::flags::normal, k_conv2.group_id(2));

    auto bo_out  = xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_add.group_id(2));

    // 6) 映射指针（一次）
    auto* prein_ptr = bo_prein.map<float*>();
    auto* w0_ptr = bo_w0.map<float*>();
    auto* b0_ptr = bo_b0.map<float*>();
    auto* w1_ptr = bo_w1.map<float*>();
    auto* b1_ptr = bo_b1.map<float*>();
    auto* w2_ptr = bo_w2.map<float*>();
    auto* b2_ptr = bo_b2.map<float*>();

    // 7) 输入（NHWC, Cin0）
    std::cout << "[INFO] Loading input (Cin0=" << Cin0 << ") from " << input_path << " ...\n";
    if (!load_bin(input_path, prein_ptr, bytes_in0)) return 2;
#ifdef DEBUG_MODE
    dump_bin("00_input_Cin0.bin", prein_ptr, bytes_in0);
#endif
    bo_prein.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // 8) conv_first 权重
    if (!load_bin(weights_dir + "/conv_first/w.bin", w0_ptr, bytes_w0)) return 2;
    if (!load_bin(weights_dir + "/conv_first/b.bin", b0_ptr, bytes_b0)) return 2;
    bo_w0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b0.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // 9) 预处理：conv_first → LeakyReLU(0.1) → bo_io0
    {
      TSTART();
      auto r0 = k_conv1(bo_prein, bo_w0, bo_b0, bo_preout, H, W, Cin0, C);
      r0.wait();
      TEND("[PRE] conv_first ");

      TSTART();
      float alpha0 = 0.1f;
      auto r1 = k_relu(bo_preout, bo_io0, H, W, C, alpha0);
      r1.wait();
      TEND("[PRE] leaky_relu ");
    }
    cur = 0; // 主体从 bo_io0 开始

    // 10) 主体循环：L 个 SCNet 残差 Shift 块
    auto t_start = std::chrono::high_resolution_clock::now();
    for (int l = 0; l < L; ++l) {
      std::cout << "[LAYER " << l << "] loading weights...\n";
      if (!load_block_weights(l, weights_dir, w1_ptr, b1_ptr, w2_ptr, b2_ptr,
                              bytes_w, bytes_b, bo_w1, bo_b1, bo_w2, bo_b2)) {
        std::cerr << "[ERROR] failed to load weights for layer " << l << "\n";
        return 2;
      }

      auto& bo_in   = (cur == 0 ? bo_io0 : bo_io1);
      auto& bo_next = (cur == 0 ? bo_io1 : bo_io0);

      // conv1
      TSTART();
      auto run_c1 = k_conv1(bo_in,  bo_w1, bo_b1, bo_c1, H, W, C, C);
      run_c1.wait();
      TEND(("[L" + std::to_string(l) + "] k_conv1 ").c_str());

      // shift8
      TSTART();
      auto run_s8 = k_shift8(bo_c1, bo_s8, H, W, C);
      run_s8.wait();
      TEND(("[L" + std::to_string(l) + "] k_shift8 ").c_str());

      // ReLU (alpha=0.0)
      TSTART();
      auto run_relu = k_relu(bo_s8, bo_relu, H, W, C, 0.0f);
      run_relu.wait();
      TEND(("[L" + std::to_string(l) + "] k_relu ").c_str());

      // conv2
      TSTART();
      auto run_c2 = k_conv2(bo_relu, bo_w2,  bo_b2,  bo_c2, H, W, C, C);
      run_c2.wait();
      TEND(("[L" + std::to_string(l) + "] k_conv2 ").c_str());

      // add residual
      TSTART();
      float res_scale = 1.0f;
      auto run_add = k_add(bo_in, bo_c2, bo_out, res_scale, H, W, C);
      run_add.wait();
      TEND(("[L" + std::to_string(l) + "] add_residual ").c_str());

#ifdef DEBUG_MODE
      bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      {
        char name_bin[64], name_csv[64];
        std::snprintf(name_bin, sizeof(name_bin), "L%02d_out.bin", l);
        std::snprintf(name_csv, sizeof(name_csv), "L%02d_out.csv", l);
        dump_bin(name_bin, bo_out.map<void*>(), bytes_inout);
        dump_csv(name_csv, bo_out.map<float*>(), H, W, C);
      }
#endif
      // D2D → 下一层输入
      bo_next.copy(bo_out);
      cur ^= 1;
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "[TIME] Inference for " << L << " layers took "
              << std::chrono::duration<double, std::milli>(t_end - t_start).count() << " ms\n";
    std::cout << "[TIME] Per-layer average: "
              << std::chrono::duration<double, std::milli>(t_end - t_start).count() / L << " ms\n";

    // 11) 最终输出（在 bo_io[cur]）
    auto& bo_final = (cur == 0 ? bo_io0 : bo_io1);
    bo_final.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
#ifdef DEBUG_MODE
    dump_bin("final_out.bin", bo_final.map<void*>(), bytes_inout);
    dump_csv("final_out.csv", bo_final.map<float*>(), H, W, C);
#endif

    // 12) 参考对比（可选）
    if (!ref_out_path.empty()) {
      std::vector<float> ref((size_t)H * W * C);
      if (!load_bin(ref_out_path, ref.data(), bytes_inout)) {
        std::cerr << "[WARN] failed to load ref_out_bin: " << ref_out_path << "\n";
      } else {
        auto* out_ptr = bo_final.map<const float*>();
        compare_and_report(out_ptr, ref.data(), (size_t)H * W * C, H, W, C, "FINAL");
      }
    }

  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 2;
  }
  return 0;
}
