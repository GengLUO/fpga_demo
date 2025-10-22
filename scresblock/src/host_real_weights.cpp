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

static void dump_bin(const std::string& path, const void* src, size_t bytes) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(src), bytes);
}

static void dump_csv(const std::string& path, const float* buf,
                     int H, int W, int C, long long max_rows = -1) {
  std::ofstream f(path);
  f.setf(std::ios::fixed);
  f << std::setprecision(6);
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

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " <xclbin> <H> <W> <C>\n";
    return 1;
  }
  const char* xclbin_path = argv[1];
  int H = std::stoi(argv[2]);
  int W = std::stoi(argv[3]);
  int C = std::stoi(argv[4]);
  std::string weights_dir = (argc >= 6) ? argv[5] : std::string("weights");

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
    auto bo_in   = xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_conv1.group_id(0));
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
    auto* in_ptr  = bo_in.map<float*>();
    auto* w1_ptr  = bo_w1.map<float*>();
    auto* b1_ptr  = bo_b1.map<float*>();
    auto* w2_ptr  = bo_w2.map<float*>();
    auto* b2_ptr  = bo_b2.map<float*>();

    // Input: random for smoke test
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (size_t i = 0; i < bytes_inout/sizeof(float); ++i) in_ptr[i] = dist(rng);
    dump_bin("01_input.bin", in_ptr, bytes_inout);
    dump_csv("01_input.csv", in_ptr, H, W, C);

    // Load weights & bias from files
    std::string w1p = weights_dir + "/w1.bin";
    std::string b1p = weights_dir + "/b1.bin";
    std::string w2p = weights_dir + "/w2.bin";
    std::string b2p = weights_dir + "/b2.bin";
    if (!load_bin(w1p, w1_ptr, bytes_w)) return 2;
    if (!load_bin(b1p, b1_ptr, bytes_b)) return 2;
    if (!load_bin(w2p, w2_ptr, bytes_w)) return 2;
    if (!load_bin(b2p, b2_ptr, bytes_b)) return 2;

    // Sync host->device for all inputs
    bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_w1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_w2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // 6) Launch kernels in sequence

    // conv1: out -> bo_c1
    auto run_c1 = k_conv1(bo_in, bo_w1, bo_b1, bo_c1, H, W, C, C);
    run_c1.wait();
    bo_c1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    dump_bin("02_c1.bin",   bo_c1.map<void*>(), bytes_inout);
    dump_csv("02_c1.csv",   bo_c1.map<float*>(), H, W, C);

    // shift8: in=bo_c1, out=bo_s8
    auto run_s8 = k_shift8(bo_c1, bo_s8, H, W, C);
    run_s8.wait();
    bo_s8.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    dump_bin("03_s8.bin",   bo_s8.map<void*>(), bytes_inout);
    dump_csv("03_s8.csv",   bo_s8.map<float*>(), H, W, C);

    // relu: in=bo_s8, out=bo_relu
    auto run_relu = k_relu(bo_s8, bo_relu, H, W, C);
    run_relu.wait();
    bo_relu.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    dump_bin("04_relu.bin", bo_relu.map<void*>(), bytes_inout);
    dump_csv("04_relu.csv", bo_relu.map<float*>(), H, W, C);

    // conv2: in=bo_relu, out=bo_c2
    auto run_c2 = k_conv2(bo_relu, bo_w2, bo_b2, bo_c2, H, W, C, C);
    run_c2.wait();
    bo_c2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    dump_bin("05_c2.bin",   bo_c2.map<void*>(), bytes_inout);
    dump_csv("05_c2.csv",   bo_c2.map<float*>(), H, W, C);

    // add residual: x=bo_in, y=bo_c2, out=bo_out, res_scale=1.0f
    float res_scale = 1.0f;
    auto run_add = k_add(bo_in, bo_c2, bo_out, res_scale, H, W, C);
    run_add.wait();
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    dump_bin("06_out.bin",  bo_out.map<void*>(), bytes_inout);
    dump_csv("06_out.csv",  bo_out.map<float*>(), H, W, C);

    // 7) Copy back and print a few samples
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto* out_ptr = bo_out.map<const float*>();
    dump_bin("07_final_output.bin", out_ptr, bytes_inout);
    dump_csv("07_final_output.csv", out_ptr, H, W, C);

    std::cout << "Done. Sample outputs:\n";
    int HW = H*W;
    for (int i = 0; i < std::min(5, HW); ++i) {
      std::cout << "pix " << i << " [0] = " << out_ptr[i*C+0] << "\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 2;
  }
  return 0;
}
