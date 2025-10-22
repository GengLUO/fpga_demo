#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_uuid.h>

#include <iostream>
#include <vector>
#include <random>
#include <cstring>
#include <cassert>
/////////////////////
#include <cmath>
#include <algorithm>

// ---- CPU 参考：NHWC, N=1 ----
static inline size_t idx(int y,int x,int c,int H,int W,int C) {
  return (static_cast<size_t>(y)*W + x)*C + c;
}

// Shift8：方向 i 读源像素的同一通道片 [i*g : (i+1)*g)
static void shift8_host(const float* in, float* out, int H, int W, int C) {
  const int g = C / 8;
  const int dy[8] = {-1,+1, 0, 0,+1,+1,-1,-1};
  const int dx[8] = { 0, 0,-1,+1,+1,-1,+1,-1};
  std::fill(out, out + static_cast<size_t>(H)*W*C, 0.0f);

  for (int y=0; y<H; ++y) {
    for (int x=0; x<W; ++x) {
      size_t ob = idx(y,x,0,H,W,C);
      for (int i=0;i<8;++i) {
        int ys=y+dy[i], xs=x+dx[i];
        int c0=i*g;
        if (ys<0 || ys>=H || xs<0 || xs>=W) continue; // 越界→保持0
        size_t sb = idx(ys,xs,0,H,W,C) + c0;
        for (int k=0;k<g;++k) out[ob + c0 + k] = in[sb + k];
      }
    }
  }
}

static inline void relu_host_inplace(float* buf, size_t N) {
  for (size_t i=0;i<N;++i) buf[i] = std::max(0.0f, buf[i]);
}

static void add_residual_host(const float* x, const float* y, float* out,
                              float res_scale, size_t N) {
  for (size_t i=0;i<N;++i) out[i] = x[i] + res_scale * y[i];
}

///////////////////////

static void fill_identity_weight(std::vector<float>& W, int Cin, int Cout) {
  assert(Cin == Cout);
  std::fill(W.begin(), W.end(), 0.0f);
  for (int c = 0; c < Cin; ++c) W[c*Cin + c] = 1.0f; // weight[co*Cin + ci]
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

    // Weights: identity; Bias: zero
    {
      std::vector<float> W(C*C), W2(C*C), B(C, 0.f), B2(C, 0.f);
      fill_identity_weight(W, C, C);
      fill_identity_weight(W2, C, C);
      std::memcpy(w1_ptr, W.data(), bytes_w);
      std::memcpy(w2_ptr, W2.data(), bytes_w);
      std::memcpy(b1_ptr, B.data(), bytes_b);
      std::memcpy(b2_ptr, B2.data(), bytes_b);
    }

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
    // bo_c1.sync(XCL_BO_SYNC_BO_TO_DEVICE); // ensure data visible to next kernel

    // shift8: in=bo_c1, out=bo_s8
    auto run_s8 = k_shift8(bo_c1, bo_s8, H, W, C);
    run_s8.wait();
    // bo_s8.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // relu: in=bo_s8, out=bo_relu
    auto run_relu = k_relu(bo_s8, bo_relu, H, W, C);
    run_relu.wait();
    // bo_relu.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // conv2: in=bo_relu, out=bo_c2
    auto run_c2 = k_conv2(bo_relu, bo_w2, bo_b2, bo_c2, H, W, C, C);
    run_c2.wait();
    // bo_c2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // add residual: x=bo_in, y=bo_c2, out=bo_out, res_scale=1.0f
    float res_scale = 1.0f;
    auto run_add = k_add(bo_in, bo_c2, bo_out, res_scale, H, W, C);
    run_add.wait();

    // 7) Copy back and print a few samples
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto* out_ptr = bo_out.map<const float*>();

    // ---- CPU 黄金参考：y_ref = x + ReLU(Shift8(x)) ----
{
  const size_t Nelts = static_cast<size_t>(H)*W*C;

  std::vector<float> s8(Nelts), yref(Nelts);
  shift8_host(in_ptr, s8.data(), H, W, C);
  relu_host_inplace(s8.data(), Nelts);
  add_residual_host(in_ptr, s8.data(), yref.data(), /*res_scale=*/1.0f, Nelts);

  double mse = 0.0, max_abs = 0.0;
  size_t max_pos = 0;
  for (size_t i=0;i<Nelts;++i) {
    double d  = static_cast<double>(out_ptr[i]) - static_cast<double>(yref[i]);
    double ad = std::fabs(d);
    mse += d*d;
    if (ad > max_abs) { max_abs = ad; max_pos = i; }
  }
  mse /= (Nelts ? Nelts : 1);
  int my = (max_pos / C) / W;
  int mx = (max_pos / C) % W;
  int mc = static_cast<int>(max_pos % C);

  std::cout << "\n[CHECK] FPGA vs CPU-golden  N=" << Nelts
            << "  MSE=" << mse
            << "  max_abs=" << max_abs
            << " at (y=" << my << ", x=" << mx << ", c=" << mc << ")\n\n";
}
// ---- End CPU reference ----



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
