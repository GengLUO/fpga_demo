// first_body_u12_cpu_ps_full.cpp
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
#include <algorithm>
#include <cmath>

#define DEBUG_MODE 1
#define TIME 1

#ifdef TIME
  #define TSTART(tag) auto tag = std::chrono::high_resolution_clock::now()
  #define TEND(tag, msg) do { \
    auto __t1 = std::chrono::high_resolution_clock::now(); \
    std::cout << msg \
      << std::chrono::duration<double, std::milli>(__t1 - (tag)).count() \
      << " ms\n"; \
  } while (0)
#else
  #define TSTART(tag)    do{}while(0)
  #define TEND(tag, msg) do{}while(0)
#endif

// ---------------- utils ----------------
static bool load_bin(const std::string& path, void* dst, size_t bytes_expected) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { std::cerr << "[load_bin] open failed: " << path << "\n"; return false; }
  f.seekg(0, std::ios::end);
  std::streamsize sz = f.tellg();
  f.seekg(0, std::ios::beg);
  if (sz != static_cast<std::streamsize>(bytes_expected)) {
    std::cerr << "[load_bin] size mismatch for " << path
              << " (got " << sz << ", expect " << bytes_expected << ")\n";
    return false;
  }
  f.read(reinterpret_cast<char*>(dst), bytes_expected);
  if (!f) { std::cerr << "[load_bin] read failed: " << path << "\n"; return false; }
  return true;
}

static void dump_bin(const std::string& path, const void* src, size_t bytes) {
#ifdef DEBUG_MODE
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(src), bytes);
#else
  (void)path; (void)src; (void)bytes;
#endif
}

static void dump_csv(const std::string& path, const float* buf,
                     int H, int W, int C, long long max_rows=-1) {
#ifdef DEBUG_MODE
  std::ofstream f(path);
  f.setf(std::ios::fixed);
  f << std::setprecision(17);
  f << "y,x,c,val\n";
  long long count=0;
  for (int y=0;y<H;++y) for (int x=0;x<W;++x) {
    size_t base=((size_t)y*W + x)*C;
    for (int c=0;c<C;++c) {
      if (max_rows>=0 && count>=max_rows) return;
      f<<y<<","<<x<<","<<c<<","<<buf[base+c]<<"\n"; ++count;
    }
  }
#else
  (void)path; (void)buf; (void)H; (void)W; (void)C; (void)max_rows;
#endif
}

static void compare_and_report(const float* a, const float* b,
                               size_t N, int H, int W, int C,
                               const char* tag = "FINAL") {
  double mse=0.0, max_abs=0.0; size_t max_pos=0;
  for (size_t i=0;i<N;++i) {
    double d=(double)a[i]-(double)b[i];
    double ad = d<0 ? -d : d;
    mse+=d*d;
    if (ad>max_abs) { max_abs=ad; max_pos=i; }
  }
  mse/= (N?N:1);
  int yy=(int)((max_pos/C)/W); int xx=(int)((max_pos/C)%W); int cc=(int)(max_pos% C);
  std::cout<<"[CHECK-"<<tag<<"] N="<<N<<"  MSE="<<std::scientific<<mse
           <<"  max_abs="<<max_abs<<" at (y="<<yy<<", x="<<xx<<", c="<<cc<<")\n"
           <<std::defaultfloat;
}

static bool write_ppm_from_nhwc_float(const std::string& path,
                                      const float* nhwc, int H, int W) {
  if (!nhwc) return false;
  const int C = 3;
  const size_t N = (size_t)H * W * C;
  std::vector<unsigned char> rgb8(N);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      size_t base = ((size_t)y * W + x) * C;
      float r = std::clamp(nhwc[base + 0], 0.0f, 1.0f);
      float g = std::clamp(nhwc[base + 1], 0.0f, 1.0f);
      float b = std::clamp(nhwc[base + 2], 0.0f, 1.0f);
      rgb8[base + 0] = static_cast<unsigned char>(r * 255.0f);
      rgb8[base + 1] = static_cast<unsigned char>(g * 255.0f);
      rgb8[base + 2] = static_cast<unsigned char>(b * 255.0f);
    }
  }

  std::ofstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "[write_ppm] open failed: " << path << "\n";
    return false;
  }

  // 写 PPM header
  f << "P6\n" << W << " " << H << "\n255\n";
  f.write(reinterpret_cast<const char*>(rgb8.data()),
          static_cast<std::streamsize>(rgb8.size()));

  if (!f) {
    std::cerr << "[write_ppm] write failed: " << path << "\n";
    return false;
  }
  return true;
}

// ------------- weight loaders -------------
static bool load_block_weights(
  int layer_idx, const std::string& weights_dir,
  float* w1_ptr, float* b1_ptr, float* w2_ptr, float* b2_ptr,
  size_t bytes_w, size_t bytes_b,
  xrt::bo& bo_w1, xrt::bo& bo_b1, xrt::bo& bo_w2, xrt::bo& bo_b2)
{
  char buf[32]; std::snprintf(buf,sizeof(buf),"block%02d",layer_idx);
  std::string base = weights_dir + "/" + buf + "/";
  if (!load_bin(base+"w1.bin", w1_ptr, bytes_w)) return false;
  if (!load_bin(base+"b1.bin", b1_ptr, bytes_b)) return false;
  if (!load_bin(base+"w2.bin", w2_ptr, bytes_w)) return false;
  if (!load_bin(base+"b2.bin", b2_ptr, bytes_b)) return false;
  bo_w1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_w2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  return true;
}

static bool load_uplayer_pair_shapes(
  const std::string& dirA, int C_out_A,
  const std::string& dirB, int C_out_B,
  const std::string& weights_dir,
  float* wA_ptr, float* bA_ptr, float* wB_ptr, float* bB_ptr,
  xrt::bo& bo_wA, xrt::bo& bo_bA, xrt::bo& bo_wB, xrt::bo& bo_bB,
  int C_in)
{
  size_t bytes_wA = (size_t)C_out_A * C_in * sizeof(float);
  size_t bytes_bA = (size_t)C_out_A * sizeof(float);
  size_t bytes_wB = (size_t)C_out_B * C_in * sizeof(float);
  size_t bytes_bB = (size_t)C_out_B * sizeof(float);

  std::string dA = weights_dir + "/" + dirA + "/";
  std::string dB = weights_dir + "/" + dirB + "/";

  if (!load_bin(dA+"w.bin", wA_ptr, bytes_wA)) return false;
  if (!load_bin(dA+"b.bin", bA_ptr, bytes_bA)) return false;
  if (!load_bin(dB+"w.bin", wB_ptr, bytes_wB)) return false;
  if (!load_bin(dB+"b.bin", bB_ptr, bytes_bB)) return false;

  bo_wA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_wB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_bB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  return true;
}

int main(int argc, char** argv) {
  std::cout << "==== HOST: conv_first + BODY + U1 + U2 (CPU PixelShuffle) ====\n";
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0]
              << " <xclbin> <H> <W> <C> [Cin0] [weights_dir] [input_bin] [ref_out] [num_layers] [base_bin] [out_img]\n";
    return 1;
  }

  const char* xclbin_path = argv[1];
  const int H    = std::stoi(argv[2]);
  const int W    = std::stoi(argv[3]);
  const int C    = std::stoi(argv[4]);        // body channels
  const int Cin0 = (argc >= 6) ? std::stoi(argv[5]) : C;
  const std::string weights_dir = (argc >= 7) ? argv[6] : "weights";
  const std::string input_path  = (argc >= 8) ? argv[7] : "";
  const std::string ref_out_path= (argc >= 9) ? argv[8] : "";
  int L = (argc >= 10) ? std::stoi(argv[9]) : 16;
  const std::string base_bin_path = (argc >= 11) ? argv[10] : "";
  const std::string out_img_path  = (argc >= 12) ? argv[11] : "";
  if (L <= 0) { std::cerr << "Error: num_layers must be > 0\n"; return 1; }

  const int C4 = 4*C;
  const int H2 = 2*H, W2 = 2*W;
  const int H4 = 4*H, W4 = 4*W;

  if ((C & 7) != 0) {
    std::cerr << "Error: C must be multiple of 8 for Shift8.\n"; return 1;
  }

  try {
    // device + xclbin
    xrt::device device(0);
    auto xclbin = xrt::xclbin{xclbin_path};
    auto uuid = device.load_xclbin(xclbin);

    // kernels
    xrt::kernel k_conv1{device, uuid, "conv1x1_ddr"};
    xrt::kernel k_conv2{device, uuid, "conv1x1_ddr"};
    xrt::kernel k_shift8{device, uuid, "shift8_ddr"};
    xrt::kernel k_relu  {device, uuid, "leaky_relu_ddr"};
    xrt::kernel k_add   {device, uuid, "add_residual_ddr"};
    xrt::kernel k_ps2{device, uuid, "pixelshuffle2x_ddr"};

    // sizes
    const size_t bytes_inout = (size_t)H * W * C * sizeof(float);
    const size_t bytes_in0   = (size_t)H * W * Cin0 * sizeof(float);
    const size_t bytes_w0    = (size_t)C * Cin0 * sizeof(float);
    const size_t bytes_b0    = (size_t)C * sizeof(float);
    const size_t bytes_w     = (size_t)C * C * sizeof(float);
    const size_t bytes_b     = (size_t)C * sizeof(float);

    // conv_first BO
    auto bo_prein  = xrt::bo(device, bytes_in0,   xrt::bo::flags::normal, k_conv1.group_id(0));
    auto bo_preout = xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_conv1.group_id(3));
    auto bo_w0     = xrt::bo(device, bytes_w0,    xrt::bo::flags::normal, k_conv1.group_id(1));
    auto bo_b0     = xrt::bo(device, bytes_b0,    xrt::bo::flags::normal, k_conv1.group_id(2));

    // body ping-pong + mid BO
    auto bo_io0 = xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_conv1.group_id(0));
    auto bo_io1 = xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_conv1.group_id(0));
    int cur = 0;

    auto bo_c1   = xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_conv1.group_id(3));
    auto bo_w1   = xrt::bo(device, bytes_w,     xrt::bo::flags::normal, k_conv1.group_id(1)); // body weights CxC
    auto bo_b1   = xrt::bo(device, bytes_b,     xrt::bo::flags::normal, k_conv1.group_id(2));

    auto bo_s8   = xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_shift8.group_id(1));
    auto bo_relu_b= xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_relu.group_id(1));

    auto bo_c2   = xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_conv2.group_id(3));
    auto bo_w2   = xrt::bo(device, bytes_w,     xrt::bo::flags::normal, k_conv2.group_id(1)); // body weights CxC
    auto bo_b2   = xrt::bo(device, bytes_b,     xrt::bo::flags::normal, k_conv2.group_id(2));

    auto bo_out  = xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_add.group_id(2));

    // map
    auto* prein_ptr = bo_prein.map<float*>();
    auto* w0_ptr = bo_w0.map<float*>();
    auto* b0_ptr = bo_b0.map<float*>();
    auto* w1_ptr = bo_w1.map<float*>();
    auto* b1_ptr = bo_b1.map<float*>();
    auto* w2_ptr = bo_w2.map<float*>();
    auto* b2_ptr = bo_b2.map<float*>();

    std::cout << "[INFO] H="<<H<<" W="<<W<<" C="<<C<<" Cin0="<<Cin0<<" L="<<L<<"\n";
    std::cout << "[INFO] Loading input (Cin0="<<Cin0<<") from " << input_path << " ...\n";
    if (!load_bin(input_path, prein_ptr, bytes_in0)) return 2;
    bo_prein.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    if (!load_bin(weights_dir+"/conv_first/w.bin", w0_ptr, bytes_w0)) return 2;
    if (!load_bin(weights_dir+"/conv_first/b.bin", b0_ptr, bytes_b0)) return 2;
    bo_w0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b0.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // conv_first → LReLU(0.1)
    {
      TSTART(t_pre_conv);
      auto r0=k_conv1(bo_prein, bo_w0, bo_b0, bo_preout, H,W,Cin0,C); r0.wait();
      TEND(t_pre_conv, "[PRE] conv_first ");
      TSTART(t_pre_lrelu);
      auto r1=k_relu(bo_preout, bo_io0, H,W,C, 0.1f); r1.wait();
      TEND(t_pre_lrelu, "[PRE] leaky_relu ");
    }
    cur = 0;

    // BODY: L 层
    auto t_loop_start = std::chrono::high_resolution_clock::now();
    for (int l=0; l<L; ++l) {
      std::cout << "[LAYER " << l << "] loading weights...\n";
      if (!load_block_weights(l, weights_dir, w1_ptr,b1_ptr,w2_ptr,b2_ptr,
                              bytes_w,bytes_b, bo_w1,bo_b1,bo_w2,bo_b2)) {
        std::cerr << "[ERROR] failed to load weights for layer " << l << "\n"; return 2;
      }
      auto& bo_in   = (cur==0 ? bo_io0 : bo_io1);
      auto& bo_next = (cur==0 ? bo_io1 : bo_io0);

      TSTART(t_c1);
      auto r1=k_conv1(bo_in,  bo_w1,bo_b1,bo_c1, H,W,C,C); r1.wait();
      TEND(t_c1, ("[L"+std::to_string(l)+"] k_conv1 ").c_str());

      TSTART(t_s8);
      auto r2=k_shift8(bo_c1, bo_s8, H,W,C); r2.wait();
      TEND(t_s8, ("[L"+std::to_string(l)+"] k_shift8 ").c_str());

      TSTART(t_relu0);
      auto r3=k_relu(bo_s8, bo_relu_b, H,W,C, 0.0f); r3.wait();
      TEND(t_relu0, ("[L"+std::to_string(l)+"] k_relu ").c_str());

      TSTART(t_c2);
      auto r4=k_conv2(bo_relu_b, bo_w2,bo_b2,bo_c2, H,W,C,C); r4.wait();
      TEND(t_c2, ("[L"+std::to_string(l)+"] k_conv2 ").c_str());

      TSTART(t_add);
      auto r5=k_add(bo_in, bo_c2, bo_out, 1.0f, H,W,C); r5.wait();
      TEND(t_add, ("[L"+std::to_string(l)+"] add_residual ").c_str());

      bo_next.copy(bo_out);
      cur ^= 1;
    }
    auto t_loop_end = std::chrono::high_resolution_clock::now();
    std::cout << "[TIME] Inference for " << L << " layers took "
              << std::chrono::duration<double,std::milli>(t_loop_end - t_loop_start).count() << " ms\n";

#if DEBUG_MODE
    // 如需调试 BODY 输出可在此 dump（不做 compare）
    { auto& bo_final = (cur==0?bo_io0:bo_io1);
      bo_final.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      dump_bin("body_final.bin", bo_final.map<void*>(), bytes_inout);
      dump_csv("body_final.csv", bo_final.map<float*>(), H,W,C);
    }
#endif

    // ---------------- U1: upconv1 + PS2 + LReLU(0.1) ----------------
    std::cout << "[U1] begin\n";
    const size_t bytes_c4_hw = (size_t)H * W * (4*C) * sizeof(float);
    const size_t bytes_2xC   = (size_t)H2 * W2 * C * sizeof(float);

    // U1 buffers
    auto bo_c_tmp    = xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_conv1.group_id(3)); // HxWxC
    auto bo_c4_tmp1  = xrt::bo(device, bytes_c4_hw, xrt::bo::flags::normal, k_conv2.group_id(3)); // HxWx4C
    auto bo_u1_out   = xrt::bo(device, bytes_2xC,   xrt::bo::flags::normal, k_relu.group_id(1));  // 2Hx2WxC
    // NEW: out-of-place Shift8 buffer for U1
    auto bo_s8_u1    = xrt::bo(device, bytes_inout, xrt::bo::flags::normal, k_shift8.group_id(1)); // HxWxC

    // U1 权重 BO
    auto bo_w1_u1 = xrt::bo(device, (size_t)C * C   * sizeof(float), xrt::bo::flags::normal, k_conv1.group_id(1));
    auto bo_b1_u1 = xrt::bo(device, (size_t)C       * sizeof(float), xrt::bo::flags::normal, k_conv1.group_id(2));
    auto bo_w2_u1 = xrt::bo(device, (size_t)C4 * C  * sizeof(float), xrt::bo::flags::normal, k_conv2.group_id(1));
    auto bo_b2_u1 = xrt::bo(device, (size_t)C4      * sizeof(float), xrt::bo::flags::normal, k_conv2.group_id(2));

    float* w1_u1 = bo_w1_u1.map<float*>(); float* b1_u1 = bo_b1_u1.map<float*>();
    float* w2_u1 = bo_w2_u1.map<float*>(); float* b2_u1 = bo_b2_u1.map<float*>();

    if (!load_uplayer_pair_shapes("upconv1_up0", C, "upconv1_up3", C4,
        weights_dir, w1_u1,b1_u1,w2_u1,b2_u1, bo_w1_u1,bo_b1_u1,bo_w2_u1,bo_b2_u1, C)) {
      std::cerr<<"[ERROR] failed to load upconv1 pair\n"; return 2;
    }

    auto& bo_body_out = (cur==0?bo_io0:bo_io1);

    std::cout<<"[U1] convA C->C\n";
    TSTART(t_u1_c1);
    { auto r = k_conv1(bo_body_out, bo_w1_u1, bo_b1_u1, bo_c_tmp, H, W, C, C); r.wait(); }
    TEND(t_u1_c1, "[U1] k_conv1 ");
#if DEBUG_MODE
    bo_c_tmp.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    dump_bin("U1_A_out.bin",  bo_c_tmp.map<void*>(), (size_t)H*W*C*sizeof(float));
    dump_csv("U1_A_out.csv",  bo_c_tmp.map<float*>(), H, W, C);
#endif

    std::cout<<"[U1] lrelu(0.02)\n";
    { auto r = k_relu(bo_c_tmp, bo_c_tmp, H, W, C, 0.02f); r.wait(); }
#if DEBUG_MODE
    bo_c_tmp.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    dump_bin("U1_A_relu.bin", bo_c_tmp.map<void*>(), (size_t)H*W*C*sizeof(float));
    dump_csv("U1_A_relu.csv",  bo_c_tmp.map<float*>(), H, W, C);
#endif

    std::cout<<"[U1] shift8\n";
    { auto r = k_shift8(bo_c_tmp, bo_s8_u1, H, W, C); r.wait(); } // out-of-place
#if DEBUG_MODE
    bo_s8_u1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    dump_bin("U1_shift.bin", bo_s8_u1.map<void*>(), (size_t)H*W*C*sizeof(float));
    dump_csv("U1_shift.csv",  bo_s8_u1.map<float*>(), H, W, C);
#endif

    std::cout<<"[U1] convB C->4C\n";
    TSTART(t_u1_c2);
    { auto r = k_conv2(bo_s8_u1, bo_w2_u1, bo_b2_u1, bo_c4_tmp1, H, W, C, C4); r.wait(); }
    TEND(t_u1_c2, "[U1] k_conv2 ");

    std::cout<<"[U1] CPU PixelShuffle (H,W,4C)->(2H,2W,C)\n";
    bo_c4_tmp1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
#if DEBUG_MODE
    dump_bin("U1_B_out_hw4c.bin", bo_c4_tmp1.map<void*>(), (size_t)H*W*(4*C)*sizeof(float));
    dump_csv("U1_B_out_hw4c.csv",  bo_c4_tmp1.map<float*>(), H, W, (4*C));
#endif

    std::cout<<"[U1] PS2 kernel (H,W,4C)->(2H,2W,C)\n";
    { auto r = k_ps2(bo_c4_tmp1, bo_u1_out, H, W, C); r.wait(); }

    std::cout<<"[U1] post lrelu(0.1)\n";
    { auto r = k_relu(bo_u1_out, bo_u1_out, H2, W2, C, 0.1f); r.wait(); }
#if DEBUG_MODE
    bo_u1_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    dump_bin("U1_out.bin", bo_u1_out.map<void*>(), (size_t)H2*W2*C*sizeof(float));
    dump_csv("U1_out.csv",  bo_u1_out.map<float*>(), H2, W2, C);
#endif

    // ---------------- U2: upconv2 + PS2 + LReLU(0.1) ----------------
    std::cout << "[U2] begin\n";
    const size_t bytes_c4_h2w2 = (size_t)H2 * W2 * (4*C) * sizeof(float);
    const size_t bytes_4xC     = (size_t)H4 * W4 * C * sizeof(float);

    // U2 buffers
    auto bo_c_tmp2   = xrt::bo(device, (size_t)H2*W2*C*sizeof(float), xrt::bo::flags::normal, k_conv1.group_id(3)); // 2Hx2WxC
    auto bo_c4_tmp2  = xrt::bo(device, bytes_c4_h2w2, xrt::bo::flags::normal, k_conv2.group_id(3));                 // 2Hx2Wx4C
    auto bo_u2_out   = xrt::bo(device, bytes_4xC,     xrt::bo::flags::normal, k_relu.group_id(1));                  // 4Hx4WxC
    // NEW: out-of-place Shift8 buffer for U2
    auto bo_s8_u2    = xrt::bo(device, (size_t)H2*W2*C*sizeof(float), xrt::bo::flags::normal, k_shift8.group_id(1)); // 2Hx2WxC

    // U2 权重 BO
    auto bo_w1_u2 = xrt::bo(device, (size_t)C * C   * sizeof(float), xrt::bo::flags::normal, k_conv1.group_id(1));
    auto bo_b1_u2 = xrt::bo(device, (size_t)C       * sizeof(float), xrt::bo::flags::normal, k_conv1.group_id(2));
    auto bo_w2_u2 = xrt::bo(device, (size_t)C4 * C  * sizeof(float), xrt::bo::flags::normal, k_conv2.group_id(1));
    auto bo_b2_u2 = xrt::bo(device, (size_t)C4      * sizeof(float), xrt::bo::flags::normal, k_conv2.group_id(2));

    float* w1_u2 = bo_w1_u2.map<float*>(); float* b1_u2 = bo_b1_u2.map<float*>();
    float* w2_u2 = bo_w2_u2.map<float*>(); float* b2_u2 = bo_b2_u2.map<float*>();

    if (!load_uplayer_pair_shapes("upconv2_up0", C, "upconv2_up3", C4,
        weights_dir, w1_u2,b1_u2,w2_u2,b2_u2, bo_w1_u2,bo_b1_u2,bo_w2_u2,bo_b2_u2, C)) {
      std::cerr<<"[ERROR] failed to load upconv2 pair\n"; return 2;
    }

    std::cout<<"[U2] convA C->C\n";
    TSTART(t_u2_c1);
    { auto r = k_conv1(bo_u1_out, bo_w1_u2, bo_b1_u2, bo_c_tmp2, H2, W2, C, C); r.wait(); }
    TEND(t_u2_c1, "[U2] k_conv1 ");
    
    std::cout<<"[U2] lrelu(0.02)\n";
    { auto r = k_relu(bo_c_tmp2, bo_c_tmp2, H2, W2, C, 0.02f); r.wait(); }

    std::cout<<"[U2] shift8\n";
    { auto r = k_shift8(bo_c_tmp2, bo_s8_u2, H2, W2, C); r.wait(); } // out-of-place
#if DEBUG_MODE
    bo_s8_u2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    dump_bin("U2_shift.bin", bo_s8_u2.map<void*>(), (size_t)H2*W2*C*sizeof(float));
    dump_csv("U2_shift.csv",  bo_s8_u2.map<float*>(), H2, W2, C);
#endif

    std::cout<<"[U2] convB C->4C\n";
    TSTART(t_u2_c2);
    { auto r = k_conv2(bo_s8_u2, bo_w2_u2, bo_b2_u2, bo_c4_tmp2, H2, W2, C, C4); r.wait(); }
    TEND(t_u2_c2, "[U2] k_conv2 ");

    std::cout<<"[U2] PS2 kernel (2H,2W,4C)->(4H,4W,C)\n";
    { auto r = k_ps2(bo_c4_tmp2, bo_u2_out, H2, W2, C); r.wait(); }

    std::cout<<"[U2] post lrelu(0.1)\n";
    { auto r = k_relu(bo_u2_out, bo_u2_out, H4, W4, C, 0.1f); r.wait(); }

#if DEBUG_MODE
    bo_u2_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    dump_bin("U2_out.bin", bo_u2_out.map<void*>(), bytes_4xC);
    dump_csv("U2_out.csv",  bo_u2_out.map<float*>(), H4, W4, C);
#endif

    std::cout<<"[DONE] BODY + U1 + U2 ok.\n";

    // ---------------- conv_hr -> LReLU(0.1) -> conv_last ----------------
    std::cout << "[POST] conv_hr + lrelu + conv_last\n";

    // sizes for post head
    const size_t bytes_hr_w   = (size_t)C * C * sizeof(float);   // 64x64
    const size_t bytes_hr_b   = (size_t)C * sizeof(float);       // 64
    const int C_out_last      = 3;
    const size_t bytes_last_w = (size_t)C_out_last * C * sizeof(float);  // 3x64
    const size_t bytes_last_b = (size_t)C_out_last * sizeof(float);      // 3
    const size_t bytes_4xRGB  = (size_t)H4 * W4 * C_out_last * sizeof(float);

    // buffers
    auto bo_hr_out     = xrt::bo(device, bytes_4xC,    xrt::bo::flags::normal, k_conv1.group_id(3)); // 4Hx4WxC
    auto bo_final_out  = xrt::bo(device, bytes_4xRGB,  xrt::bo::flags::normal, k_conv2.group_id(3)); // 4Hx4Wx3

    auto bo_w_hr   = xrt::bo(device, bytes_hr_w,   xrt::bo::flags::normal, k_conv1.group_id(1));
    auto bo_b_hr   = xrt::bo(device, bytes_hr_b,   xrt::bo::flags::normal, k_conv1.group_id(2));
    auto bo_w_last = xrt::bo(device, bytes_last_w, xrt::bo::flags::normal, k_conv2.group_id(1));
    auto bo_b_last = xrt::bo(device, bytes_last_b, xrt::bo::flags::normal, k_conv2.group_id(2));

    float* w_hr   = bo_w_hr.map<float*>();
    float* b_hr   = bo_b_hr.map<float*>();
    float* w_last = bo_w_last.map<float*>();
    float* b_last = bo_b_last.map<float*>();

    // load weights
    if (!load_bin(weights_dir + "/conv_hr/w.bin",   w_hr,   bytes_hr_w))   return 2;
    if (!load_bin(weights_dir + "/conv_hr/b.bin",   b_hr,   bytes_hr_b))   return 2;
    if (!load_bin(weights_dir + "/conv_last/w.bin", w_last, bytes_last_w)) return 2;
    if (!load_bin(weights_dir + "/conv_last/b.bin", b_last, bytes_last_b)) return 2;

    bo_w_hr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b_hr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_w_last.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b_last.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // conv_hr: C -> C on (4H,4W)
    { auto r = k_conv1(bo_u2_out, bo_w_hr, bo_b_hr, bo_hr_out, H4, W4, C, C); r.wait(); }
    // lrelu(0.1)
    { auto r = k_relu(bo_hr_out, bo_hr_out, H4, W4, C, 0.1f); r.wait(); }
    // conv_last: C -> 3
    { auto r = k_conv2(bo_hr_out, bo_w_last, bo_b_last, bo_final_out, H4, W4, C, C_out_last); r.wait(); }

    // dump final output
    bo_final_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
#if DEBUG_MODE
    dump_bin("final_out.bin", bo_final_out.map<void*>(), bytes_4xRGB);
    dump_csv("final_out.csv", bo_final_out.map<float*>(), H4, W4, C_out_last);
#endif



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////




    if (!ref_out_path.empty()) {
      std::vector<float> ref((size_t)H4*W4*C_out_last);
      if (load_bin(ref_out_path, ref.data(), bytes_4xRGB)) {
        compare_and_report(bo_final_out.map<const float*>(), ref.data(),
                           (size_t)H4*W4*C_out_last, H4, W4, C_out_last, "FINAL");
      } else {
        std::cerr << "[WARN] Couldn't load ref_out: " << ref_out_path << "\n";
      }
    }

    // ---------------- (NEW) 与 base.bin 相加 + 输出图片 ----------------
    if (!base_bin_path.empty()) {
      std::cout << "[POST] Loading base and adding to final_out ...\n";
      std::vector<float> base((size_t)H4*W4*C_out_last);
      if (!load_bin(base_bin_path, base.data(), bytes_4xRGB)) {
        std::cerr << "[ERROR] failed to load base_bin: " << base_bin_path << "\n";
        return 2;
      }
      const float* final_ptr = bo_final_out.map<const float*>();
      std::vector<float> sum((size_t)H4*W4*C_out_last);
      for (size_t i=0;i<sum.size();++i) sum[i] = final_ptr[i] + base[i];

      // 保存 bin/csv
      {
        std::ofstream fb("final_plus_base.bin", std::ios::binary);
        fb.write(reinterpret_cast<const char*>(sum.data()), (std::streamsize)bytes_4xRGB);
      }
#if DEBUG_MODE
      dump_csv("final_plus_base.csv", sum.data(), H4, W4, C_out_last);
#endif

      // 写图（PPM）
      if (!out_img_path.empty()) {
        // 不管扩展名是什么，都写 PPM（如果你想要 PNG 我可以给你换成 lodepng 版本）
        std::string ppm_path = out_img_path;
        // 若没有 .ppm 后缀，自动补一个
        if (ppm_path.size() < 4 || ppm_path.substr(ppm_path.size()-4) != ".ppm")
          ppm_path += ".ppm";
        bool ok = write_ppm_from_nhwc_float(ppm_path, sum.data(), H4, W4);
        if (ok) std::cout << "[POST] Wrote image: " << ppm_path << "\n";
        else    std::cerr << "[POST] Write image failed: " << ppm_path << "\n";
      }
    } else {
      // 如果没给 base.bin，但给了 out_img_path，我们也可直接把 final_out 写图，便于可视化
      if (!out_img_path.empty()) {
        std::string ppm_path = out_img_path;
        if (ppm_path.size() < 4 || ppm_path.substr(ppm_path.size()-4) != ".ppm")
          ppm_path += ".ppm";
        bool ok = write_ppm_from_nhwc_float(ppm_path, bo_final_out.map<const float*>(), H4, W4);
        if (ok) std::cout << "[POST] Wrote image (final_out only): " << ppm_path << "\n";
        else    std::cerr << "[POST] Write image failed: " << ppm_path << "\n";
      }
    }

    std::cout<<"[DONE] BODY + U1 + U2 + conv_hr + conv_last ok.\n";

  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 2;
  }
  return 0;
}
