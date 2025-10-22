#include <stdint.h>

static inline int idx_nhwc(int y, int x, int c, int H, int W, int C) {
#pragma HLS INLINE
    return ((y * W + x) * C + c);
}

extern "C" void shift8_ddr(
    const float* in,  // [H*W*C] NHWC
    float* out,       // [H*W*C] NHWC
    int H, int W, int C
){
#pragma HLS INTERFACE m_axi     port=in   offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi     port=out  offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=in       bundle=control
#pragma HLS INTERFACE s_axilite port=out      bundle=control
#pragma HLS INTERFACE s_axilite port=H        bundle=control
#pragma HLS INTERFACE s_axilite port=W        bundle=control
#pragma HLS INTERFACE s_axilite port=C        bundle=control
#pragma HLS INTERFACE s_axilite port=return   bundle=control

    // 要求 C 是 8 的倍数：分 8 片
    if ((C & 7) != 0) return;
    const int g = C >> 3;

    const int dy[8] = {-1, +1,  0,  0, +1, +1, -1, -1};
    const int dx[8] = { 0,  0, -1, +1, +1, -1, +1, -1};

Row:for (int y = 0; y < H; ++y) {
Col:    for (int x = 0; x < W; ++x) {
            const int out_base = idx_nhwc(y, x, 0, H, W, C);

Dir:        for (int i = 0; i < 8; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
                const int ys = y + dy[i];
                const int xs = x + dx[i];
                const int c0 = i * g;

                if (ys < 0 || ys >= H || xs < 0 || xs >= W) {
                ZLP:    for (int k = 0; k < g; ++k) {
#pragma HLS PIPELINE II=1
                            out[out_base + c0 + k] = 0.0f;
                        }
                } else {
                    const int src_base = idx_nhwc(ys, xs, 0, H, W, C) + c0;
                CPY:    for (int k = 0; k < g; ++k) {
#pragma HLS PIPELINE II=1
                            out[out_base + c0 + k] = in[src_base + k];
                        }
                }
            }
        }
    }
}
