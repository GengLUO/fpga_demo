// // pixelshuffle2x_ddr.cpp
// #include <stdint.h>

// extern "C" void pixelshuffle2x_ddr(
//     const float* in,   // [H*W*(4C)] NHWC
//     float* out,        // [(2H)*(2W)*C] NHWC
//     int H, int W, int C   // C = output channels
// ){
// #pragma HLS INTERFACE m_axi     port=in   offset=slave bundle=gmem0
// #pragma HLS INTERFACE m_axi     port=out  offset=slave bundle=gmem1
// #pragma HLS INTERFACE s_axilite port=in       bundle=control
// #pragma HLS INTERFACE s_axilite port=out      bundle=control
// #pragma HLS INTERFACE s_axilite port=H        bundle=control
// #pragma HLS INTERFACE s_axilite port=W        bundle=control
// #pragma HLS INTERFACE s_axilite port=C        bundle=control
// #pragma HLS INTERFACE s_axilite port=return   bundle=control

//     const int outH = H * 2;
//     const int outW = W * 2;
//     const int Cin  = 4 * C;

//     auto idx_in  = [W, Cin](int y, int x, int c)->long  { return ((long)y * W + (long)x) * Cin + c; };
//     auto idx_out = [outW, C](int y, int x, int c)->long { return ((long)y * outW + (long)x) * C + c; };

// OUT_Y:
//     for (int y = 0; y < H; ++y) {
//     OUT_X:
//         for (int x = 0; x < W; ++x) {
//         CH:
//             for (int c = 0; c < C; ++c) {
// #pragma HLS PIPELINE II=1
//                 // 输入通道布局：c, c+C, c+2C, c+3C  -> (2x,2y) 四个子像素
//                 float v00 = in[idx_in(y, x, c + 0*C)]; // (2y, 2x)
//                 float v01 = in[idx_in(y, x, c + 1*C)]; // (2y, 2x+1)
//                 float v10 = in[idx_in(y, x, c + 2*C)]; // (2y+1, 2x)
//                 float v11 = in[idx_in(y, x, c + 3*C)]; // (2y+1, 2x+1)

//                 int oy = y << 1;
//                 int ox = x << 1;

//                 out[idx_out(oy,   ox,   c)] = v00;
//                 out[idx_out(oy,   ox+1, c)] = v01;
//                 out[idx_out(oy+1, ox,   c)] = v10;
//                 out[idx_out(oy+1, ox+1, c)] = v11;
//             }
//         }
//     }
// }

// pixelshuffle2x_ddr.cpp
#include <stdint.h>

extern "C" void pixelshuffle2x_ddr(
    const float* in,   // [H * W * (4*C)] NHWC
    float* out,        // [(2H) * (2W) * C] NHWC
    int H, int W, int C)  // C = output channels
{
#pragma HLS INTERFACE m_axi     port=in   offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi     port=out  offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=in       bundle=control
#pragma HLS INTERFACE s_axilite port=out      bundle=control
#pragma HLS INTERFACE s_axilite port=H        bundle=control
#pragma HLS INTERFACE s_axilite port=W        bundle=control
#pragma HLS INTERFACE s_axilite port=C        bundle=control
#pragma HLS INTERFACE s_axilite port=return   bundle=control

    const int outH = H * 2;
    const int outW = W * 2;
    const int Cin  = 4 * C;

    // NHWC 线性访问: idx = (y * W + x) * C + c
    // PyTorch PixelShuffle(2): out[2y+i, 2x+j, c] = in[y, x, 4*c + i*2 + j]
OUT_Y:
    for (int y = 0; y < H; ++y) {
    OUT_X:
        for (int x = 0; x < W; ++x) {
        CH:
            for (int c = 0; c < C; ++c) {
#pragma HLS PIPELINE II=1

                // 通道基址
                const int base_in = (y * W + x) * Cin + (c << 2); // 4*c
                const int oy = y << 1;
                const int ox = x << 1;

                // 子像素四个分支，对应 (i,j)=(0,0),(0,1),(1,0),(1,1)
                const float v00 = in[base_in + 0]; // (i=0,j=0)
                const float v01 = in[base_in + 1]; // (i=0,j=1)
                const float v10 = in[base_in + 2]; // (i=1,j=0)
                const float v11 = in[base_in + 3]; // (i=1,j=1)

                const int base_out_00 = (oy * outW + ox) * C + c;
                const int base_out_01 = (oy * outW + (ox + 1)) * C + c;
                const int base_out_10 = ((oy + 1) * outW + ox) * C + c;
                const int base_out_11 = ((oy + 1) * outW + (ox + 1)) * C + c;

                out[base_out_00] = v00;
                out[base_out_01] = v01;
                out[base_out_10] = v10;
                out[base_out_11] = v11;
            }
        }
    }
}
