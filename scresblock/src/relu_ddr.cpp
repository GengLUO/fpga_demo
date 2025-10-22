#include <stdint.h>

extern "C" void relu_ddr(
    const float* in,  // [H*W*C] NHWC
    float* out,       // [H*W*C]
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

    const long N = (long)H * W * C;

ELT:for (long i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        float v = in[i];
        out[i] = (v > 0.f) ? v : 0.f;
    }
}
