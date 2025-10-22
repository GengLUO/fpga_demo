#include <stdint.h>

extern "C" void add_residual_ddr(
    const float* x,     // identity: [H*W*C]
    const float* y,     // block out: [H*W*C]
    float* out,         // out = x + res_scale * y
    float res_scale,
    int H, int W, int C
){
#pragma HLS INTERFACE m_axi     port=x    offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi     port=y    offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=out  offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=x          bundle=control
#pragma HLS INTERFACE s_axilite port=y          bundle=control
#pragma HLS INTERFACE s_axilite port=out        bundle=control
#pragma HLS INTERFACE s_axilite port=res_scale  bundle=control
#pragma HLS INTERFACE s_axilite port=H          bundle=control
#pragma HLS INTERFACE s_axilite port=W          bundle=control
#pragma HLS INTERFACE s_axilite port=C          bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

    const long N = (long)H * W * C;

ELT:for (long i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        out[i] = x[i] + res_scale * y[i];
    }
}
