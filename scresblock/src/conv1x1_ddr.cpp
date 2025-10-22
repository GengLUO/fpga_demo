#include <stdint.h>

extern "C" void conv1x1_ddr(
    const float* in,          // [H*W*C_in] NHWC
    const float* weight,      // [C_out*C_in], row-major: w[co*C_in + ci]
    const float* bias,        // [C_out] or nullptr
    float* out,               // [H*W*C_out]
    int H, int W, int C_in, int C_out
) {
#pragma HLS INTERFACE m_axi     port=in     offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi     port=weight offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=bias   offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi     port=out    offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=in         bundle=control
#pragma HLS INTERFACE s_axilite port=weight     bundle=control
#pragma HLS INTERFACE s_axilite port=bias       bundle=control
#pragma HLS INTERFACE s_axilite port=out        bundle=control
#pragma HLS INTERFACE s_axilite port=H          bundle=control
#pragma HLS INTERFACE s_axilite port=W          bundle=control
#pragma HLS INTERFACE s_axilite port=C_in       bundle=control
#pragma HLS INTERFACE s_axilite port=C_out      bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

    const int HW = H * W;

PIX: for (int p = 0; p < HW; ++p) {
        const float* in_pix = &in[(long)p * C_in];

    CO: for (int co = 0; co < C_out; ++co) {
            float acc = (bias ? bias[co] : 0.0f);
            const float* w_row = &weight[(long)co * C_in];

        CI: for (int ci = 0; ci < C_in; ++ci) {
#pragma HLS PIPELINE II=1
                acc += in_pix[ci] * w_row[ci];
            }
            out[(long)p * C_out + co] = acc;
        }
    }
}
