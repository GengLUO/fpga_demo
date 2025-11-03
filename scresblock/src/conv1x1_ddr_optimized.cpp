#include <stdint.h>
#include <hls_stream.h>
#include <ap_int.h>

extern "C" void conv1x1_ddr_optimized(
    const float* in,          // [H*W*C_in]  NHWC, contiguous
    const float* weight,      // [C_out*C_in], row-major: w[co*C_in + ci]
    const float* bias,        // [C_out] (nullable)
    float* out,               // [H*W*C_out]
    int H, int W, int C_in, int C_out
) {
#pragma HLS INTERFACE m_axi     port=in     offset=slave bundle=gmem0  depth=4096  max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi     port=weight offset=slave bundle=gmem1  depth=4096  max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi     port=bias   offset=slave bundle=gmem2  depth=4096  max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi     port=out    offset=slave bundle=gmem3  depth=4096  max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=in         bundle=control
#pragma HLS INTERFACE s_axilite port=weight     bundle=control
#pragma HLS INTERFACE s_axilite port=bias       bundle=control
#pragma HLS INTERFACE s_axilite port=out        bundle=control
#pragma HLS INTERFACE s_axilite port=H          bundle=control
#pragma HLS INTERFACE s_axilite port=W          bundle=control
#pragma HLS INTERFACE s_axilite port=C_in       bundle=control
#pragma HLS INTERFACE s_axilite port=C_out      bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

    // ===== 参数与上限（可按板卡资源调）======
    const int MAXC = 128;     // 允许 Cin/Cout ≤ 128；64×64≈16KB，128×128≈64KB
    const int TCin  = 16;     // Cin 分块大小（配合 ARRAY_PARTITION 完全展开）
    const int TCout = 16;     // Cout 分块大小
#pragma HLS INLINE off

    // ===== 片上常驻权重与偏置 =====
    static float W_local[MAXC][MAXC]; // [co][ci]
#pragma HLS BIND_STORAGE   variable=W_local type=RAM_2P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=W_local dim=2 factor=16 cyclic
    static float B_local[MAXC];
#pragma HLS BIND_STORAGE   variable=B_local type=RAM_1P impl=BRAM

    const int HW = H * W;

    // --- 安全检查（防越界；综合时常量化也可去掉）---
    if (C_in > MAXC || C_out > MAXC || H <= 0 || W <= 0) {
        return;
    }

    // ===== 1) 一次性把权重 / 偏置搬上片 =====
    // 权重 row-major: weight[co*C_in + ci]
    Weight_Load_Co:
    for (int co = 0; co < C_out; ++co) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
    Weight_Load_Ci:
        for (int ci = 0; ci < C_in; ++ci) {
#pragma HLS PIPELINE II=1
            W_local[co][ci] = weight[(long)co * C_in + ci];
        }
    }

    if (bias) {
    Bias_Load:
        for (int co = 0; co < C_out; ++co) {
#pragma HLS PIPELINE II=1
            B_local[co] = bias[co];
        }
    } else {
    Bias_Zero:
        for (int co = 0; co < C_out; ++co) {
#pragma HLS PIPELINE II=1
            B_local[co] = 0.0f;
        }
    }

    // ===== 2) 主计算：像素流式，通道分块 + 阵列乘加 =====
    // 结构：对每个像素 p，分块迭代 co_t (TCout)：
    //       初始化 acc[TCout] = B_local 切片；
    //       循环 ci_t (TCin)：读取 x_blk[TCin] + W_blk[TCout][TCin]，做完全展开 MAC；
    //       最后写回 acc。
    Pixel_Loop:
    for (int p = 0; p < HW; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=1
        const float* in_pix = &in[(long)p * C_in];

    Co_Tile_Loop:
        for (int co0 = 0; co0 < C_out; co0 += TCout) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=8
            const int tco = ((co0 + TCout) <= C_out) ? TCout : (C_out - co0);

            // 局部累加器（一个 tile 的 Cout）
            float acc_blk[TCout];
#pragma HLS ARRAY_PARTITION variable=acc_blk complete
        Acc_Init:
            for (int to = 0; to < tco; ++to) {
#pragma HLS UNROLL
                acc_blk[to] = B_local[co0 + to];
            }

        Ci_Tile_Loop:
            for (int ci0 = 0; ci0 < C_in; ci0 += TCin) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=8
                const int tci = ((ci0 + TCin) <= C_in) ? TCin : (C_in - ci0);

                // 读入本像素的输入切片 x_blk[TCin]
                float x_blk[TCin];
#pragma HLS ARRAY_PARTITION variable=x_blk complete
            X_Load:
                for (int ti = 0; ti < tci; ++ti) {
#pragma HLS PIPELINE II=1
                    x_blk[ti] = in_pix[ci0 + ti];
                }

                // 从片上权重阵列取出当前 (co,ci) tile
                float w_blk[TCout][TCin]; // 仅用于指导分块并行（综合器会优化）
#pragma HLS ARRAY_PARTITION variable=w_blk dim=1 complete
#pragma HLS ARRAY_PARTITION variable=w_blk dim=2 complete
            W_Load:
                for (int to = 0; to < tco; ++to) {
#pragma HLS UNROLL
                W_Load_Inner:
                    for (int ti = 0; ti < tci; ++ti) {
#pragma HLS UNROLL
                        w_blk[to][ti] = W_local[co0 + to][ci0 + ti];
                    }
                }

                // 完全展开的 MAC 阵列：to×ti
            MAC_Array:
                for (int to = 0; to < tco; ++to) {
#pragma HLS UNROLL
                MAC_Inner:
                    for (int ti = 0; ti < tci; ++ti) {
#pragma HLS UNROLL
                        acc_blk[to] += w_blk[to][ti] * x_blk[ti];
                    }
                }
            } // end Ci_Tile_Loop

            // 写回此 tile 的输出
        Out_Store:
            for (int to = 0; to < tco; ++to) {
#pragma HLS PIPELINE II=1
                out[(long)p * C_out + (co0 + to)] = acc_blk[to];
            }
        } // end Co_Tile_Loop
    } // end Pixel_Loop
}
