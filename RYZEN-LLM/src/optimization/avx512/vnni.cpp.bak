// Copyright (c) 2024 RYZEN-LLM Project
// Licensed under MIT License
//
// VNNI INT8 Operations
// [REF:OL-005b] - Optimization Layer: AVX-512 SIMD Primitives
//
// This file implements INT8 quantized operations using AVX-512 VNNI
// for accelerated inference in Phase 2 optimization layer.

#include <immintrin.h>
#include <cstdint>

namespace ryzen_llm
{
    namespace avx512
    {

        class VNNI
        {
        public:
            VNNI() = default;
            ~VNNI() = default;

            /**
             * AVX-512 VNNI INT8 Dot Product
             */
            void DotProductINT8(const int8_t *a, const int8_t *b, int32_t *result, size_t size)
            {
                __m512i sum = _mm512_setzero_si512();

                for (size_t i = 0; i < size; i += 64)
                {
                    __m512i va = _mm512_loadu_si512((__m512i *)&a[i]);
                    __m512i vb = _mm512_loadu_si512((__m512i *)&b[i]);
                    __m512i prod = _mm512_dpbusds_epi32(sum, va, vb);
                    sum = _mm512_add_epi32(sum, prod);
                }

                *result = _mm512_reduce_add_epi32(sum);
            }

            /**
             * Quantize FP32 to INT8
             */
            void QuantizeINT8(const float *input, int8_t *output, size_t size,
                              float scale, int8_t zero_point)
            {
                __m512 scale_vec = _mm512_set1_ps(scale);
                __m512 zp_vec = _mm512_set1_ps(static_cast<float>(zero_point));

                for (size_t i = 0; i < size; i += 16)
                {
                    __m512 x = _mm512_loadu_ps(&input[i]);
                    __m512 scaled = _mm512_div_ps(x, scale_vec);
                    __m512 rounded = _mm512_roundscale_ps(scaled, _MM_FROUND_TO_NEAREST_INT);
                    __m512 shifted = _mm512_add_ps(rounded, zp_vec);

                    __m512 min_val = _mm512_set1_ps(-128.0f);
                    __m512 max_val = _mm512_set1_ps(127.0f);
                    __m512 clamped = _mm512_min_ps(_mm512_max_ps(shifted, min_val), max_val);

                    __m512i int_vals = _mm512_cvtps_epi32(clamped);
                    __m128i packed = _mm512_cvtepi32_epi8(int_vals);
                    _mm_storeu_si128((__m128i *)&output[i], packed);
                }
            }

            /**
             * Dequantize INT32 to FP32
             */
            void DequantizeFP32(const int32_t *input, float *output, size_t size, float scale)
            {
                __m512 scale_vec = _mm512_set1_ps(scale);

                for (size_t i = 0; i < size; i += 16)
                {
                    __m512i int_vals = _mm512_loadu_si512((__m512i *)&input[i]);
                    __m512 float_vals = _mm512_cvtepi32_ps(int_vals);
                    __m512 result = _mm512_mul_ps(float_vals, scale_vec);
                    _mm512_storeu_ps(&output[i], result);
                }
            }
        };

    } // namespace avx512
} // namespace ryzen_llm
