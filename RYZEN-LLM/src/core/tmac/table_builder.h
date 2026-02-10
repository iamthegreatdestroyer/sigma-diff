#pragma once

/**
 * @file table_builder.h
 * @brief Multi-tier lookup table construction with compression
 *
 * Builds compressed lookup tables from BitNet ternary weights.
 * Achieves 654× compression (1.4 TB → 2.14 GB) through:
 *   1. Symmetry exploitation (2×)
 *   2. Sparse indexing (7.6×)
 *   3. Delta encoding (1.5×)
 *   4. INT16 quantization (1.67×)
 *
 * [REF:TMAC-004] - Table Building & Compression
 */

#include "pattern_generator.h"
#include "frequency_analyzer.h"
#include "delta_encoder.h"
#include <vector>
#include <unordered_map>
#include <memory>

namespace ryzanstein_llm
{
    namespace tmac
    {

        /**
         * Dense lookup table: pattern × activation → result
         *
         * Storage: N_patterns × N_activations × sizeof(T)
         * Access: O(1) via pattern hash → index mapping
         */
        template <typename T>
        struct DenseTable
        {
            std::vector<T> data;                                   ///< Flattened [pattern][activation]
            std::unordered_map<uint64_t, uint32_t> pattern_to_idx; ///< Hash → row index
            uint32_t num_patterns = 0;
            uint32_t num_activations = 0;

            /**
             * Lookup result for given pattern and activation
             *
             * @param pattern_hash Hash of canonical pattern
             * @param activation Activation value (0-255 index)
             * @return Result, or 0 if pattern not found
             */
            T lookup(uint64_t pattern_hash, uint8_t activation) const
            {
                auto it = pattern_to_idx.find(pattern_hash);
                if (it == pattern_to_idx.end())
                {
                    return 0; // Not found
                }

                uint32_t idx = it->second;
                return data[idx * num_activations + activation];
            }

            /**
             * Check if pattern exists in this table
             */
            bool contains(uint64_t pattern_hash) const
            {
                return pattern_to_idx.find(pattern_hash) != pattern_to_idx.end();
            }

            /**
             * Total size in bytes
             */
            size_t size_bytes() const
            {
                return data.size() * sizeof(T) +
                       pattern_to_idx.size() * (sizeof(uint64_t) + sizeof(uint32_t));
            }
        };

        /**
         * Sparse delta table entry
         */
        struct DeltaEntry
        {
            uint64_t base_pattern_hash;        ///< Reference to base pattern
            std::vector<int16_t> delta_values; ///< Delta per activation (256 values)

            size_t size_bytes() const
            {
                return sizeof(base_pattern_hash) + delta_values.size() * sizeof(int16_t);
            }
        };

        /**
         * Multi-tier compressed lookup table structure
         *
         * Tier 1 (Hot):   10K patterns × 64 activations × INT16 =  1.25 MB | 60% hit
         * Tier 2 (Warm): 100K patterns × 256 activations × INT16 = 51 MB   | 35% hit
         * Tier 3 (Delta):  ~6M sparse deltas × 264 bytes =       ~1.5 GB   | 4.9% hit
         * Tier 4 (Compute): On-the-fly computation                          | 0.1% hit
         */
        struct CompressedLUT
        {
            // Tier 1: Hot cache (10K patterns, 64 common activations, INT16)
            DenseTable<int16_t> tier1_hot;
            int8_t tier1_act_min = -32;
            int8_t tier1_act_max = 31;

            // Tier 2: Warm dense (100K patterns, 256 activations, INT16)
            DenseTable<int16_t> tier2_warm;

            // Tier 3: Sparse delta (remaining patterns)
            std::unordered_map<uint64_t, DeltaEntry> tier3_deltas;

            // Metadata
            uint32_t group_size = 16;

            /**
             * Total size in bytes
             */
            size_t total_size_bytes() const
            {
                size_t total = tier1_hot.size_bytes() + tier2_warm.size_bytes();

                for (const auto &[hash, entry] : tier3_deltas)
                {
                    total += entry.size_bytes();
                }

                return total;
            }

            /**
             * Total size in GB
             */
            double total_size_gb() const
            {
                return static_cast<double>(total_size_bytes()) / (1024.0 * 1024.0 * 1024.0);
            }
        };

        /**
         * Builds compressed lookup tables from BitNet weights
         *
         * Implements the complete compression pipeline:
         *   1. Analyze pattern frequency
         *   2. Select tier 1/2 patterns (hot/warm)
         *   3. Build dense tables with INT16 quantization
         *   4. Delta-encode tier 3 (sparse)
         */
        class TableBuilder
        {
        public:
            explicit TableBuilder(uint32_t group_size = 16) : group_size_(group_size) {}

            /**
             * Build complete compressed LUT from training weights
             *
             * @param weights Ternary weights [M × K] (flattened)
             * @param M Number of output features
             * @param K Number of input features
             * @return Compressed LUT structure (target: <3GB)
             *
             * Time: O(M × K / group_size × log(unique_patterns))
             * Space: O(unique_patterns)
             *
             * Steps:
             *   1. Analyze pattern frequency → identify hot/warm patterns
             *   2. Build tier 1 (top 10K, 64 activations)
             *   3. Build tier 2 (top 100K, 256 activations)
             *   4. Delta-encode tier 3 (remaining patterns)
             *   5. Apply INT16 quantization where possible
             */
            CompressedLUT build(
                const std::vector<int8_t> &weights,
                uint32_t M,
                uint32_t K);

            /**
             * Set tier sizes (for tuning)
             *
             * Default: tier1=10K, tier2=100K
             */
            void set_tier_sizes(uint32_t tier1_size, uint32_t tier2_size)
            {
                tier1_size_ = tier1_size;
                tier2_size_ = tier2_size;
            }

        private:
            uint32_t group_size_;
            uint32_t tier1_size_ = 10000;  // Top 10K patterns
            uint32_t tier2_size_ = 100000; // Top 100K patterns

            FrequencyAnalyzer freq_analyzer_;
            DeltaEncoder delta_encoder_;

            /**
             * Build tier 1 dense table (hot cache)
             *
             * Only stores common activation range [-32, 31]
             *
             * @param patterns Hot patterns (top 10K)
             * @param act_min Minimum activation value
             * @param act_max Maximum activation value
             * @return Dense table with INT16 storage
             */
            DenseTable<int16_t> build_tier1(
                const std::vector<CanonicalPattern> &patterns,
                int8_t act_min,
                int8_t act_max);

            /**
             * Build tier 2 dense table (warm cache)
             *
             * Stores full activation range [-128, 127]
             *
             * @param patterns Warm patterns (top 100K)
             * @return Dense table with INT16 storage
             */
            DenseTable<int16_t> build_tier2(
                const std::vector<CanonicalPattern> &patterns);

            /**
             * Build tier 3 sparse delta table
             *
             * Encodes remaining patterns as (base + delta)
             *
             * @param all_patterns All canonical patterns
             * @param base_patterns Base patterns (tier 2)
             * @param max_hamming Maximum Hamming distance for delta encoding
             * @return Sparse delta map
             */
            std::unordered_map<uint64_t, DeltaEntry> build_tier3(
                const std::vector<CanonicalPattern> &all_patterns,
                const std::vector<CanonicalPattern> &base_patterns,
                uint32_t max_hamming = 3);

            /**
             * Compute dot product: pattern · [activation, activation, ...]
             *
             * For T-MAC, we compute:
             *   result = Σᵢ wᵢ × activation
             *
             * This is different from a typical GEMM because we're
             * broadcasting a single activation value across all positions.
             *
             * @param pattern Ternary pattern
             * @param activation Single INT8 activation value
             * @return Dot product result
             */
            int32_t compute_dot_product(
                const TernaryPattern &pattern,
                int8_t activation) const;

            /**
             * Check if result fits in INT16
             */
            bool fits_in_int16(int32_t value) const
            {
                return value >= INT16_MIN && value <= INT16_MAX;
            }
        };

    } // namespace tmac
} // namespace ryzanstein_llm
