/**
 * @file table_builder.cpp
 * @brief Implementation of multi-tier table building
 *
 * [REF:TMAC-004] - Table Building & Compression
 */

#include "table_builder.h"
#include <iostream>
#include <algorithm>

namespace ryzen_llm
{
    namespace tmac
    {

        CompressedLUT TableBuilder::build(
            const std::vector<int8_t> &weights,
            uint32_t M,
            uint32_t K)
        {
            std::cout << "Building compressed T-MAC lookup tables...\n";
            std::cout << "  Weight matrix: " << M << " × " << K << "\n";
            std::cout << "  Group size: " << group_size_ << "\n\n";

            // Step 1: Analyze pattern frequency
            std::cout << "[1/5] Analyzing pattern frequency..." << std::flush;
            auto frequencies = freq_analyzer_.analyze_weights(weights, M, K, group_size_);
            std::cout << " Done. Found " << frequencies.size() << " unique patterns.\n";

            // Print frequency summary
            freq_analyzer_.print_summary(frequencies, 10);
            std::cout << "\n";

            // Step 2: Extract hot and warm patterns
            std::cout << "[2/5] Selecting tier patterns..." << std::flush;
            auto tier1_patterns = freq_analyzer_.get_top_k(frequencies, tier1_size_);
            auto tier2_patterns = freq_analyzer_.get_top_k(frequencies, tier2_size_);

            double tier1_coverage = freq_analyzer_.compute_coverage(frequencies, tier1_size_);
            double tier2_coverage = freq_analyzer_.compute_coverage(frequencies, tier2_size_);

            std::cout << " Done.\n";
            std::cout << "  Tier 1 (hot):  " << tier1_patterns.size() << " patterns ("
                      << (tier1_coverage * 100.0) << "% coverage)\n";
            std::cout << "  Tier 2 (warm): " << tier2_patterns.size() << " patterns ("
                      << (tier2_coverage * 100.0) << "% coverage)\n\n";

            // Step 3: Build tier 1 and tier 2 tables
            std::cout << "[3/5] Building tier 1 (hot cache)..." << std::flush;
            auto tier1 = build_tier1(tier1_patterns, -32, 31);
            std::cout << " Done. Size: " << (tier1.size_bytes() / 1024.0 / 1024.0) << " MB\n";

            std::cout << "[4/5] Building tier 2 (warm cache)..." << std::flush;
            auto tier2 = build_tier2(tier2_patterns);
            std::cout << " Done. Size: " << (tier2.size_bytes() / 1024.0 / 1024.0) << " MB\n";

            // Step 4: Generate all canonical patterns for tier 3
            std::cout << "[5/5] Building tier 3 (delta encoding)..." << std::flush;

            // We need all patterns to know which ones aren't in tier 1/2
            // For now, we'll just create tier 3 from frequency analysis
            std::vector<CanonicalPattern> tier3_candidates;
            for (size_t i = tier2_size_; i < frequencies.size(); ++i)
            {
                tier3_candidates.push_back(frequencies[i].pattern);
            }

            auto tier3 = build_tier3(tier3_candidates, tier2_patterns, 3);
            std::cout << " Done. Entries: " << tier3.size()
                      << ", Size: " << (tier3.size() * 264 / 1024.0 / 1024.0) << " MB (est.)\n\n";

            // Step 5: Assemble final structure
            CompressedLUT lut;
            lut.tier1_hot = std::move(tier1);
            lut.tier1_act_min = -32;
            lut.tier1_act_max = 31;
            lut.tier2_warm = std::move(tier2);
            lut.tier3_deltas = std::move(tier3);
            lut.group_size = group_size_;

            // Print final statistics
            std::cout << "Table building complete!\n";
            std::cout << "  Total size: " << lut.total_size_gb() << " GB\n";
            std::cout << "  Target: <3.0 GB → "
                      << (lut.total_size_gb() < 3.0 ? "✓ PASS" : "✗ FAIL") << "\n";

            return lut;
        }

        DenseTable<int16_t> TableBuilder::build_tier1(
            const std::vector<CanonicalPattern> &patterns,
            int8_t act_min,
            int8_t act_max)
        {
            DenseTable<int16_t> table;

            uint32_t num_acts = act_max - act_min + 1; // e.g., 64 for [-32, 31]
            table.num_patterns = patterns.size();
            table.num_activations = num_acts;

            // Allocate storage: [patterns][activations]
            table.data.resize(patterns.size() * num_acts);

            // Build table
            for (size_t p_idx = 0; p_idx < patterns.size(); ++p_idx)
            {
                const auto &pattern = patterns[p_idx];
                uint64_t hash = pattern.hash();

                // Store pattern mapping
                table.pattern_to_idx[hash] = p_idx;

                // Compute results for all activation values
                // Use int16_t to avoid overflow when act_max=127 (int8_t max)
                for (int16_t act = act_min; act <= act_max; ++act)
                {
                    uint32_t act_idx = static_cast<uint32_t>(act - act_min);

                    int32_t result = compute_dot_product(pattern.pattern, static_cast<int8_t>(act));

                    // Apply flip if needed
                    if (pattern.flip)
                    {
                        result = -result;
                    }

                    // Quantize to INT16 (should always fit for this range)
                    if (!fits_in_int16(result))
                    {
                        std::cerr << "Warning: Result " << result
                                  << " exceeds INT16 range at tier 1\n";
                    }

                    int16_t quantized = static_cast<int16_t>(result);
                    table.data[p_idx * num_acts + act_idx] = quantized;
                }
            }

            return table;
        }

        DenseTable<int16_t> TableBuilder::build_tier2(
            const std::vector<CanonicalPattern> &patterns)
        {
            return build_tier1(patterns, -128, 127); // Full INT8 range
        }

        std::unordered_map<uint64_t, DeltaEntry> TableBuilder::build_tier3(
            const std::vector<CanonicalPattern> &all_patterns,
            const std::vector<CanonicalPattern> &base_patterns,
            uint32_t max_hamming)
        {
            std::unordered_map<uint64_t, DeltaEntry> delta_map;

            size_t encoded_count = 0;
            size_t skipped_count = 0;

            for (const auto &pattern : all_patterns)
            {
                // Find nearest base
                auto nearest = delta_encoder_.find_nearest_base(pattern, base_patterns, max_hamming);

                if (nearest.has_value())
                {
                    // Encode as delta
                    auto delta_values = delta_encoder_.encode_delta(pattern, nearest.value());

                    DeltaEntry entry;
                    entry.base_pattern_hash = nearest.value().hash();
                    entry.delta_values = std::move(delta_values);

                    delta_map[pattern.hash()] = std::move(entry);
                    encoded_count++;
                }
                else
                {
                    // Too far from any base → skip (will use tier 4 fallback)
                    skipped_count++;
                }
            }

            if (skipped_count > 0)
            {
                std::cout << "\n  Warning: " << skipped_count << " patterns skipped (Hamming > "
                          << max_hamming << "). Will use fallback computation.\n";
            }

            return delta_map;
        }

        int32_t TableBuilder::compute_dot_product(
            const TernaryPattern &pattern,
            int8_t activation) const
        {
            int32_t result = 0;

            // For T-MAC with a single activation value:
            // result = Σᵢ wᵢ × activation
            //
            // This effectively broadcasts the activation across all positions
            for (int8_t w : pattern)
            {
                result += w * activation;
            }

            return result;
        }

    } // namespace tmac
} // namespace ryzen_llm
