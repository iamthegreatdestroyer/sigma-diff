/**
 * @file lut_lookup.cpp
 * @brief Implementation of runtime T-MAC lookup
 *
 * [REF:TMAC-005] - Runtime Lookup
 */

#include "lut_lookup.h"
#include <iostream>
#include <iomanip>

namespace ryzen_llm
{
    namespace tmac
    {

        LUTLookup::LUTLookup(std::shared_ptr<CompressedLUT> lut)
            : lut_(std::move(lut))
        {
            if (!lut_)
            {
                throw std::invalid_argument("LUT cannot be null");
            }

            std::cout << "Initialized LUTLookup engine\n";
            std::cout << "  Total size: " << lut_->total_size_gb() << " GB\n";
            std::cout << "  Tier 1 patterns: " << lut_->tier1_hot.num_patterns << "\n";
            std::cout << "  Tier 2 patterns: " << lut_->tier2_warm.num_patterns << "\n";
            std::cout << "  Tier 3 patterns: " << lut_->tier3_deltas.size() << "\n";
        }

        int32_t LUTLookup::lookup(
            const TernaryPattern &pattern,
            int8_t activation)
        {
            // Step 1: Canonicalize pattern
            auto canonical = pattern_gen_.canonicalize(pattern);
            uint64_t hash = canonical.hash();

            // Step 2: Tier 1 lookup (hot cache)
            int32_t tier1_idx = get_tier1_index(activation);
            if (tier1_idx >= 0 && lut_->tier1_hot.contains(hash))
            {
                int16_t result = lut_->tier1_hot.lookup(hash, tier1_idx);

                // Apply flip if needed
                int32_t final_result = canonical.flip ? -result : result;

                stats_.increment_tier1();
                return final_result;
            }

            // Step 3: Tier 2 lookup (warm cache)
            uint8_t tier2_idx = get_tier2_index(activation);
            if (lut_->tier2_warm.contains(hash))
            {
                int16_t result = lut_->tier2_warm.lookup(hash, tier2_idx);

                // Apply flip if needed
                int32_t final_result = canonical.flip ? -result : result;

                stats_.increment_tier2();
                return final_result;
            }

            // Step 4: Tier 3 lookup (delta reconstruction)
            auto tier3_it = lut_->tier3_deltas.find(hash);
            if (tier3_it != lut_->tier3_deltas.end())
            {
                const auto &delta_entry = tier3_it->second;

                // Get base result from tier 2
                int16_t base_result = lut_->tier2_warm.lookup(
                    delta_entry.base_pattern_hash,
                    tier2_idx);

                // Add delta
                int16_t delta = delta_entry.delta_values[tier2_idx];
                int32_t result = base_result + delta;

                // Apply flip if needed
                int32_t final_result = canonical.flip ? -result : result;

                stats_.increment_tier3();
                return final_result;
            }

            // Step 5: Fallback computation
            stats_.increment_fallback();
            return compute_fallback(pattern, activation);
        }

        void LUTLookup::lookup_batch(
            const TernaryPattern &pattern,
            const int8_t *activations,
            int32_t *results,
            uint32_t count)
        {
            // Canonicalize once for entire batch
            auto canonical = pattern_gen_.canonicalize(pattern);
            uint64_t hash = canonical.hash();

            // Check which tier contains this pattern
            bool in_tier1 = lut_->tier1_hot.contains(hash);
            bool in_tier2 = lut_->tier2_warm.contains(hash);
            bool in_tier3 = lut_->tier3_deltas.find(hash) != lut_->tier3_deltas.end();

            // Process batch
            for (uint32_t i = 0; i < count; ++i)
            {
                int8_t act = activations[i];

                if (in_tier1)
                {
                    int32_t tier1_idx = get_tier1_index(act);
                    if (tier1_idx >= 0)
                    {
                        int16_t result = lut_->tier1_hot.lookup(hash, tier1_idx);
                        results[i] = canonical.flip ? -result : result;
                        stats_.increment_tier1();
                        continue;
                    }
                }

                if (in_tier2)
                {
                    uint8_t tier2_idx = get_tier2_index(act);
                    int16_t result = lut_->tier2_warm.lookup(hash, tier2_idx);
                    results[i] = canonical.flip ? -result : result;
                    stats_.increment_tier2();
                    continue;
                }

                if (in_tier3)
                {
                    const auto &delta_entry = lut_->tier3_deltas.at(hash);
                    uint8_t tier2_idx = get_tier2_index(act);

                    int16_t base_result = lut_->tier2_warm.lookup(
                        delta_entry.base_pattern_hash,
                        tier2_idx);
                    int16_t delta = delta_entry.delta_values[tier2_idx];
                    int32_t result = base_result + delta;

                    results[i] = canonical.flip ? -result : result;
                    stats_.increment_tier3();
                    continue;
                }

                // Fallback
                results[i] = compute_fallback(pattern, act);
                stats_.increment_fallback();
            }
        }

        int32_t LUTLookup::compute_fallback(
            const TernaryPattern &pattern,
            int8_t activation)
        {
            int32_t result = 0;

            // Direct dot product computation
            for (int8_t w : pattern)
            {
                result += w * activation;
            }

            return result;
        }

        void LUTLookup::print_stats() const
        {
            // Use thread-safe accessor methods for atomic reads
            const uint64_t t1 = stats_.tier1_hits();
            const uint64_t t2 = stats_.tier2_hits();
            const uint64_t t3 = stats_.tier3_hits();
            const uint64_t fb = stats_.fallback_count();
            const uint64_t total = t1 + t2 + t3 + fb;

            if (total == 0)
            {
                std::cout << "No lookups performed yet.\n";
                return;
            }

            std::cout << "\nT-MAC Lookup Statistics\n";
            std::cout << "=======================\n";
            std::cout << "Total lookups: " << total << "\n\n";

            std::cout << std::fixed << std::setprecision(2);
            std::cout << "Tier 1 (hot):    " << std::setw(10) << t1
                      << " (" << (stats_.tier1_rate() * 100.0) << "%)\n";
            std::cout << "Tier 2 (warm):   " << std::setw(10) << t2
                      << " (" << (stats_.tier2_rate() * 100.0) << "%)\n";
            std::cout << "Tier 3 (delta):  " << std::setw(10) << t3
                      << " (" << (stats_.tier3_rate() * 100.0) << "%)\n";
            std::cout << "Fallback (comp): " << std::setw(10) << fb
                      << " (" << (stats_.fallback_rate() * 100.0) << "%)\n\n";

            std::cout << "Overall hit rate: " << (stats_.hit_rate() * 100.0) << "%\n";
            std::cout << "Target: >95% → "
                      << (stats_.hit_rate() > 0.95 ? "✓ PASS" : "✗ FAIL") << "\n";
        }

    } // namespace tmac
} // namespace ryzen_llm
