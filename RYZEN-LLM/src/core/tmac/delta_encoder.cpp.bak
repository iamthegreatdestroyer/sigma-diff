/**
 * @file delta_encoder.cpp
 * @brief Implementation of delta encoding for pattern compression
 *
 * [REF:TMAC-003] - Delta Compression
 */

#include "delta_encoder.h"
#include <algorithm>
#include <limits>

namespace ryzen_llm
{
    namespace tmac
    {

        std::optional<CanonicalPattern> DeltaEncoder::find_nearest_base(
            const CanonicalPattern &target,
            const std::vector<CanonicalPattern> &base_patterns,
            uint32_t max_distance)
        {
            if (base_patterns.empty())
            {
                return std::nullopt;
            }

            uint32_t min_distance = std::numeric_limits<uint32_t>::max();
            const CanonicalPattern *nearest = nullptr;

            // Linear search for nearest base
            // TODO: For large base sets (>10K), consider using LSH or k-d tree
            for (const auto &base : base_patterns)
            {
                uint32_t dist = hamming_distance(target.pattern, base.pattern);

                if (dist < min_distance)
                {
                    min_distance = dist;
                    nearest = &base;

                    // Early exit if we found an exact match
                    if (dist == 0)
                    {
                        break;
                    }
                }
            }

            // Return nearest only if within threshold
            if (nearest && min_distance <= max_distance)
            {
                return *nearest;
            }

            return std::nullopt;
        }

        uint32_t DeltaEncoder::hamming_distance(
            const TernaryPattern &p1,
            const TernaryPattern &p2) const
        {
            uint32_t distance = 0;

            for (size_t i = 0; i < p1.size(); ++i)
            {
                if (p1[i] != p2[i])
                {
                    ++distance;
                }
            }

            return distance;
        }

        std::vector<int16_t> DeltaEncoder::encode_delta(
            const CanonicalPattern &target,
            const CanonicalPattern &base)
        {
            std::vector<int16_t> delta_values;
            delta_values.reserve(256);

            // Compute delta for each activation value
            for (int16_t act = -128; act < 128; ++act)
            {
                int8_t activation = static_cast<int8_t>(act);

                int32_t target_result = compute_dot_product(target.pattern, activation);
                int32_t base_result = compute_dot_product(base.pattern, activation);

                // Apply flip flags
                if (target.flip)
                    target_result = -target_result;
                if (base.flip)
                    base_result = -base_result;

                int32_t delta = target_result - base_result;

                // Clamp to INT16 range (should always fit for Hamming distance ≤ 3)
                if (delta < INT16_MIN)
                    delta = INT16_MIN;
                if (delta > INT16_MAX)
                    delta = INT16_MAX;

                delta_values.push_back(static_cast<int16_t>(delta));
            }

            return delta_values;
        }

        int32_t DeltaEncoder::compute_dot_product(
            const TernaryPattern &pattern,
            int8_t activation) const
        {
            int32_t result = 0;

            // For ternary weights with a single activation value,
            // we need to broadcast the activation across all positions
            //
            // Actually, in T-MAC, we're computing the dot product of
            // a pattern (16 weights) with 16 activation values.
            // But for delta encoding, we need the result for a single
            // activation value repeated.
            //
            // So: result = Σᵢ wᵢ × activation

            for (int8_t w : pattern)
            {
                result += w * activation;
            }

            return result;
        }

    } // namespace tmac
} // namespace ryzen_llm
