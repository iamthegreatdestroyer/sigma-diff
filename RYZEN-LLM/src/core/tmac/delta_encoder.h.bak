#pragma once

/**
 * @file delta_encoder.h
 * @brief Delta encoding for sparse pattern representation
 *
 * Finds nearest base pattern via Hamming distance and encodes
 * the target pattern as (base + delta). Achieves ~1.5× compression
 * for patterns with small Hamming distances.
 *
 * [REF:TMAC-003] - Delta Compression
 */

#include "pattern_generator.h"
#include <vector>
#include <optional>

namespace ryzen_llm
{
    namespace tmac
    {

        /**
         * Delta encoder for sparse pattern representation
         *
         * Implements Compression Technique 3 from TMAC_COMPRESSION_ANALYSIS.md:
         * - Adjacent patterns often differ by only a few positions (Hamming distance)
         * - Store: (base_id, position_mask, delta_values) instead of full table
         * - Compression: ~3.9× for patterns with H ≤ 3
         */
        class DeltaEncoder
        {
        public:
            DeltaEncoder() = default;

            /**
             * Find nearest base pattern via Hamming distance
             *
             * Searches base_patterns for the pattern with minimum
             * Hamming distance to target.
             *
             * @param target Pattern to encode
             * @param base_patterns Available base patterns (tier 2 patterns)
             * @param max_distance Maximum Hamming distance (default: 3)
             * @return Nearest base pattern, or nullopt if too far
             *
             * Time: O(B × n) where B = |base_patterns|, n = pattern length
             * Space: O(1)
             *
             * Note: For large base sets, consider using locality-sensitive hashing
             * or k-d trees to accelerate nearest neighbor search.
             */
            std::optional<CanonicalPattern> find_nearest_base(
                const CanonicalPattern &target,
                const std::vector<CanonicalPattern> &base_patterns,
                uint32_t max_distance = 3);

            /**
             * Compute Hamming distance between two patterns
             *
             * Hamming distance = number of positions where patterns differ
             *
             * @param p1 First pattern
             * @param p2 Second pattern
             * @return Number of differing positions (0 to 16)
             *
             * Time: O(n)
             * Space: O(1)
             *
             * Example:
             *   p1 = [+1, -1,  0, +1]
             *   p2 = [+1, -1, +1, +1]
             *   hamming_distance(p1, p2) = 1  (position 2 differs)
             */
            uint32_t hamming_distance(
                const TernaryPattern &p1,
                const TernaryPattern &p2) const;

            /**
             * Encode delta between target and base
             *
             * For each activation value x ∈ [-128, 127]:
             *   delta[x] = dot(target, x) - dot(base, x)
             *
             * This allows reconstruction:
             *   lookup(target, x) = lookup(base, x) + delta[x]
             *
             * @param target Target pattern to encode
             * @param base Base pattern (nearest neighbor)
             * @return Delta values for all 256 activation values
             *
             * Time: O(256 × n) = O(n) since 256 is constant
             * Space: O(256) = O(1)
             *
             * Storage: 256 × INT16 = 512 bytes (vs 1024 bytes for full table)
             */
            std::vector<int16_t> encode_delta(
                const CanonicalPattern &target,
                const CanonicalPattern &base);

            /**
             * Apply delta to base result
             *
             * Simple helper for lookup reconstruction:
             *   result = base_result + delta
             *
             * @param base_result Result from base pattern lookup
             * @param delta Delta value for this activation
             * @return Reconstructed result
             *
             * Time: O(1)
             */
            int32_t apply_delta(int32_t base_result, int16_t delta) const
            {
                return base_result + static_cast<int32_t>(delta);
            }

        private:
            /**
             * Compute dot product: pattern · activation
             *
             * Helper for delta encoding.
             *
             * @param pattern Ternary pattern
             * @param activation Single INT8 activation value
             * @return Dot product result
             */
            int32_t compute_dot_product(
                const TernaryPattern &pattern,
                int8_t activation) const;
        };

    } // namespace tmac
} // namespace ryzen_llm
