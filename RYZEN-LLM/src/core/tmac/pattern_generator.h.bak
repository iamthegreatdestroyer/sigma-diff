#pragma once

/**
 * @file pattern_generator.h
 * @brief Ternary pattern generation with symmetry exploitation
 *
 * Generates canonical forms of 3^16 ternary patterns, reducing storage
 * from 43M to ~21.5M via symmetry (w ~ -w equivalence).
 *
 * Part of T-MAC implementation for BitNet b1.58 inference.
 *
 * [REF:TMAC-001] - T-MAC Pattern Generation
 */

#include <array>
#include <cstdint>
#include <vector>
#include <unordered_set>
#include <optional>

namespace ryzen_llm
{
    namespace tmac
    {

        /**
         * Ternary pattern with 16 weights
         * Each element ∈ {-1, 0, +1}
         */
        using TernaryPattern = std::array<int8_t, 16>;

        /**
         * Canonical form of pattern with flip flag
         *
         * Canonical form ensures w ~ -w are stored only once.
         * If flip=true, the actual result should be negated.
         */
        struct CanonicalPattern
        {
            TernaryPattern pattern; ///< Canonical representation (more +1s than -1s)
            bool flip;              ///< If true, negate result when looking up

            /**
             * Compute FNV-1a hash for pattern indexing
             *
             * @return 64-bit hash value
             */
            uint64_t hash() const;

            /**
             * Equality comparison (for deduplication)
             */
            bool operator==(const CanonicalPattern &other) const;
        };

        /**
         * Pattern generator with symmetry exploitation
         *
         * Implements Theorem 1 from TMAC_COMPRESSION_ANALYSIS.md:
         *   For w' = -w: w' · x = -(w · x)
         *   Therefore: Store only one representative per {w, -w} equivalence class
         *
         * Reduction: 3^16 = 43,046,721 → ~21,523,360 canonical patterns (2× compression)
         */
        class PatternGenerator
        {
        public:
            PatternGenerator() = default;

            /**
             * Generate all canonical ternary patterns
             *
             * @param group_size Number of weights per group (default: 16)
             * @return Vector of canonical patterns (~21.5M unique entries)
             *
             * Time complexity: O(3^n) generation
             * Space complexity: O(3^n / 2) for canonical forms
             *
             * Note: This is a one-time operation during table building.
             * For inference, we only canonicalize individual patterns (O(n)).
             */
            std::vector<CanonicalPattern> generate_all_patterns(uint32_t group_size = 16);

            /**
             * Convert pattern to canonical form
             *
             * Algorithm:
             *   1. Count +1s and -1s in pattern
             *   2. If count(-1) > count(+1): flip all signs → canonical form
             *   3. If count(-1) == count(+1): Use first non-zero element as tiebreaker
             *   4. Return canonical form + flip flag
             *
             * @param pattern Input ternary pattern
             * @return Canonical form with flip flag
             *
             * Time: O(n) where n = pattern length
             * Space: O(1)
             *
             * Example:
             *   Input:  [-1, +1, 0, -1, +1, 0, ...]  (2 positives, 2 negatives)
             *   Output: [+1, -1, 0, +1, -1, 0, ...], flip=false
             */
            CanonicalPattern canonicalize(const TernaryPattern &pattern) const;

            /**
             * Check if pattern is self-symmetric (all zeros)
             *
             * The zero vector is the only pattern where w = -w.
             *
             * @param pattern Input pattern
             * @return true if pattern == -pattern (i.e., all zeros)
             */
            bool is_self_symmetric(const TernaryPattern &pattern) const;

            /**
             * Count non-zero elements in pattern
             *
             * Used for zero-compression optimization (RLE).
             * Patterns with many zeros can be stored more efficiently.
             *
             * @param pattern Input pattern
             * @return Number of non-zero elements (0 to 16)
             */
            uint32_t count_non_zeros(const TernaryPattern &pattern) const;

            /**
             * Get sparsity level (percentage of zeros)
             *
             * @param pattern Input pattern
             * @return Sparsity in range [0.0, 1.0]
             */
            float get_sparsity(const TernaryPattern &pattern) const;

        private:
            /**
             * Recursive pattern generation
             *
             * Generates all 3^n combinations via depth-first traversal.
             * Deduplicates via hash set to store only canonical forms.
             *
             * @param current Pattern being constructed
             * @param pos Current position (0 to group_size-1)
             * @param seen Hash set for deduplication
             * @param results Output vector of canonical patterns
             */
            void generate_recursive(
                TernaryPattern &current,
                uint32_t pos,
                std::unordered_set<uint64_t> &seen,
                std::vector<CanonicalPattern> &results);
        };

    } // namespace tmac
} // namespace ryzen_llm

/**
 * Hash function for CanonicalPattern (for unordered containers)
 */
namespace std
{
    template <>
    struct hash<ryzen_llm::tmac::CanonicalPattern>
    {
        size_t operator()(const ryzen_llm::tmac::CanonicalPattern &p) const
        {
            return p.hash();
        }
    };
}
