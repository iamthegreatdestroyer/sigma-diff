#pragma once

/**
 * @file frequency_analyzer.h
 * @brief Analyze pattern frequency in BitNet weights for tier assignment
 *
 * Determines which patterns should go into hot/warm/cold tiers based
 * on their occurrence frequency in real model weights.
 *
 * [REF:TMAC-002] - Pattern Frequency Analysis
 */

#include "pattern_generator.h"
#include <unordered_map>
#include <vector>

namespace ryzen_llm
{
    namespace tmac
    {

        /**
         * Pattern frequency statistics from training weights
         */
        struct PatternFrequency
        {
            CanonicalPattern pattern; ///< The pattern
            uint64_t count;           ///< Number of occurrences in weights
            double probability;       ///< Normalized probability (count / total)

            /**
             * For sorting by frequency (descending)
             */
            bool operator<(const PatternFrequency &other) const
            {
                return count > other.count; // Descending order
            }
        };

        /**
         * Analyzes pattern frequency in real BitNet weights
         *
         * Used to determine hot/warm/cold tier assignments:
         * - Tier 1 (hot):  Top 10K patterns (~60% coverage)
         * - Tier 2 (warm): Top 100K patterns (~95% coverage)
         * - Tier 3 (cold): Remaining patterns (~5% coverage)
         *
         * This follows the 80-20 rule observed in real networks:
         * A small fraction of patterns account for most occurrences.
         */
        class FrequencyAnalyzer
        {
        public:
            FrequencyAnalyzer() = default;

            /**
             * Analyze pattern distribution in weight tensor
             *
             * Extracts all group_size patterns from flattened weights
             * and counts their occurrences.
             *
             * @param weights Ternary weight tensor (flattened [M × K])
             * @param M Number of output features
             * @param K Number of input features
             * @param group_size Pattern group size (default: 16)
             * @return Sorted patterns by frequency (descending)
             *
             * Time: O(M × K / group_size × log(unique_patterns))
             * Space: O(unique_patterns)
             *
             * Example:
             *   For M=4096, K=4096, group_size=16:
             *   Total patterns = 4096 × 4096 / 16 = 1,048,576
             *   Unique patterns ≈ 100K (due to parameter sharing)
             */
            std::vector<PatternFrequency> analyze_weights(
                const std::vector<int8_t> &weights,
                uint32_t M,
                uint32_t K,
                uint32_t group_size = 16);

            /**
             * Extract top-k most frequent patterns
             *
             * @param frequencies Full frequency distribution (must be sorted)
             * @param k Number of patterns to extract
             * @return Top k patterns (for tier 1 or tier 2)
             *
             * Time: O(k)
             * Space: O(k)
             */
            std::vector<CanonicalPattern> get_top_k(
                const std::vector<PatternFrequency> &frequencies,
                uint32_t k);

            /**
             * Compute cumulative frequency coverage
             *
             * Answers: "What % of occurrences do the top k patterns account for?"
             *
             * @param frequencies Frequency distribution (must be sorted)
             * @param k Number of top patterns
             * @return Cumulative probability [0.0, 1.0]
             *
             * Example:
             *   Top 10K patterns → ~0.60 (60% coverage)
             *   Top 100K patterns → ~0.95 (95% coverage)
             */
            double compute_coverage(
                const std::vector<PatternFrequency> &frequencies,
                uint32_t k) const;

            /**
             * Print frequency distribution summary
             *
             * Useful for debugging and tuning tier sizes.
             *
             * @param frequencies Frequency distribution
             * @param top_n Number of top patterns to display
             */
            void print_summary(
                const std::vector<PatternFrequency> &frequencies,
                uint32_t top_n = 20) const;

        private:
            PatternGenerator pattern_gen_;
        };

    } // namespace tmac
} // namespace ryzen_llm
