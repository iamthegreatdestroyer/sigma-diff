/**
 * @file frequency_analyzer.cpp
 * @brief Implementation of pattern frequency analysis
 *
 * [REF:TMAC-002] - Pattern Frequency Analysis
 */

#include "frequency_analyzer.h"
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace ryzen_llm
{
    namespace tmac
    {

        std::vector<PatternFrequency> FrequencyAnalyzer::analyze_weights(
            const std::vector<int8_t> &weights,
            uint32_t M,
            uint32_t K,
            uint32_t group_size)
        {
            if (weights.size() != static_cast<size_t>(M) * K)
            {
                throw std::invalid_argument("Weight size mismatch: expected M×K elements");
            }

            if (K % group_size != 0)
            {
                throw std::invalid_argument("K must be divisible by group_size");
            }

            // Map: pattern hash → count
            std::unordered_map<uint64_t, PatternFrequency> freq_map;

            uint64_t total_patterns = 0;

            // Iterate over all rows (output features)
            for (uint32_t m = 0; m < M; ++m)
            {
                // Iterate over groups in this row
                uint32_t num_groups = K / group_size;

                for (uint32_t g = 0; g < num_groups; ++g)
                {
                    // Extract pattern for this group
                    TernaryPattern pattern;

                    for (uint32_t i = 0; i < group_size; ++i)
                    {
                        size_t idx = m * K + g * group_size + i;
                        pattern[i] = weights[idx];
                    }

                    // Canonicalize pattern
                    auto canonical = pattern_gen_.canonicalize(pattern);
                    uint64_t h = canonical.hash();

                    // Update frequency
                    auto it = freq_map.find(h);
                    if (it != freq_map.end())
                    {
                        it->second.count++;
                    }
                    else
                    {
                        freq_map[h] = PatternFrequency{canonical, 1, 0.0};
                    }

                    total_patterns++;
                }
            }

            // Convert map to sorted vector
            std::vector<PatternFrequency> frequencies;
            frequencies.reserve(freq_map.size());

            for (auto &[hash, freq] : freq_map)
            {
                // Compute probability
                freq.probability = static_cast<double>(freq.count) / total_patterns;
                frequencies.push_back(freq);
            }

            // Sort by frequency (descending)
            std::sort(frequencies.begin(), frequencies.end());

            return frequencies;
        }

        std::vector<CanonicalPattern> FrequencyAnalyzer::get_top_k(
            const std::vector<PatternFrequency> &frequencies,
            uint32_t k)
        {
            std::vector<CanonicalPattern> top_patterns;
            top_patterns.reserve(std::min(k, static_cast<uint32_t>(frequencies.size())));

            for (uint32_t i = 0; i < k && i < frequencies.size(); ++i)
            {
                top_patterns.push_back(frequencies[i].pattern);
            }

            return top_patterns;
        }

        double FrequencyAnalyzer::compute_coverage(
            const std::vector<PatternFrequency> &frequencies,
            uint32_t k) const
        {
            double cumulative = 0.0;

            for (uint32_t i = 0; i < k && i < frequencies.size(); ++i)
            {
                cumulative += frequencies[i].probability;
            }

            return cumulative;
        }

        void FrequencyAnalyzer::print_summary(
            const std::vector<PatternFrequency> &frequencies,
            uint32_t top_n) const
        {
            std::cout << "Pattern Frequency Analysis Summary\n";
            std::cout << "===================================\n";
            std::cout << "Total unique patterns: " << frequencies.size() << "\n";

            if (frequencies.empty())
            {
                std::cout << "No patterns found.\n";
                return;
            }

            // Compute total count
            uint64_t total_count = 0;
            for (const auto &freq : frequencies)
            {
                total_count += freq.count;
            }
            std::cout << "Total pattern instances: " << total_count << "\n\n";

            // Print top N patterns
            std::cout << "Top " << std::min(top_n, static_cast<uint32_t>(frequencies.size()))
                      << " Most Frequent Patterns:\n";
            std::cout << std::fixed << std::setprecision(4);

            for (uint32_t i = 0; i < top_n && i < frequencies.size(); ++i)
            {
                const auto &freq = frequencies[i];

                std::cout << "  #" << (i + 1) << ": ";
                std::cout << "count=" << freq.count << " ";
                std::cout << "prob=" << (freq.probability * 100.0) << "% ";

                // Print pattern
                std::cout << "pattern=[";
                for (size_t j = 0; j < freq.pattern.pattern.size(); ++j)
                {
                    if (j > 0)
                        std::cout << ",";
                    std::cout << static_cast<int>(freq.pattern.pattern[j]);
                }
                std::cout << "] flip=" << (freq.pattern.flip ? "Y" : "N") << "\n";
            }

            // Print coverage statistics
            std::cout << "\nCumulative Coverage:\n";
            for (uint32_t k : {100, 1000, 10000, 100000})
            {
                if (k <= frequencies.size())
                {
                    double coverage = compute_coverage(frequencies, k);
                    std::cout << "  Top " << k << " patterns: "
                              << (coverage * 100.0) << "%\n";
                }
            }
        }

    } // namespace tmac
} // namespace ryzen_llm
