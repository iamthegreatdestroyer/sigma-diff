/**
 * @file pattern_generator.cpp
 * @brief Implementation of ternary pattern generation with symmetry
 *
 * [REF:TMAC-001] - T-MAC Pattern Generation
 */

#include "pattern_generator.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ryzanstein_llm
{
    namespace tmac
    {

        uint64_t CanonicalPattern::hash() const
        {
            // FNV-1a hash for pattern indexing
            // Maps {-1, 0, +1} → {0, 1, 2} for stable hashing
            uint64_t h = 14695981039346656037ULL; // FNV offset basis

            for (int8_t w : pattern)
            {
                uint8_t value = static_cast<uint8_t>(w + 1); // Map to 0,1,2
                h ^= value;
                h *= 1099511628211ULL; // FNV prime
            }

            // Mix in flip flag for complete representation
            h ^= static_cast<uint64_t>(flip);

            return h;
        }

        bool CanonicalPattern::operator==(const CanonicalPattern &other) const
        {
            return flip == other.flip && pattern == other.pattern;
        }

        std::vector<CanonicalPattern> PatternGenerator::generate_all_patterns(uint32_t group_size)
        {
            if (group_size != 16)
            {
                // Currently only support group_size=16
                // Could be extended for different sizes
                throw std::invalid_argument("Only group_size=16 is currently supported");
            }

            std::vector<CanonicalPattern> results;
            results.reserve(21523361); // Expected: ~(3^16 / 2) + 1

            std::unordered_set<uint64_t> seen;
            seen.reserve(21523361);

            TernaryPattern current;
            current.fill(0);

            // Generate all patterns recursively
            generate_recursive(current, 0, seen, results);

            return results;
        }

        void PatternGenerator::generate_recursive(
            TernaryPattern &current,
            uint32_t pos,
            std::unordered_set<uint64_t> &seen,
            std::vector<CanonicalPattern> &results)
        {
            if (pos == 16)
            {
                // Base case: complete pattern generated
                auto canonical = canonicalize(current);
                uint64_t h = canonical.hash();

                // Only store if not already seen (deduplication)
                if (seen.find(h) == seen.end())
                {
                    seen.insert(h);
                    results.push_back(canonical);
                }
                return;
            }

            // Recursive case: try all three values at this position
            for (int8_t val : {-1, 0, 1})
            {
                current[pos] = val;
                generate_recursive(current, pos + 1, seen, results);
            }
        }

        CanonicalPattern PatternGenerator::canonicalize(const TernaryPattern &pattern) const
        {
            // Count +1s and -1s
            int count_pos = 0;
            int count_neg = 0;

            for (int8_t w : pattern)
            {
                if (w == 1)
                {
                    ++count_pos;
                }
                else if (w == -1)
                {
                    ++count_neg;
                }
            }

            // Determine if we should flip
            bool should_flip = false;

            if (count_neg > count_pos)
            {
                // More negatives than positives → flip
                should_flip = true;
            }
            else if (count_neg == count_pos && count_neg > 0)
            {
                // Tie-break: use first non-zero element
                // Canonical form has +1 as first non-zero
                for (int8_t w : pattern)
                {
                    if (w != 0)
                    {
                        should_flip = (w == -1);
                        break;
                    }
                }
            }
            // else: more positives (or all zeros) → don't flip

            CanonicalPattern result;
            result.flip = should_flip;

            if (should_flip)
            {
                // Flip all signs
                for (size_t i = 0; i < 16; ++i)
                {
                    result.pattern[i] = -pattern[i];
                }
            }
            else
            {
                // Keep as-is
                result.pattern = pattern;
            }

            return result;
        }

        bool PatternGenerator::is_self_symmetric(const TernaryPattern &pattern) const
        {
            // Check if all elements are zero
            // Zero vector is the only self-symmetric pattern (w = -w)
            for (int8_t w : pattern)
            {
                if (w != 0)
                {
                    return false;
                }
            }
            return true;
        }

        uint32_t PatternGenerator::count_non_zeros(const TernaryPattern &pattern) const
        {
            uint32_t count = 0;
            for (int8_t w : pattern)
            {
                if (w != 0)
                {
                    ++count;
                }
            }
            return count;
        }

        float PatternGenerator::get_sparsity(const TernaryPattern &pattern) const
        {
            uint32_t zeros = 16 - count_non_zeros(pattern);
            return static_cast<float>(zeros) / 16.0f;
        }

    } // namespace tmac
} // namespace ryzanstein_llm
