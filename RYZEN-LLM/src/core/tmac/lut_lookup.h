#pragma once

/**
 * @file lut_lookup.h
 * @brief Runtime lookup engine for compressed T-MAC tables
 *
 * Provides O(1) average lookup with 95% cache hit rate through
 * multi-tier search strategy:
 *   - Tier 1: 60% hit, ~2 cycles
 *   - Tier 2: 35% hit, ~75 cycles
 *   - Tier 3: 4.9% hit, ~250 cycles
 *   - Tier 4: 0.1% hit, ~100 cycles (fallback)
 *
 * Expected latency: ~40 cycles = 10 ns @ 4GHz
 *
 * Thread Safety:
 *   - Lookups are thread-safe (read-only on LUT data)
 *   - Stats use atomic counters (no false sharing)
 *   - Padding ensures each counter is on its own cache line
 *
 * [REF:TMAC-005] - Runtime Lookup
 */

#include "table_builder.h"
#include <cstdint>
#include <memory>
#include <string>
#include <atomic>

namespace ryzanstein_llm
{
    namespace tmac
    {

        /**
         * Runtime lookup engine for compressed T-MAC tables
         *
         * Thread-safe for read-only operations after initialization.
         */
        class LUTLookup
        {
        public:
            /**
             * Initialize from pre-built CompressedLUT structure
             *
             * @param lut Compressed LUT structure (moves ownership)
             */
            explicit LUTLookup(std::shared_ptr<CompressedLUT> lut);

            /**
             * Lookup result for pattern × activation
             *
             * @param pattern Ternary weight pattern (16 elements)
             * @param activation Single INT8 activation value
             * @return Dot product result (INT32)
             *
             * Time: O(1) with 95% probability, O(16) worst case
             *
             * Algorithm:
             *   1. Canonicalize pattern → O(16)
             *   2. Tier 1 lookup (hot) → O(1), 60% hit
             *   3. Tier 2 lookup (warm) → O(1), 35% hit
             *   4. Tier 3 lookup (delta) → O(1), 4.9% hit
             *   5. Fallback computation → O(16), 0.1% hit
             */
            int32_t lookup(
                const TernaryPattern &pattern,
                int8_t activation);

            /**
             * Batch lookup for multiple activation values
             *
             * Optimized for sequential patterns with prefetching.
             * Reuses canonicalization across all activations.
             *
             * @param pattern Ternary weight pattern (shared across batch)
             * @param activations Array of activation values
             * @param results Output array (preallocated, same size as activations)
             * @param count Number of activations to process
             *
             * Time: O(count) amortized
             */
            void lookup_batch(
                const TernaryPattern &pattern,
                const int8_t *activations,
                int32_t *results,
                uint32_t count);

            /**
             * Lookup statistics for performance monitoring
             *
             * Thread Safety:
             *   - Uses atomic counters for lock-free thread-safe updates
             *   - Each counter padded to 64-byte cache line to prevent false sharing
             *   - Relaxed memory ordering for maximum performance
             */
            struct alignas(64) Stats
            {
            private:
                // Cache-line padded atomic counters to prevent false sharing
                // Each counter gets its own 64-byte cache line
                struct alignas(64) AtomicCounter
                {
                    std::atomic<uint64_t> value{0};
                    char padding[64 - sizeof(std::atomic<uint64_t>)]; // Pad to cache line
                };

                mutable AtomicCounter tier1_hits_;     ///< Hot cache hits
                mutable AtomicCounter tier2_hits_;     ///< Warm cache hits
                mutable AtomicCounter tier3_hits_;     ///< Delta reconstruction hits
                mutable AtomicCounter fallback_count_; ///< On-the-fly computations

            public:
                Stats() = default;

                // Non-copyable due to atomics, but movable
                Stats(const Stats &other)
                {
                    tier1_hits_.value.store(other.tier1_hits_.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
                    tier2_hits_.value.store(other.tier2_hits_.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
                    tier3_hits_.value.store(other.tier3_hits_.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
                    fallback_count_.value.store(other.fallback_count_.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
                }

                Stats &operator=(const Stats &other)
                {
                    if (this != &other)
                    {
                        tier1_hits_.value.store(other.tier1_hits_.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
                        tier2_hits_.value.store(other.tier2_hits_.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
                        tier3_hits_.value.store(other.tier3_hits_.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
                        fallback_count_.value.store(other.fallback_count_.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
                    }
                    return *this;
                }

                // Thread-safe increment methods (relaxed ordering for perf)
                void increment_tier1() const { tier1_hits_.value.fetch_add(1, std::memory_order_relaxed); }
                void increment_tier2() const { tier2_hits_.value.fetch_add(1, std::memory_order_relaxed); }
                void increment_tier3() const { tier3_hits_.value.fetch_add(1, std::memory_order_relaxed); }
                void increment_fallback() const { fallback_count_.value.fetch_add(1, std::memory_order_relaxed); }

                // Batch increment for reduced atomic overhead in tight loops
                void add_tier1(uint64_t count) const { tier1_hits_.value.fetch_add(count, std::memory_order_relaxed); }
                void add_tier2(uint64_t count) const { tier2_hits_.value.fetch_add(count, std::memory_order_relaxed); }
                void add_tier3(uint64_t count) const { tier3_hits_.value.fetch_add(count, std::memory_order_relaxed); }
                void add_fallback(uint64_t count) const { fallback_count_.value.fetch_add(count, std::memory_order_relaxed); }

                // Accessors (for reading statistics)
                uint64_t tier1_hits() const { return tier1_hits_.value.load(std::memory_order_relaxed); }
                uint64_t tier2_hits() const { return tier2_hits_.value.load(std::memory_order_relaxed); }
                uint64_t tier3_hits() const { return tier3_hits_.value.load(std::memory_order_relaxed); }
                uint64_t fallback_count() const { return fallback_count_.value.load(std::memory_order_relaxed); }

                // Reset all counters
                void reset() const
                {
                    tier1_hits_.value.store(0, std::memory_order_relaxed);
                    tier2_hits_.value.store(0, std::memory_order_relaxed);
                    tier3_hits_.value.store(0, std::memory_order_relaxed);
                    fallback_count_.value.store(0, std::memory_order_relaxed);
                }

                /**
                 * Overall hit rate (tier 1-3)
                 */
                double hit_rate() const
                {
                    uint64_t t1 = tier1_hits(), t2 = tier2_hits(), t3 = tier3_hits(), fb = fallback_count();
                    uint64_t total = t1 + t2 + t3 + fb;
                    return total > 0
                               ? static_cast<double>(t1 + t2 + t3) / total
                               : 0.0;
                }

                /**
                 * Tier-specific hit rates
                 */
                double tier1_rate() const
                {
                    uint64_t t1 = tier1_hits(), t2 = tier2_hits(), t3 = tier3_hits(), fb = fallback_count();
                    uint64_t total = t1 + t2 + t3 + fb;
                    return total > 0 ? static_cast<double>(t1) / total : 0.0;
                }

                double tier2_rate() const
                {
                    uint64_t t1 = tier1_hits(), t2 = tier2_hits(), t3 = tier3_hits(), fb = fallback_count();
                    uint64_t total = t1 + t2 + t3 + fb;
                    return total > 0 ? static_cast<double>(t2) / total : 0.0;
                }

                double tier3_rate() const
                {
                    uint64_t t1 = tier1_hits(), t2 = tier2_hits(), t3 = tier3_hits(), fb = fallback_count();
                    uint64_t total = t1 + t2 + t3 + fb;
                    return total > 0 ? static_cast<double>(t3) / total : 0.0;
                }

                double fallback_rate() const
                {
                    uint64_t t1 = tier1_hits(), t2 = tier2_hits(), t3 = tier3_hits(), fb = fallback_count();
                    uint64_t total = t1 + t2 + t3 + fb;
                    return total > 0 ? static_cast<double>(fb) / total : 0.0;
                }

                /**
                 * Total number of lookups
                 */
                uint64_t total_lookups() const
                {
                    return tier1_hits() + tier2_hits() + tier3_hits() + fallback_count();
                }
            };

            /**
             * Get accumulated statistics
             */
            const Stats &get_stats() const { return stats_; }

            /**
             * Reset statistics counters (thread-safe, uses atomic stores)
             */
            void reset_stats() { stats_.reset(); }

            /**
             * Print performance statistics
             */
            void print_stats() const;

        private:
            std::shared_ptr<CompressedLUT> lut_;
            PatternGenerator pattern_gen_;
            mutable Stats stats_; // mutable for thread-local stats in future

            /**
             * Fallback: compute dot product directly
             *
             * Used when pattern is not in any tier.
             *
             * @param pattern Ternary pattern
             * @param activation Activation value
             * @return Computed result
             *
             * Time: O(16)
             */
            int32_t compute_fallback(
                const TernaryPattern &pattern,
                int8_t activation);

            /**
             * Convert activation to tier 1 index
             *
             * Maps [-32, 31] → [0, 63]
             *
             * @param activation Activation value
             * @return Index, or -1 if out of tier 1 range
             */
            int32_t get_tier1_index(int8_t activation) const
            {
                if (activation < lut_->tier1_act_min || activation > lut_->tier1_act_max)
                {
                    return -1;
                }
                return activation - lut_->tier1_act_min;
            }

            /**
             * Convert activation to tier 2 index
             *
             * Maps [-128, 127] → [0, 255]
             *
             * @param activation Activation value
             * @return Index [0, 255]
             */
            uint8_t get_tier2_index(int8_t activation) const
            {
                return static_cast<uint8_t>(activation + 128);
            }
        };

    } // namespace tmac
} // namespace ryzanstein_llm
