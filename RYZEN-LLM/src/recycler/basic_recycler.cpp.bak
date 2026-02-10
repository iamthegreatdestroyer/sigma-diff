#include "basic_recycler.h"
#include <chrono>
#include <functional>
#include <cstdio>

namespace ryzen_llm
{
    namespace recycler
    {

        BasicTokenRecycler::BasicTokenRecycler(size_t max_cache_entries)
            : max_entries_(max_cache_entries) {}

        ProcessedContext BasicTokenRecycler::process_input(
            const std::vector<int32_t> &tokens,
            const std::string &conversation_id)
        {
            (void)conversation_id; // Not used in basic implementation

            stats_.inputs_processed++;
            stats_.total_tokens_input += tokens.size();

            ProcessedContext result;
            result.original_token_count = tokens.size();

            // Fast path for tiny inputs
            if (tokens.size() < 10)
            {
                stats_.fast_path_bypasses++;
                result.tokens = tokens;
                result.processing_mode = ProcessingMode::FAST_PATH;
                result.effective_tokens = tokens.size();
                result.compression_ratio = 1.0f;
                stats_.total_tokens_output += tokens.size();
                return result;
            }

            // Check cache for exact match
            uint64_t hash = compute_hash(tokens);
            auto it = cache_.find(hash);

            if (it != cache_.end())
            {
                // Exact hit
                stats_.exact_hits++;
                it->second.access_count++;

                result.tokens = tokens; // Still return tokens (basic impl)
                result.rsu_reference = it->second.rsu_id;
                result.processing_mode = ProcessingMode::EXACT_HIT;
                result.effective_tokens = tokens.size();
                result.compression_ratio = 1.0f; // No actual compression in basic impl
                result.recycled_kv_sequence_id = it->second.kv_sequence_id;

                if (result.recycled_kv_sequence_id)
                {
                    stats_.total_kv_cache_hits++;
                }
            }
            else
            {
                // Cache miss - store new entry
                stats_.fresh_encodes++;
                evict_if_needed();

                std::string rsu_id = generate_rsu_id(hash);
                cache_[hash] = CacheEntry{tokens, rsu_id, std::nullopt, 1};
                lru_order_.push_back(hash);
                stats_.rsus_created++;

                result.tokens = tokens;
                result.rsu_reference = rsu_id;
                result.processing_mode = ProcessingMode::FRESH_ENCODE;
                result.effective_tokens = tokens.size();
                result.compression_ratio = 1.0f;
            }

            stats_.total_tokens_output += result.effective_tokens;
            return result;
        }

        InjectionResult BasicTokenRecycler::prepare_context_injection(
            const std::vector<int32_t> &base_tokens,
            const std::vector<std::string> &rsu_references,
            size_t max_tokens)
        {
            (void)rsu_references; // Not used in basic implementation

            InjectionResult result;
            result.tokens = base_tokens;

            // Truncate if needed
            if (result.tokens.size() > max_tokens)
            {
                size_t overflow = result.tokens.size() - max_tokens;
                result.tokens.erase(result.tokens.begin(), result.tokens.begin() + overflow);
            }

            result.fresh_token_count = result.tokens.size();
            return result;
        }

        void BasicTokenRecycler::register_kv_cache(
            const std::string &rsu_reference,
            size_t kv_sequence_id,
            const std::vector<int32_t> &anchor_positions)
        {
            (void)anchor_positions; // Not used in basic implementation

            // Find cache entry by RSU reference and link KV
            for (auto &[hash, entry] : cache_)
            {
                if (entry.rsu_id == rsu_reference)
                {
                    entry.kv_sequence_id = kv_sequence_id;
                    break;
                }
            }
        }

        RecyclerStatistics BasicTokenRecycler::get_statistics() const
        {
            return stats_;
        }

        void BasicTokenRecycler::reset_statistics()
        {
            stats_ = RecyclerStatistics{};
        }

        uint64_t BasicTokenRecycler::compute_hash(const std::vector<int32_t> &tokens) const
        {
            // Simple FNV-1a hash
            uint64_t hash = 14695981039346656037ULL;
            for (int32_t token : tokens)
            {
                hash ^= static_cast<uint64_t>(token);
                hash *= 1099511628211ULL;
            }
            return hash;
        }

        std::string BasicTokenRecycler::generate_rsu_id(uint64_t hash) const
        {
            auto now = std::chrono::system_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now.time_since_epoch())
                          .count();

            char buffer[64];
            snprintf(buffer, sizeof(buffer), "rsu_%08llx_%lld",
                     static_cast<unsigned long long>(hash & 0xFFFFFFFF),
                     static_cast<long long>(ms));
            return std::string(buffer);
        }

        void BasicTokenRecycler::evict_if_needed()
        {
            while (cache_.size() >= max_entries_ && !lru_order_.empty())
            {
                uint64_t oldest = lru_order_.front();
                lru_order_.pop_front();
                cache_.erase(oldest);
            }
        }

        // Factory implementation
        std::unique_ptr<ITokenRecycler> TokenRecyclerFactory::create(
            bool use_sigma,
            const std::string &config_path)
        {
            if (use_sigma)
            {
                // FUTURE: Return SigmaRSUEngine when available
                // return std::make_unique<SigmaRSUEngine>(config_path);

                // For now, fall back to basic implementation with warning
                // TODO(SIGMA): Implement SigmaRSUEngine integration
                (void)config_path;
                return std::make_unique<BasicTokenRecycler>(100);
            }

            return std::make_unique<BasicTokenRecycler>(100);
        }

    } // namespace recycler
} // namespace ryzen_llm
