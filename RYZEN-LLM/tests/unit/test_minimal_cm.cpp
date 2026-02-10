#include <iostream>
#include <vector>
#include <random>
#include <cmath>

int main()
{
    std::cout << "Test 1: Basic allocation\n";
    std::vector<float> v1(128);
    std::cout << "✓ Allocated vector of 128 floats\n";

    std::cout << "Test 2: Xavier calculation\n";
    float xavier_scale = std::sqrt(1.0f / 128.0f);
    std::cout << "✓ Xavier scale: " << xavier_scale << "\n";

    std::cout << "Test 3: RNG setup\n";
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-xavier_scale, xavier_scale);
    std::cout << "✓ RNG initialized\n";

    std::cout << "Test 4: RNG generation\n";
    for (int i = 0; i < 10; ++i)
    {
        float val = dist(rng);
        std::cout << "  " << i << ": " << val << "\n";
    }
    std::cout << "✓ RNG generated 10 values\n";

    std::cout << "Test 5: Large allocation (128x128)\n";
    uint32_t size = 128 * 128;
    std::cout << "  Size: " << size << " floats\n";
    std::vector<float> v2(size);
    std::cout << "✓ Allocated " << size << " floats\n";

    std::cout << "Test 6: Fill with RNG\n";
    for (uint32_t i = 0; i < size; ++i)
    {
        v2[i] = dist(rng);
        if (i % 2048 == 0 && i > 0)
        {
            std::cout << "  Filled " << i << " values...\n";
        }
    }
    std::cout << "✓ Filled all " << size << " values\n";

    std::cout << "\n✓ All tests passed!\n";
    return 0;
}
