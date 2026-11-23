// Modified in the NInfer Inkling Ada fork.

#pragma once

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace q4_oracle {

struct Block {
    __half scale;
    std::array<std::uint8_t, 16> codes;
};
static_assert(sizeof(Block) == 18);

template <class Read>
Block encode(Read read) {
    float peak = 0.0f;
    for (int i = 0; i < 32; ++i) {
        if (std::abs(read(i)) > std::abs(peak)) { peak = read(i); }
    }
    const float scale = peak / -8.0f;
    const float inverse = scale != 0.0f ? 1.0f / scale : 0.0f;
    const auto code = [&](int i) { return std::clamp(int(read(i) * inverse + 8.5f), 0, 15); };
    Block block{__float2half(scale), {}};
    for (int i = 0; i < 16; ++i) {
        block.codes[i] = std::uint8_t(code(i) | (code(i + 16) << 4));
    }
    return block;
}

inline __half decode(const Block& block, int i) {
    const int pair = block.codes[i % 16];
    const int value = (i < 16 ? pair & 15 : pair >> 4) - 8;
    return __float2half(__half2float(block.scale) * float(value));
}

// Keep a dense, dequantized reference for the existing FP64 attention oracle.
inline std::vector<Block> encode_plane(std::vector<__half>& decoded) {
    std::vector<Block> encoded;
    encoded.reserve(decoded.size() / 32);
    for (std::size_t start = 0; start < decoded.size(); start += 32) {
        const auto block = encode([&](int i) { return __half2float(decoded[start + i]); });
        for (int i = 0; i < 32; ++i) { decoded[start + i] = decode(block, i); }
        encoded.push_back(block);
    }
    return encoded;
}

} // namespace q4_oracle
