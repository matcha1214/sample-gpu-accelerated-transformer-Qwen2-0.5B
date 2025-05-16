#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <map>

/**
 * Byte-pair encoding, with special token handling.
 */
class BPE {
public:
    // map token ID number -> string
    std::vector<std::string> vocab;
    // map string -> token ID number
    std::map<std::string, uint32_t> inverse_vocab;
    // subset of inverse_vocab with special tokens only, such as "<|im_start|>" -> vocab["<|im_start|>"]
    std::map<std::string, uint32_t> special_tokens;
    // map utf8 byte -> token ID number (subset of inverse_vocab, but only single bytes)
    std::map<char, uint32_t> inverse_vocab_char;
    // (vocab["a"], vocab["b"]) -> vocab["ab"]
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> merges;

    explicit BPE(const std::string &model_dir);
    [[nodiscard]] std::vector<uint32_t> encode(const std::string &str) const;
    [[nodiscard]] std::string decode(uint32_t token) const;
    [[nodiscard]] std::string decode(const std::vector<uint32_t> &tokens) const;
};
