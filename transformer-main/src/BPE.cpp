#include "BPE.h"

#include "vendor/json.hpp"
#include <fstream>
#include <set>


std::string fix_encoding(const std::string &str) {
    // inverse of https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L9
    std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> converter;
    std::vector<uint8_t> bytes;
    for (char16_t c : converter.from_bytes(str)) {
        if (c < 256) {
            bytes.push_back(c);
        } else if (c < 289) {
            bytes.push_back(c - 256);
        } else if (c < 323) {
            bytes.push_back(c - 162);
        } else if (c < 324) {
            bytes.push_back(c - 150);
        } else {
            throw std::invalid_argument("Invalid character encoding");
        }
    }
    return {bytes.begin(), bytes.end()};
}

BPE::BPE(const std::string &model_dir) {
    std::ifstream file(model_dir + "/tokenizer.json");
    nlohmann::json conf = nlohmann::json::parse(file);
    this->vocab.resize(conf["model"]["vocab"].size() + conf["added_tokens"].size());
    for (auto &[str, idx] : conf["model"]["vocab"].items()) {
        this->vocab[idx] = fix_encoding(str);
    }
    for (auto &token_info : conf["added_tokens"]) {
        uint32_t token_id = token_info["id"];
        std::string content = token_info["content"];
        this->vocab[token_id] = content;
        this->special_tokens[content] = token_id;
    }
    for (size_t i = 0; i < this->vocab.size(); i++) {
        auto &str = this->vocab[i];
        this->inverse_vocab[str] = i;
        if (str.size() == 1) {
            this->inverse_vocab_char[str[0]] = i;
        }
    }
    for (auto &merge: conf["model"]["merges"]) {
        // merge: "a b", convert to numeric IDs
        std::string merge_str = merge;
        size_t space = merge_str.find(' ');
        std::string a = fix_encoding(merge_str.substr(0, space));
        std::string b = fix_encoding(merge_str.substr(space + 1));
        auto pair = std::make_pair(this->inverse_vocab.at(a), this->inverse_vocab.at(b));
        this->merges[pair] = this->inverse_vocab[a + b];
    }
}

std::vector<uint32_t> BPE::encode(const std::string &str) const {
    // based on https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/minbpe/basic.py#L57
    std::vector<uint32_t> tokens;
    for (size_t i = 0; i < str.size();) {
        // replace special tokens immediately
        bool special_token_found = false;
        for (auto &[special_token_str, special_token_id] : this->special_tokens) {
            if (str.substr(i, special_token_str.size()) == special_token_str) {
                tokens.push_back(special_token_id);
                i += special_token_str.size();
                special_token_found = true;
                break;
            }
        }
        if (!special_token_found) {
            // initially, map each utf8 byte to one token
            tokens.push_back(this->inverse_vocab_char.at(str[i]));
            i++;
        }
    }
    while (tokens.size() >= 2) {
        // find pair with the lowest merge idx
        std::set<std::pair<uint32_t, uint32_t>> pairs_seen;
        for (size_t i = 0; i < tokens.size() - 1; i++) {
            auto pair = std::make_pair(tokens[i], tokens[i + 1]);
            pairs_seen.insert(pair);
        }
        uint32_t best_pair_merge_idx = UINT32_MAX;
        auto best_pair = std::make_pair(0U, 0U);
        for (auto pair : pairs_seen) {
            auto it = this->merges.find(pair);
            if (it != this->merges.end()) {
                uint32_t merge_idx = it->second;
                // merge lower idx pairs first
                if (merge_idx < best_pair_merge_idx) {
                    best_pair_merge_idx = merge_idx;
                    best_pair = pair;
                }
            }
        }
        if (best_pair_merge_idx == UINT32_MAX) {
            // no merges found
            break;
        } else {
            // replace best_pair with the merged token
            // https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/minbpe/base.py#L25
            std::vector<uint32_t> new_tokens;
            size_t i = 0;
            while (i < tokens.size()) {
                if (tokens[i] == best_pair.first && i < tokens.size() - 1 && tokens[i + 1] == best_pair.second) {
                    new_tokens.push_back(best_pair_merge_idx);
                    i += 2;
                } else {
                    new_tokens.push_back(tokens[i]);
                    i++;
                }
            }
            tokens = std::move(new_tokens);
        }
    }
    return tokens;
}

std::string BPE::decode(uint32_t token) const {
    return this->vocab.at(token);
}

std::string BPE::decode(const std::vector<uint32_t> &tokens) const {
    std::stringstream ss;
    for (uint32_t token : tokens) {
        ss << this->decode(token);
    }
    return ss.str();
}
