#include "../BPE.h"
#include "../qwen2/Qwen2Loader.h"
#include <iostream>

const char *TEST_STR1 = "Hello Hola Bonjour Hallo Ciao Olá Привет 你好 こんにちは 안녕하세요 नमस्ते السلام عليكم שלום Hej Hei Halló Merhaba Szia Sawubona Salam";
// reference values generated with huggingface AutoTokenizer
const std::vector<uint32_t> TEST_STR1_REFERENCE{9707, 472, 7924, 13481, 29262, 19851, 385, 356, 22516, 11959, 1953, 79484, 26991, 8178, 220, 108386, 220, 89015, 95170, 144370, 91145, 14925, 101, 87244, 78368, 30484, 97, 34370, 130781, 124794, 124613, 124756, 123881, 1260, 73, 1260, 72, 10926, 1794, 8755, 10573, 64, 44190, 685, 37007, 392, 6721, 8211, 309};

const char *TEST_STR2 = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2<|im_end|>\n<|im_start|>assistant\n2+2=4<|im_end|>\n";
const std::vector<uint32_t> TEST_STR2_REFERENCE{151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838, 374, 220, 17, 10, 17, 151645, 198, 151644, 77091, 198, 17, 10, 17, 28, 19, 151645, 198};

bool check_tokens(const std::vector<uint32_t> &encoded, const std::vector<uint32_t> &reference) {
    for (size_t i = 0; i < std::min(encoded.size(), reference.size()); i++) {
        if (encoded[i] != reference[i]) {
            std::cerr << "wrong token at position " << i << ", got " << encoded[i] << ", expected " << reference[i] << std::endl;
            return false;
        }
    }
    if (encoded.size() != reference.size()) {
        std::cerr << "wrong number of tokens, got " << encoded.size() << ", expected " << reference.size() << std::endl;
        return false;
    }
    return true;
}

void print_tokens(const BPE &bpe, const std::vector<uint32_t> &encoded) {
    for (size_t i = 0; i < encoded.size(); i++) {
        if (i > 0) {
            std::cerr << ", ";
        }
        std::cerr << encoded[i] << "(" << bpe.vocab[encoded[i]] << ")";
    }
    std::cerr << std::endl;
}

void check_and_print(const BPE &bpe, const char *str, const std::vector<uint32_t> &reference) {
    auto encoded = bpe.encode(str);
    if (!check_tokens(encoded, reference)) {
        std::cerr << "encoded:" << std::endl;
        print_tokens(bpe, encoded);
        std::cerr << "reference:" << std::endl;
        print_tokens(bpe, reference);
        std::exit(1);
    }
}

int main() {
    BPE bpe(Qwen2Loader::get_model_dir());

    check_and_print(bpe, TEST_STR1, TEST_STR1_REFERENCE);
    check_and_print(bpe, TEST_STR2, TEST_STR2_REFERENCE);
}