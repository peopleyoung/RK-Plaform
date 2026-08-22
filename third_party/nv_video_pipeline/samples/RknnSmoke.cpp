#include <rknn_api.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
std::string shape(const rknn_tensor_attr& attr) {
    std::string result = "[";
    for (uint32_t index = 0; index < attr.n_dims; ++index) {
        if (index != 0)
            result += ",";
        result += std::to_string(attr.dims[index]);
    }
    return result + "]";
}
}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "usage: rknn_smoke MODEL.rknn [MODEL.rknn ...]\n";
        return 2;
    }
    for (int arg = 1; arg < argc; ++arg) {
        std::ifstream input(argv[arg], std::ios::binary | std::ios::ate);
        if (!input) {
            std::cerr << "missing model: " << argv[arg] << '\n';
            return 3;
        }
        const auto end = input.tellg();
        if (end <= 0) {
            std::cerr << "empty model: " << argv[arg] << '\n';
            return 3;
        }
        const auto           size = static_cast<size_t>(end);
        std::vector<uint8_t> model(size);
        input.seekg(0);
        if (!input.read(reinterpret_cast<char*>(model.data()), static_cast<std::streamsize>(size))) {
            std::cerr << "failed to read model: " << argv[arg] << '\n';
            return 3;
        }
        rknn_context context = 0;
        int          status  = rknn_init(&context, model.data(), static_cast<uint32_t>(model.size()), 0, nullptr);
        if (status != RKNN_SUCC) {
            std::cerr << argv[arg] << ": rknn_init failed: " << status << '\n';
            return 4;
        }
        rknn_input_output_num io{};
        status = rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
        if (status != RKNN_SUCC) {
            std::cerr << argv[arg] << ": IO query failed: " << status << '\n';
            rknn_destroy(context);
            return 5;
        }
        std::cout << argv[arg] << ": status=" << status << " inputs=" << io.n_input << " outputs=" << io.n_output
                  << '\n';
        for (uint32_t index = 0; index < io.n_input; ++index) {
            rknn_tensor_attr attr{};
            attr.index = index;
            status     = rknn_query(context, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
            if (status != RKNN_SUCC) {
                std::cerr << argv[arg] << ": input query failed: " << status << '\n';
                rknn_destroy(context);
                return 5;
            }
            std::cout << "  input[" << index << "] " << attr.name << " " << shape(attr) << " fmt=" << attr.fmt
                      << " type=" << attr.type << '\n';
        }
        for (uint32_t index = 0; index < io.n_output; ++index) {
            rknn_tensor_attr attr{};
            attr.index = index;
            status     = rknn_query(context, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
            if (status != RKNN_SUCC) {
                std::cerr << argv[arg] << ": output query failed: " << status << '\n';
                rknn_destroy(context);
                return 5;
            }
            std::cout << "  output[" << index << "] " << attr.name << " " << shape(attr) << " fmt=" << attr.fmt
                      << " type=" << attr.type << '\n';
        }
        rknn_destroy(context);
    }
    return 0;
}
