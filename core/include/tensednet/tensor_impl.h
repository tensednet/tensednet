#pragma once
#include <vector>
#include <cstdint>
#include <functional>
#include <memory>

namespace tensednet {

    struct Storage {
        std::shared_ptr<float[]> ptr;
        int64_t size{0};
        Storage() = default;

        explicit Storage(int64_t n)
            : ptr(new float[n](), std::default_delete<float[]>()), size(n) {}

        Storage(float* raw, int64_t n, std::function<void(float*)> deleter)
            : ptr(raw, deleter), size(n) {}
    };

    struct TensorImpl {
        std::shared_ptr<Storage> storage;
        int64_t offset{0};
        std::vector<int64_t> shape;
        std::vector<int64_t> strides;

        bool requires_grad{false};
        std::shared_ptr<TensorImpl> grad_storage;
        
        std::function<void()> backward_fn;
         
        std::vector<std::shared_ptr<TensorImpl>> parents;
         
        TensorImpl() = default;
    };
    
    inline std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape) {
        std::vector<int64_t> strides(shape.size());
        int64_t stride = 1;
        for (int64_t i = (int64_t)shape.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }
}
