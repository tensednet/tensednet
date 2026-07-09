#pragma once
#include <vector>
#include <cstdint>
#include <functional>
#include <memory>

namespace tensednet {
    struct TensorImpl {
        std::vector<float> data;
        std::vector<int64_t> shape;
        bool requires_grad{false};
        std::vector<float> grad;
        
        std::function<void()> backward_fn;
         
        std::vector<std::shared_ptr<TensorImpl>> parents;
         
        TensorImpl() = default;
        TensorImpl(std::vector<float> data_, std::vector<int64_t> shape_, bool requires_grad_ = false) : data(std::move(data_)), shape(std::move(shape_)), requires_grad(requires_grad_) {}
    };
}
