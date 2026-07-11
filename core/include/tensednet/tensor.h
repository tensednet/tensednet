#pragma once
#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include "tensednet/tensor_impl.h"

namespace tensednet
{
    class Tensor
    {
    public:
        std::shared_ptr<TensorImpl> impl;
     
        Tensor();
        Tensor(std::vector<float> data, std::vector<int64_t> shape, bool requires_grad = false);

        static Tensor zeros(const std::vector<int64_t>& shape, bool requires_grad = false);
        static Tensor ones(const std::vector<int64_t>& shape, bool requires_grad = false);
        static Tensor rand(const std::vector<int64_t>& shape, bool requires_grad = false);
        static Tensor from_scalar(float value, bool requires_grad = false);

        static Tensor from_blob(float* raw, std::vector<int64_t> shape, std::function<void(float*)> deleter = nullptr, bool requires_grad = false);

        float* data_ptr() { return impl->storage->ptr.get() + impl->offset; }
        const float* data_ptr() const { return impl->storage->ptr.get() + impl->offset; }
        float* grad_ptr(); // alloc grad_storage if not exists (on first call)
        bool has_grad() const { return (bool)impl->grad_storage; }
     
        const std::vector<int64_t>& shape() const { return impl->shape; }
        const std::vector<int64_t>& strides() const { return impl->strides; }
        bool requires_grad() const { return impl->requires_grad; }
        void set_requires_grad(bool rg) { impl->requires_grad = rg; }

        std::vector<float> data() const;
        std::vector<float> grad() const;

        int64_t numel() const;
        int64_t ndim() const { return (int64_t)impl->shape.size(); }
        std::string shape_str() const;

        Tensor operator+(const Tensor& other) const;
        Tensor operator*(const Tensor& other) const;
        Tensor operator-(const Tensor& other) const;
        Tensor matmul(const Tensor& other) const;
        Tensor relu() const;
        Tensor sum() const;
        Tensor mean() const;

        void backward();
        void zero_grad();
    };
}
