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
     
        std::vector<float>& data() { return impl->data; }
        const std::vector<float>& data() const { return impl->data; }
        std::vector<int64_t>& shape() { return impl->shape; }
        const std::vector<int64_t>& shape() const { return impl->shape; }
        std::vector<float>& grad() { return impl->grad; }
        const std::vector<float>& grad() const { return impl->grad; }
        bool requires_grad() const { return impl->requires_grad; }
        void set_requires_grad(bool rg) { impl->requires_grad = rg; }

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
