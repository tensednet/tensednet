#pragma once
#include <vector>
#include <memory>
#include <string>
#include <functional>

namespace tensednet
{
    class Tensor
    {
    public:
        std::vector<float> data;
        std::vector<int64_t> shape;

        bool rquired_grad{false};
        std::vector<float> grad;
        function<void()> backward_fn;

        Tensor() = default;
        Tensor(std::vector<float> data, std::vector<int64_t> shape, bool rquired_grad = false);

        static Tensor zeros(std::vector<int64_t> shape, bool rquired_grad = false);
        static Tensor ones(std::vector<int64_t> shape, bool rquired_grad = false);
        static Tensor random(std::vector<int64_t> shape, bool rquired_grad = false);
        static Tensor from_scalar(float value, bool rquired_grad = false);


        int64_t numel() const;
        int64_t ndim() const{return (int64_t)shape.size();}
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
