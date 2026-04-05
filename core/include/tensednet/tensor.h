#pragma once
#include <vector>
#include <memory>
#include <string>
#include <functional>

using namespace std;
namespace tensednet
{
    class Tensor
    {
    public:
        vector<float> data;
        vector<int64_t> shape;

        bool rquired_grad{false};
        vector<float> grad;
        function<void()> backward_fn;

        Tensor() = default;
        Tensor(vector<float> data, vector<int64_t> shape, bool rquired_grad = false);

        static Tensor zeros(vector<int64_t> shape, bool rquired_grad = false);
        static Tensor ones(vector<int64_t> shape, bool rquired_grad = false);
        static Tensor random(vector<int64_t> shape, bool rquired_grad = false);
        static Tensor from_scalar(float value, bool rquired_grad = false);


        int64_t numel() const;
        int64_t ndim() const{return (int64_t)shape.size();}
        string shape_str() const;

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
