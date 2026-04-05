#include "tensednet/tensor.h"
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <sstream>

using namespace std;

namespace tensednet {
    
    static int64_t numel_from_shape(const vector<int64_t>& shape) {
        if (shape.empty()) return 1; // scalar 
        int64_t numel = 1;
        for (auto dim : shape) {
            if (dim <= 0) throw invalid_argument("Shape dimensions must be positive.");
            numel *= dim;
        }
        return numel;
    }

    // Constructor
    
    Tensor::Tensor(vector<float> data, vector<int64_t> shape, bool requires_grad) 
        : data(move(data)), shape(move(shape)), requires_grad(requires_grad) {
        int64_t expected_numel = numel_from_shape(this->shape);
        if (this->data.size() != expected_numel) {
            throw invalid_argument("Data size does not match shape. Expected " + to_string(expected_numel) + " elements, got " + to_string(this->data.size()));
        }
    }

    Tensor Tensor::zeros(const vector<int64_t>& shape, bool requires_grad) {
        int64_t numel = numel_from_shape(shape);
        return Tensor(vector<float>(numel, 0.0f), shape, requires_grad);
    }
    Tensor Tensor::ones(const vector<int64_t>& shape, bool requires_grad) {
        int64_t numel = numel_from_shape(shape);
        return Tensor(vector<float>(numel, 1.0f), shape, requires_grad);
    }
    Tensor Tensor::rand(const vector<int64_t>& shape, bool requires_grad) {
        int64_t numel = numel_from_shape(shape);
        vector<float> data(numel);
        for (auto& val : data) {
            val = static_cast<float>(rand()) / RAND_MAX; // Random float in [0, 1)
        }
        return Tensor(move(data), shape, requires_grad);
    }
}
