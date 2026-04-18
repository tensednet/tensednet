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

    // metadata

    int64_t Tensor::numel() const {
        return numel_from_shape(shape);
    }

    string Tensor::shape_str() const {
        ostringstream ss;
        ss << "(";
        for (size_t i = 0; i < shape.size(); ++i) {
            ss << shape[i];
            if (i < shape.size() - 1) ss << ", ";
        }
        ss << ")";
        return ss.str();
    }

    Tensor Tensor::operator+(const Tensor& other) const {
        if (data.size() != other.data.size()) {
            throw runtime_error("Shape mismatch for addition: " + shape_str() + " vs " + other.shape_str());
        }
        vector<float> out_data(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            out_data[i] = data[i] + other.data[i];
        }
        bool rg = requires_grad || other.requires_grad;
        Tensor out(move(out_data), shape, rg);
        if (rg) {
            auto self_ref = make_shared<Tensor>(*this);
            auto other_ref = make_shared<Tensor>(other);
            auto out_ref = make_shared<Tensor>(out);

            out_ref->backward_fn = [self_ref, other_ref, out_ref]() {
                if (self_ref->requires_grad) {
                    if (self_ref->grad.empty())
                        self_ref->grad.assign(self_ref->data.size(), 0.0f);
                    for (size_t i = 0; i < self_ref->grad.size(); ++i) {
                        self_ref->grad[i] += out_ref->grad[i]; // dL/dself = dL/dout * dout/dself = dL/dout * 1
                                                               // since dout/dself = 1 for addition
                    }
                }
                if (other_ref->requires_grad) {
                    if (other_ref->grad.empty())
                        other_ref->grad.assign(other_ref->data.size(), 0.0f);
                    for (size_t i = 0; i < other_ref->grad.size(); ++i) {
                        other_ref->grad[i] += out_ref->grad[i]; // dL/dother = dL/dout * dout/dother = dL/dout * 1
                                                                // since dout/dother = 1 for addition
                    }
                }
            };
            return *out_ref;
        }
        return out;
    }
}
