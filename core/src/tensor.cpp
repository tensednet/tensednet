#include "tensednet/tensor.h"
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <unordered_set>
#include <functional>

namespace tensednet {
    
    static int64_t numel_from_shape(const std::vector<int64_t>& shape) {
        if (shape.empty()) return 1; // scalar
        int64_t n = 1;
        for (auto dim : shape) {
            if (dim <= 0) throw std::invalid_argument("Shape dimensions must be positive.");
            n *= dim;
        }
        return n;
    }

    // ----- constructor -----

    Tensor::Tensor() : impl(std::make_shared<TensorImpl>()) {}
    
    Tensor::Tensor(std::vector<float> data, std::vector<int64_t> shape, bool requires_grad) {
        int64_t expected = numel_from_shape(shape);
        if ((int64_t)data.size() != expected) {
            throw std::invalid_argument(
                "Data size does not match shape. Expected " + std::to_string(expected) +
                " elements, got " + std::to_string(data.size()));
        }
        impl = std::make_shared<TensorImpl>(std::move(data), std::move(shape), requires_grad);
    }
     
    Tensor Tensor::zeros(const std::vector<int64_t>& shape, bool requires_grad) {
        int64_t n = numel_from_shape(shape);
        return Tensor(std::vector<float>(n, 0.0f), shape, requires_grad);
    }
     
    Tensor Tensor::ones(const std::vector<int64_t>& shape, bool requires_grad) {
        int64_t n = numel_from_shape(shape);
        return Tensor(std::vector<float>(n, 1.0f), shape, requires_grad);
    }
     
    Tensor Tensor::rand(const std::vector<int64_t>& shape, bool requires_grad) {
        int64_t n = numel_from_shape(shape);
        std::vector<float> data(n);
        for (auto& v : data) v = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        return Tensor(std::move(data), shape, requires_grad);
    }
     
    Tensor Tensor::from_scalar(float value, bool requires_grad) {
        return Tensor(std::vector<float>{value}, std::vector<int64_t>{}, requires_grad);
    }

    // ----- metadata -----

    int64_t Tensor::numel() const {
        return numel_from_shape(impl->shape);
    }

    std::string Tensor::shape_str() const {
        std::ostringstream ss;
        ss << "(";
        for (size_t i = 0; i < impl->shape.size(); ++i) {
            ss << impl->shape[i];
            if (i + 1 < impl->shape.size()) ss << ", ";
        }
        ss << ")";
        return ss.str();
    }

    // ----- help ahh funcs -----

    static void accumulate_grad(const std::shared_ptr<TensorImpl>& node, const std::vector<float>& contrib) {
        if (node->grad.size() != node->data.size())
            node->grad.assign(node->data.size(), 0.0f);
        for (size_t i = 0; i < contrib.size(); ++i)
            node->grad[i] += contrib[i];
    }

    // ----- eleement-wsie operations -----

    Tensor Tensor::operator+(const Tensor& other) const {
        if (impl->data.size() != other.impl->data.size())
            throw std::runtime_error("Shape mismatch for addition: " + shape_str() + " vs " + other.shape_str());
     
        std::vector<float> out_data(impl->data.size());
        for (size_t i = 0; i < out_data.size(); ++i)
            out_data[i] = impl->data[i] + other.impl->data[i];
     
        bool rg = impl->requires_grad || other.impl->requires_grad;
        Tensor out(std::move(out_data), impl->shape, rg);
     
        if (rg) {
            auto a = impl;
            auto b = other.impl;
            auto o = out.impl;
            o->parents = {a, b};
            o->backward_fn = [a, b, o]() {
                if (a->requires_grad) accumulate_grad(a, o->grad);
                if (b->requires_grad) accumulate_grad(b, o->grad);
            };
        }
        return out;
    }


    // ----- backward / zero_grad impl -----

    void Tensor::backward() {
        // build the grpah in topological order using dfs
        // this makes sure that before a node's backward_fn runs
        // every node that depnds on it has alr been added to node->grad
        std::vector<TensorImpl*> topo;
        std::unordered_set<TensorImpl*> visited;
     
        std::function<void(const std::shared_ptr<TensorImpl>&)> build =
            [&](const std::shared_ptr<TensorImpl>& t) {
                if (!t || visited.count(t.get())) return;
                visited.insert(t.get());
                for (auto& p : t->parents) build(p);
                topo.push_back(t.get());
            };
        build(impl);
     
        // start this tensor's grad with 1s (dL/dL = 1)
        // assumes backward() is called on the final scalar loss
        impl->grad.assign(impl->data.size(), 1.0f);
     
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            if ((*it)->backward_fn) (*it)->backward_fn();
        }
    }
     
    void Tensor::zero_grad() {
        impl->grad.clear();
    }
}
