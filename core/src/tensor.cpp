#include "tensednet/tensor.h"
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <algorithm>
#include <unordered_set>
#include <functional>

namespace tensednet {
    
    static int64_t numel_from_shape(const std::vector<int64_t>& shape) {
        int64_t n = 1;
        for (auto d : shape) {
            if (d <= 0) throw std::invalid_argument("Shape dimensions must be positive.");
            n *= d;
        }
        return n;
    }

    // ----- constructor -----

    static std::shared_ptr<TensorImpl> make_impl(std::vector<int64_t> shape, bool requires_grad) {
        auto impl = std::make_shared<TensorImpl>();
        int64_t n = numel_from_shape(shape);
        impl->storage = std::make_shared<Storage>(n);
        impl->offset = 0;
        impl->shape = std::move(shape);
        impl->strides = contiguous_strides(impl->shape);
        impl->requires_grad = requires_grad;
        return impl;
    }
    
    Tensor::Tensor() : impl(std::make_shared<TensorImpl>()) {}
     
    Tensor::Tensor(std::vector<float> data, std::vector<int64_t> shape, bool requires_grad) {
        int64_t expected = numel_from_shape(shape);
        if ((int64_t)data.size() != expected) {
            throw std::invalid_argument(
                "Data size does not match shape. Expected " + std::to_string(expected) +
                " elements, got " + std::to_string(data.size()));
        }
        impl = make_impl(shape, requires_grad);
        std::copy(data.begin(), data.end(), impl->storage->ptr.get());
    }
     
    Tensor Tensor::zeros(const std::vector<int64_t>& shape, bool requires_grad) {
        // make_impl alr init with zeros
        Tensor t;
        t.impl = make_impl(shape, requires_grad);
        return t;
    }
     
    Tensor Tensor::ones(const std::vector<int64_t>& shape, bool requires_grad) {
        Tensor t;
        t.impl = make_impl(shape, requires_grad);
        std::fill(t.data_ptr(), t.data_ptr() + t.numel(), 1.0f);
        return t;
    }
     
    Tensor Tensor::rand(const std::vector<int64_t>& shape, bool requires_grad) {
        Tensor t;
        t.impl = make_impl(shape, requires_grad);
        float* p = t.data_ptr();
        for (int64_t i = 0; i < t.numel(); ++i)
            p[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        return t;
    }
     
    Tensor Tensor::from_scalar(float value, bool requires_grad) {
        Tensor t;
        t.impl = make_impl({}, requires_grad);
        t.data_ptr()[0] = value;
        return t;
    }

    Tensor Tensor::from_blob(float* raw, std::vector<int64_t> shape,
                              std::function<void(float*)> deleter, bool requires_grad) {
        Tensor t;
        t.impl = std::make_shared<TensorImpl>();
        int64_t n = numel_from_shape(shape);
        t.impl->storage = std::make_shared<Storage>(raw, n, std::move(deleter));
        t.impl->offset = 0;
        t.impl->shape = std::move(shape);
        t.impl->strides = contiguous_strides(t.impl->shape);
        t.impl->requires_grad = requires_grad;
        return t;
    }

    // ----- element access -----
     
    float* Tensor::grad_ptr() {
        if (!impl->grad_storage) {
            impl->grad_storage = std::make_shared<Storage>(numel());
        }
        return impl->grad_storage->ptr.get();
    }
     
    std::vector<float> Tensor::data() const {
        const float* p = data_ptr();
        return std::vector<float>(p, p + numel());
    }
     
    std::vector<float> Tensor::grad() const {
        if (!impl->grad_storage) return std::vector<float>(numel(), 0.0f);
        float* p = impl->grad_storage->ptr.get();
        return std::vector<float>(p, p + numel());
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

    static void accumulate_grad(const std::shared_ptr<TensorImpl>& node, const float* contrib, int64_t count) {
        if (!node->grad_storage) node->grad_storage = std::make_shared<Storage>(count);
        float* g = node->grad_storage->ptr.get();
        for (int64_t i = 0; i < count; ++i) g[i] += contrib[i];
    }
     
    static int64_t numel_of(const std::shared_ptr<TensorImpl>& t) { return numel_from_shape(t->shape); }

    // ----- eleement-wsie operations -----

    Tensor Tensor::operator+(const Tensor& other) const {
        if (numel() != other.numel())
            throw std::runtime_error("Shape mismatch for addition: " + shape_str() + " vs " + other.shape_str());
     
        Tensor out;
        out.impl = make_impl(impl->shape, impl->requires_grad || other.impl->requires_grad);
        const float* a_p = data_ptr();
        const float* b_p = other.data_ptr();
        float* o_p = out.data_ptr();
        int64_t n = numel();
        for (int64_t i = 0; i < n; ++i) o_p[i] = a_p[i] + b_p[i];
     
        if (out.impl->requires_grad) {
            auto a = impl, b = other.impl, o = out.impl;
            o->parents = {a, b};
            o->backward_fn = [a, b, o]() {
                int64_t n = numel_of(o);
                const float* og = o->grad_storage->ptr.get();
                if (a->requires_grad) accumulate_grad(a, og, n); // d(a+b)/da = 1
                if (b->requires_grad) accumulate_grad(b, og, n); // d(a+b)/db = 1
            };
        }
        return out;
    }

    Tensor Tensor::operator-(const Tensor& other) const {
        if (numel() != other.numel())
            throw std::runtime_error("Shape mismatch for subtraction: " + shape_str() + " vs " + other.shape_str());
     
        Tensor out;
        out.impl = make_impl(impl->shape, impl->requires_grad || other.impl->requires_grad);
        const float* a_p = data_ptr();
        const float* b_p = other.data_ptr();
        float* o_p = out.data_ptr();
        int64_t n = numel();
        for (int64_t i = 0; i < n; ++i) o_p[i] = a_p[i] - b_p[i];
     
        if (out.impl->requires_grad) {
            auto a = impl, b = other.impl, o = out.impl;
            o->parents = {a, b};
            o->backward_fn = [a, b, o]() {
                int64_t n = numel_of(o);
                const float* og = o->grad_storage->ptr.get();
                if (a->requires_grad) accumulate_grad(a, og, n); // d(a-b)/da = 1
                if (b->requires_grad) {
                    std::vector<float> neg(n);
                    for (int64_t i = 0; i < n; ++i) neg[i] = -og[i];
                    accumulate_grad(b, neg.data(), n); // d(a-b)/db = -1
                }
            };
        }
        return out;
    }

    Tensor Tensor::operator*(const Tensor& other) const {
        if (numel() != other.numel())
            throw std::runtime_error("Shape mismatch for multiplication: " + shape_str() + " vs " + other.shape_str());
     
        Tensor out;
        out.impl = make_impl(impl->shape, impl->requires_grad || other.impl->requires_grad);
        const float* a_p = data_ptr();
        const float* b_p = other.data_ptr();
        float* o_p = out.data_ptr();
        int64_t n = numel();
        for (int64_t i = 0; i < n; ++i) o_p[i] = a_p[i] * b_p[i];
     
        if (out.impl->requires_grad) {
            auto a = impl, b = other.impl, o = out.impl;
            o->parents = {a, b};
            o->backward_fn = [a, b, o]() {
                int64_t n = numel_of(o);
                const float* og = o->grad_storage->ptr.get();
                const float* ad = a->storage->ptr.get() + a->offset;
                const float* bd = b->storage->ptr.get() + b->offset;
                if (a->requires_grad) {
                    std::vector<float> contrib(n);
                    for (int64_t i = 0; i < n; ++i) contrib[i] = og[i] * bd[i]; // d(ab)/da = b
                    accumulate_grad(a, contrib.data(), n);
                }
                if (b->requires_grad) {
                    std::vector<float> contrib(n);
                    for (int64_t i = 0; i < n; ++i) contrib[i] = og[i] * ad[i]; // d(ab)/db = a
                    accumulate_grad(b, contrib.data(), n);
                }
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
        impl->grad_storage = std::make_shared<Storage>(numel());
        std::fill(impl->grad_storage->ptr.get(), impl->grad_storage->ptr.get() + numel(), 1.0f);
     
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            if ((*it)->backward_fn) (*it)->backward_fn();
        }
    }
     
    void Tensor::zero_grad() {
        impl->grad_storage.reset();
    }
}
