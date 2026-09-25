#include "hpc_vector_norm.hpp"
#include "kernel_unified.hpp"

namespace hpc {

template <class C>
static void Run(const void* x1, const void* x2, const void* gamma, const void* bias, void* y,
                uint32_t rows, uint32_t cols, float eps) {
    using S = typename C::S;
    KernelUnifiedPipeline<C>::Execute(static_cast<const S*>(x1), static_cast<const S*>(x2),
                                      static_cast<const S*>(gamma), static_cast<const S*>(bias),
                                      static_cast<S*>(y), rows, cols, eps);
}

void FusedResidualNormalize(const void* x1, const void* x2, const void* gamma, const void* bias, void* y,
                            uint32_t rows, uint32_t cols, DataType dtype, float eps) {
    switch (dtype) {
        case DataType::FP32: Run<F32>(x1, x2, gamma, bias, y, rows, cols, eps); break;
        case DataType::FP16: Run<F16>(x1, x2, gamma, bias, y, rows, cols, eps); break;
        case DataType::BF16: Run<BF16>(x1, x2, gamma, bias, y, rows, cols, eps); break;
    }
}

} // namespace hpc
