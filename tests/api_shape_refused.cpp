// The API shape of the target (docs/TARGET_API_SHAPE.md), checked at compile time by
// tests/check_refused_forms.cmake: CASE 0 uses the target's forms and must compile; every other CASE
// uses one form the target does not have and must fail on the static_assert that names it.
#include "dsa_runtime.hpp"

using namespace dsa;

int main() {
    TPipe pipe;
    TBuf<QuePosition::VECCALC> buf;
    TBuf<QuePosition::VECOUT> out;
    pipe.InitBuffer(buf, 2048);
    pipe.InitBuffer(out, 256);
    LocalTensor<float> l = buf.Get<float>(), o = out.Get<float>();
    LocalTensor<half> h = buf.Get<half>()[512];
    alignas(64) static float mem[64];
    float* p = mem;
    GlobalTensor<float> g;
    g.SetGlobalBuffer(mem, 64);
#if CASE == 0  // The target's forms
    DataCopy(l, g, 8);
    DataCopy(g, o, 8);
    DataCopyPad(l, g[1], DataCopyExtParams{1, 20, 0, 0, 0}, DataCopyPadExtParams<float>{true, 0, 0, 0.0f});
    DataCopyPad(g[1], o, DataCopyExtParams{1, 20, 0, 0, 0});
    LoadPad(l, g, 5);
    StorePad(g, o, 5);
    Cast(h, l, RoundMode::CAST_RINT, 8u);
    Cast(l, h, RoundMode::CAST_NONE, 8u);
    BlockReduceSum(l[256], l, 1, 64, 1, 1, 8);
    WholeReduceSum(l[256], l, 64, 1, 1, 1, 8);
#elif CASE == 1
    DataCopy(l, p, 8);
#elif CASE == 2
    DataCopy(p, o, 8);
#elif CASE == 3
    DataCopyPad(l, p, 5u);
#elif CASE == 4
    DataCopyPad(p, o, 5u);
#elif CASE == 5
    DataCopyPad(l, g, 5u);
#elif CASE == 6
    DataCopyPad(g, o, 5u);
#elif CASE == 7
    DataCopyPad(l, p, DataCopyExtParams{1, 20, 0, 0, 0});
#elif CASE == 8
    DataCopyPad(p, o, DataCopyExtParams{1, 20, 0, 0, 0});
#elif CASE == 9
    LoadPad(l, p, 5);
#elif CASE == 10
    StorePad(p, o, 5);
#elif CASE == 11
    Cast(l, h, 8u, [](half x) { return float(x); });
#elif CASE == 12
    Cast(l, h, 8u);
#elif CASE == 13
    (void)VectorReduceSum(l, 64);
#elif CASE == 14
    (void)VectorInvRms(1.0f, 0.5f, 1e-6f);
#elif CASE == 15
    (void)VectorInvRms(1.0f, 64u, 1e-6f);
#elif CASE == 16
    BlockReduceSum(l[256], l, 64u);
#elif CASE == 17
    WholeReduceSum(l[256], l, 64u);
#elif CASE == 18
    LocalMemAllocator<Hardware::Scratchpad> spm;
    (void)spm;
#elif CASE == 19
    l.pos = QuePosition::VECIN;
#endif
    (void)l, (void)o, (void)h, (void)p, (void)g;
    return 0;
}
