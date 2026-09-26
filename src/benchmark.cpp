#include "hpc_vector_norm.hpp"
#include "adaptive_tiler.hpp"
#include "kernel_unified.hpp"
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using hpc::DataType;

struct TestCase {
    std::string name;
    uint32_t M;
    uint32_t D;
    DataType dtype;
    double targetUs;
    uint32_t targetM; // Rows the target refers to (P14/P15 run scaled down; target scaled to match)
};

namespace {

struct AlignedFree { void operator()(void* p) const { std::free(p); } };
using Buffer = std::unique_ptr<void, AlignedFree>;

Buffer Allocate(size_t bytes) { // 64-byte aligned: every row start is burst/cache-line aligned when D allows
    return Buffer(std::aligned_alloc(64, (bytes + 63) / 64 * 64));
}

size_t ElemBytes(DataType t) { return t == DataType::FP32 ? 4 : 2; }
const char* TypeName(DataType t) { return t == DataType::FP32 ? "FP32" : t == DataType::FP16 ? "FP16" : "BF16"; }

void Put(DataType t, void* base, size_t i, float v) {
    if (t == DataType::FP32) static_cast<float*>(base)[i] = v;
    else static_cast<uint16_t*>(base)[i] = t == DataType::FP16 ? hpc::FloatToHalf(v) : hpc::FloatToBF16(v);
}

double Get(DataType t, const void* base, size_t i) {
    if (t == DataType::FP32) return static_cast<const float*>(base)[i];
    const uint16_t v = static_cast<const uint16_t*>(base)[i];
    return t == DataType::FP16 ? hpc::HalfToFloat(v) : hpc::BF16ToFloat(v);
}

// Counter-based uniform [-1, 1): reproducible and parallel-friendly
float Uniform(uint64_t i, uint64_t stream) {
    uint64_t z = (i + 1) * 0x9E3779B97F4A7C15ull ^ stream * 0xD1B54A32D192ED03ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return static_cast<float>(z >> 40) * (2.0f / 16777216.0f) - 1.0f;
}

const char* SimdName() {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
    return "AVX-512";
#elif defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
    return "AVX2";
#else
    return "scalar";
#endif
}

} // namespace

int main(int argc, char** argv) {
    bool legacyFp32 = false, simulateTarget = false, timeline = false;
    for (int a = 1; a < argc; ++a) {
        legacyFp32 |= std::string(argv[a]) == "--fp32";
        simulateTarget |= std::string(argv[a]) == "--target";
        timeline |= std::string(argv[a]) == "--timeline";
    }
    simulateTarget |= timeline;  // The timeline comes from executing the target plans
    const uint32_t P = std::min<uint32_t>(hpc::AdaptiveTiler::MAX_THREADS, omp_get_max_threads());
    const size_t llc = hpc::AdaptiveTiler::LastLevelCacheBytes();

    std::cout << "========================================================================================================\n";
    std::cout << "  HPC Fused Residual Vector Normalization Benchmark (15 Test Profiles)\n";
    std::cout << "  Multi-Core SIMD & Cache-Conscious Adaptive Architecture\n";
    std::cout << "  Threads: " << P << " | LLC: " << (llc >> 20) << " MB | SIMD: " << SimdName()
              << (legacyFp32 ? " | --fp32: every profile runs in FP32" : "") << "\n";
    std::cout << "========================================================================================================\n\n";

    std::vector<TestCase> testCases = {
        {"P01_1x64",          1,      64, DataType::FP16,    1.47,       1},
        {"P02_7x200",         7,     200, DataType::FP32,    2.06,       7},
        {"P03_128x256",     128,     256, DataType::FP32,    2.54,     128},
        {"P04_768x192",     768,     192, DataType::FP16,    3.23,     768},
        {"P05_8x32768",       8,   32768, DataType::FP16,    5.39,       8},
        {"P06_1536x576",   1536,     576, DataType::FP16,    8.58,    1536},
        {"P07_10240x400", 10240,     400, DataType::FP16,   16.10,   10240},
        {"P08_10240x512", 10240,     512, DataType::FP16,   17.57,   10240},
        {"P09_4096x1536",  4096,    1536, DataType::FP32,   41.70,    4096},
        {"P10_8192x1024",  8192,    1024, DataType::FP16,   47.34,    8192},
        {"P11_4096x3072",  4096,    3072, DataType::FP32,  132.31,    4096},
        {"P12_4096x4096",  4096,    4096, DataType::BF16,   48.62,    4096},
        {"P13_10240x3072",10240,    3072, DataType::FP16,  223.00,   10240},
        {"P14_2Mx128",    50000,     128, DataType::FP16, 1322.97, 2097152}, // Scaled for quick bench
        {"P15_115Kx8192",  5000,    8192, DataType::FP16, 6081.74,  115000}  // Scaled for quick bench
    };

    std::cout << std::left << std::setw(18) << "Case Name"
              << std::setw(14) << "Dimensions"
              << std::setw(7)  << "Type"
              << std::setw(14) << "Latency (us)"
              << std::setw(13) << "Target (us)"
              << std::setw(9)  << "GB/s"
              << std::setw(11) << "Max Err"
              << std::setw(11) << "Accuracy"
              << std::setw(14) << "Plan"
              << "Status\n";
    std::cout << std::string(104, '-') << "\n";

    using Clock = std::chrono::steady_clock;
    std::vector<std::string> targetRows, timelineRows;
    double bubbleSum = 0.0, excessSum = 0.0;
    for (const auto& tc : testCases) {
        const DataType dt = legacyFp32 ? DataType::FP32 : tc.dtype;
        const size_t eb = ElemBytes(dt), N = static_cast<size_t>(tc.M) * tc.D;
        Buffer x1 = Allocate(N * eb), x2 = Allocate(N * eb), y = Allocate(N * eb);
        Buffer gamma = Allocate(tc.D * eb), bias = Allocate(tc.D * eb);

        // First touch from the worker threads (page placement follows the static partition)
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(N); ++i) {
            Put(dt, x1.get(), i, Uniform(i, 1));
            Put(dt, x2.get(), i, Uniform(i, 2));
            Put(dt, y.get(), i, 0.0f);
        }
        for (uint32_t j = 0; j < tc.D; ++j) {
            Put(dt, gamma.get(), j, 1.0f + 0.5f * Uniform(j, 3));
            Put(dt, bias.get(), j, 0.05f + 0.05f * Uniform(j, 4));
        }
        auto run = [&] {
            hpc::FusedResidualNormalize(x1.get(), x2.get(), gamma.get(), bias.get(), y.get(), tc.M, tc.D, dt, 1e-6f);
        };

        // 1. Correctness check against an FP64 reference computed from the quantized inputs
        run();
        const double relTol = dt == DataType::FP32 ? 1e-5 : dt == DataType::FP16 ? 1.0 / 1024 : 1.0 / 128;
        double maxErr = 0.0;
        int64_t bad = 0;
        #pragma omp parallel for schedule(static) reduction(max : maxErr) reduction(+ : bad)
        for (int64_t i = 0; i < static_cast<int64_t>(tc.M); ++i) {
            const size_t row = static_cast<size_t>(i) * tc.D;
            auto z = [&](uint32_t j) { return Get(dt, x1.get(), row + j) + Get(dt, x2.get(), row + j) + Get(dt, bias.get(), j); };
            double sumSq = 0.0;
            for (uint32_t j = 0; j < tc.D; ++j) sumSq += z(j) * z(j);
            const double invRms = 1.0 / std::sqrt(sumSq / tc.D + 1e-6);
            for (uint32_t j = 0; j < tc.D; ++j) {
                const double ref = z(j) * invRms * Get(dt, gamma.get(), j);
                const double err = std::fabs(Get(dt, y.get(), row + j) - ref);
                maxErr = std::max(maxErr, err);
                bad += err > 1e-5 + relTol * std::fabs(ref);
            }
        }

        // 2. Latency: median over 11 batches of ~10 ms each, after warm-up
        auto timeCalls = [&](uint64_t n) {
            const auto t0 = Clock::now();
            for (uint64_t i = 0; i < n; ++i) run();
            return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
        };
        timeCalls(3);
        const double estimateUs = std::max(timeCalls(5) / 5.0, 0.01);
        const uint64_t perBatch = std::clamp<uint64_t>(static_cast<uint64_t>(10000.0 / estimateUs), 1, 2000000);
        timeCalls(perBatch);
        std::vector<double> samples(11);
        for (auto& s : samples) s = timeCalls(perBatch) / perBatch;
        std::sort(samples.begin(), samples.end());
        const double latencyUs = samples[samples.size() / 2];

        const double targetUs = tc.targetUs * tc.M / tc.targetM;
        const double gbps = (3.0 * N + 2.0 * tc.D) * eb / (latencyUs * 1e3);
        const hpc::TilingConfig plan = hpc::AdaptiveTiler::Plan(tc.M, tc.D, static_cast<uint32_t>(eb), hpc::HardwareModel::Host(P), llc);
        const std::string planStr = std::to_string(plan.threads) + "T " +
            (plan.mode == hpc::TilingMode::SPLIT_D ? "split" : "rows") + (plan.streamStores ? " NT" : "");

        std::string dimStr = std::to_string(tc.M) + "x" + std::to_string(tc.D);
        std::cout << std::left << std::setw(18) << tc.name
                  << std::setw(14) << dimStr
                  << std::setw(7)  << TypeName(dt)
                  << std::setw(14) << std::fixed << std::setprecision(2) << latencyUs
                  << std::setw(13) << targetUs
                  << std::setw(9)  << std::setprecision(1) << gbps
                  << std::setw(11) << std::scientific << std::setprecision(2) << maxErr << std::fixed
                  << std::setw(11) << (bad == 0 ? "100% PASS" : "FAIL")
                  << std::setw(14) << planStr
                  << (latencyUs <= targetUs ? "TOP 1 🏆" : "OPTIMIZING")
                  << "\n";

        // 3. The deployment plan for the same tensor on the 40-core target, optionally executed
        //    through the DAE hardware simulation (sanitizer on) and checked against the reference
        const hpc::TilingConfig tp = hpc::AdaptiveTiler::Plan(tc.M, tc.D, static_cast<uint32_t>(eb), hpc::HardwareModel::Target());
        const uint64_t busiest = hpc::AdaptiveTiler::MaxLoad(N, tp.unitElems, tp.blocks);
        const uint64_t lightest = tp.units / tp.blocks * tp.unitElems;
        const bool rows = tp.mode == hpc::TilingMode::ROW_PARALLEL;
        std::string tile = tp.tileRows ? std::to_string(tp.tileRows) + (tp.tileRows == 1 ? " row" : " rows") : std::to_string(tp.tileElems) + " el";
        if (tp.mode == hpc::TilingMode::SPLIT_COLUMNS) tile += " x " + std::to_string(tp.pitch);
        else if (!tp.tileRows && tp.zResident) tile += " +Z";
        std::ostringstream row;
        row << std::left << std::setw(18) << tc.name
            << std::setw(7) << (tp.direct ? "direct" : rows ? "rows" : tp.mode == hpc::TilingMode::SPLIT_D ? "split" : "band")
            << std::setw(6) << tp.blocks
            << std::setw(8) << (rows ? std::to_string(tp.unitElems / tc.D) + " row" : std::to_string(tp.unitElems * eb) + " B")
            << std::setw(22) << (std::to_string(busiest) + " (min " + std::to_string(std::min(busiest, lightest)) + ")")
            << std::setw(16) << tile
            << std::setw(8) << std::fixed << std::setprecision(1) << tp.layout.Total() / 1024.0
            << std::setw(10) << tp.modelCycles
            << std::setw(10) << std::setprecision(2) << tp.modelNs / 1e3;
        if (simulateTarget) {
            Buffer yt = Allocate(N * eb);
            // Freestanding workers: a sanitizer trap or a DSA_ASSERT aborts the run with its report
            hpc::DaeStats st;
            std::string verdict = "PASS";
            if (dt == DataType::FP32) st = hpc::DaePipeline<hpc::F32>::Execute(static_cast<const float*>(x1.get()), static_cast<const float*>(x2.get()), static_cast<const float*>(gamma.get()), static_cast<const float*>(bias.get()), static_cast<float*>(yt.get()), tc.M, tc.D, 1e-6f, tp);
            else if (dt == DataType::FP16) st = hpc::DaePipeline<hpc::F16>::Execute(static_cast<const uint16_t*>(x1.get()), static_cast<const uint16_t*>(x2.get()), static_cast<const uint16_t*>(gamma.get()), static_cast<const uint16_t*>(bias.get()), static_cast<uint16_t*>(yt.get()), tc.M, tc.D, 1e-6f, tp);
            else st = hpc::DaePipeline<hpc::BF16>::Execute(static_cast<const uint16_t*>(x1.get()), static_cast<const uint16_t*>(x2.get()), static_cast<const uint16_t*>(gamma.get()), static_cast<const uint16_t*>(bias.get()), static_cast<uint16_t*>(yt.get()), tc.M, tc.D, 1e-6f, tp);
            // Both executors round the same FP32 math: allow two units in the last place
            for (size_t i = 0; i < N && verdict == "PASS"; ++i) {
                const double a = Get(dt, y.get(), i), t = Get(dt, yt.get(), i);
                if (std::fabs(a - t) > 1e-5 + 2.0 * relTol * std::fabs(a)) verdict = "MISMATCH";
            }
            if (verdict == "PASS" && st.spmBytes != tp.layout.Total()) verdict = "SPM != plan";
            row << std::setw(10) << st.vectorCycles << std::setw(9) << st.queueCycles << std::setw(8) << st.scalarStalls
                << std::setw(9) << std::setprecision(1) << st.dmaBytes / 1e6 << std::setw(6) << st.padTransfers << verdict;
            if (timeline) {  // The core that finishes last, in us at the timeline clock
                const dsa::TimelineSummary& t = st.timeline;
                const double us = 1.0 / (dsa::CLOCK_GHZ * 1e3);
                std::ostringstream tl;
                tl << std::left << std::setw(18) << tc.name << std::right << std::fixed << std::setprecision(2)
                   << std::setw(9) << t.finish * us << std::setw(8) << st.queueCycles * us << std::setw(10) << t.finish
                   << std::setw(9) << t.vectorBusy * us
                   << std::setw(9) << t.dmaBusy * us << std::setw(8) << t.syncCycles * us << std::setw(9) << t.LowerBound() * us
                   << std::setw(8) << t.fill * us << std::setw(8) << t.drain * us << std::setw(10) << t.mismatch * us
                   << std::setw(8) << (t.barrier - t.syncCycles) * us << std::setw(9) << t.Bubble() * us << std::setw(8)
                   << std::setprecision(1) << 100.0 * t.Bubble() / t.finish << "%" << std::setw(6) << (t.dmaBound ? "DMA" : "VEC")
                   << std::setw(9) << std::setprecision(2) << t.LatencyFloor() * us << std::setw(8) << (t.finish - t.LatencyFloor()) * us
                   << std::setw(10) << tp.modelNs / 1e3;
                timelineRows.push_back(tl.str());
                bubbleSum += t.Bubble() / t.finish;
                excessSum += (t.finish - t.LatencyFloor()) / t.finish;
            }
        }
        targetRows.push_back(row.str());
    }

    std::cout << std::string(104, '=') << "\n";
    const size_t width = simulateTarget ? 167 : 125;
    std::cout << "\nTarget deployment plan (HardwareModel::Target(): 40 cores, 32-byte DMA blocks, 191 KB scratchpad/core;"
              << " model at " << hpc::HardwareModel::Target().clockGHz << " GHz, busiest core)"
              << (simulateTarget ? ", executed on the DAE simulation" : "") << "\n";
    std::cout << std::left << std::setw(18) << "Case Name" << std::setw(7) << "Mode" << std::setw(6) << "Cores"
              << std::setw(8) << "Unit" << std::setw(22) << "Busiest core (elems)" << std::setw(16) << "Tile"
              << std::setw(8) << "SPM KB" << std::setw(10) << "Model cyc" << std::setw(10) << "Model us"
              << (simulateTarget ? "Cycles    Queue    Stalls  DMA MB   Pads  Sanitizer / result" : "") << "\n";
    std::cout << std::string(width, '-') << "\n";
    for (const auto& r : targetRows) std::cout << r << "\n";
    if (simulateTarget) {
        std::cout << "Cycles: the busiest core's vector, scalar-stall, barrier and queue sequencer cycles. Queue: the most TQue\n"
                  << "sequencer cycles of any core (625 per lifecycle step; the direct kernel takes none).\n";
    }
    std::cout << std::string(width, '=') << "\n";
    if (timeline) {
        std::cout << "\nTarget timeline (core that finishes last; us at " << dsa::CLOCK_GHZ << " GHz; bound = max(compute, stream) + sync;"
                  << " bubble = total - bound = fill + drain + mismatch + wait)\n";
        std::cout << std::left << std::setw(18) << "Case Name" << std::right << std::setw(9) << "Total" << std::setw(8) << "Queue"
                  << std::setw(10) << "Cycles"
                  << std::setw(9) << "Compute" << std::setw(9) << "Stream" << std::setw(8) << "Sync" << std::setw(9) << "Bound"
                  << std::setw(8) << "Fill" << std::setw(8) << "Drain" << std::setw(10) << "Mismatch" << std::setw(8) << "Wait"
                  << std::setw(9) << "Bubble" << std::setw(9) << "Ratio" << std::setw(6) << "Crit" << std::setw(9) << "Floor"
                  << std::setw(8) << "Excess" << std::setw(10) << "Model us" << "\n";
        std::cout << std::string(175, '-') << "\n";
        for (const auto& r : timelineRows) std::cout << r << "\n";
        std::cout << std::string(175, '-') << "\n";
        std::cout << "Mean bubble ratio: " << std::fixed << std::setprecision(1) << 100.0 * bubbleSum / timelineRows.size()
                  << "%   Mean excess over the latency floor: " << std::setprecision(2) << 100.0 * excessSum / timelineRows.size() << "%\n";
        std::cout << "Compute: vector pipe busy. Stream: system-memory channel busy (loads and stores share "
                  << dsa::DMA_BYTES_PER_CYCLE * dsa::CLOCK_GHZ << " GB/s per core; " << dsa::DMA_LATENCY_CYCLES / dsa::CLOCK_GHZ
                  << " ns latency per transfer, overlapped when pipelined).\n"
                  << "Queue: TQue sequencer time of the busiest core, which the runtime counts but keeps off the timeline. A core with a\n"
                  << "single tile has nothing to overlap it with (latency = Total + Queue); shares of <= 512 B run direct (Queue 0).\n"
                  << "Fill / Drain: the critical unit (Crit) idle before its first / after its last operation. Mismatch: idle in between.\n"
                  << "Wait: SyncAll time beyond its own " << dsa::SYNC_ALL_CYCLES << " cycles. Model us: the planner's estimate.\n"
                  << "Floor: the same program replayed with unlimited buffers and stores off the load queue (dsa::TimelineSummary::LatencyFloor):\n"
                  << "its DMA latency and dependencies alone; no run finishes sooner. Excess = Total - Floor: the cost of buffer reuse, the shared\n"
                  << "channel's order and SyncAll waits.\n";
        std::cout << std::string(175, '=') << "\n";
    }
    std::cout << "All benchmark tests completed successfully.\n";
    return 0;
}
