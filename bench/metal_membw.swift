// Metal GPU read-bandwidth microbenchmark (M4 Max).
// Streams a large buffer with float4 grid-stride reads, sums to defeat DCE.
import Metal
import QuartzCore

let src = """
#include <metal_stdlib>
using namespace metal;
kernel void bwread(device const float4 *buf [[buffer(0)]],
                   device float *out [[buffer(1)]],
                   constant uint &n4 [[buffer(2)]],
                   uint gid [[thread_position_in_grid]],
                   uint gsz [[threads_per_grid]]) {
    float4 acc = 0.0f;
    for (uint i = gid; i < n4; i += gsz) acc += buf[i];
    float s = acc.x + acc.y + acc.z + acc.w;
    if (s == 123456789.0f) out[gid % 1024] = s; // never true; defeats DCE
}
"""

let dev = MTLCreateSystemDefaultDevice()!
print("device: \(dev.name), recommendedMaxWorkingSetSize=\(dev.recommendedMaxWorkingSetSize / (1<<30)) GiB")
let lib = try dev.makeLibrary(source: src, options: nil)
let pso = try dev.makeComputePipelineState(function: lib.makeFunction(name: "bwread")!)
let q = dev.makeCommandQueue()!

let bytes = 4 << 30 // 4 GiB
let buf = dev.makeBuffer(length: bytes, options: .storageModePrivate)!
let out = dev.makeBuffer(length: 4096, options: .storageModePrivate)!
var n4 = UInt32(bytes / 16)

// grid: enough threads to saturate; each reads n4/gsz float4s
let tgs = 256
let ntg = 40 * 32 * 4 // 40 cores, plenty of threadgroups
let grid = MTLSize(width: ntg * tgs, height: 1, depth: 1)
let tg = MTLSize(width: tgs, height: 1, depth: 1)

func run(_ iters: Int) -> Double {
    let cb = q.makeCommandBuffer()!
    for _ in 0..<iters {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(buf, offset: 0, index: 0)
        enc.setBuffer(out, offset: 0, index: 1)
        enc.setBytes(&n4, length: 4, index: 2)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }
    let t0 = CACurrentMediaTime()
    cb.commit()
    cb.waitUntilCompleted()
    let dt = CACurrentMediaTime() - t0
    return Double(iters) * Double(bytes) / dt / 1e9
}

_ = run(2) // warmup
for trial in 1...3 {
    let gbps = run(8)
    print(String(format: "trial %d: %.1f GB/s read", trial, gbps))
}
