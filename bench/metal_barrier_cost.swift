// Measure Apple GPU inter-dispatch costs: empty kernels, with/without
// memoryBarrier, serial vs concurrent encoder, and a real dependency chain.
import Metal
import QuartzCore

let src = """
#include <metal_stdlib>
using namespace metal;
kernel void nop(device float* buf [[buffer(0)]],
                uint gid [[thread_position_in_grid]]) {
    if (gid == 0xFFFFFFFF) buf[0] = 1.0f;  // never
}
kernel void touch(device float* buf [[buffer(0)]],
                  uint gid [[thread_position_in_grid]]) {
    buf[gid] += 1.0f;   // real read-write so hazards/barriers matter
}
"""

let dev = MTLCreateSystemDefaultDevice()!
let lib = try dev.makeLibrary(source: src, options: nil)
let nopPSO = try dev.makeComputePipelineState(function: lib.makeFunction(name: "nop")!)
let touchPSO = try dev.makeComputePipelineState(function: lib.makeFunction(name: "touch")!)
let q = dev.makeCommandQueue()!
let buf = dev.makeBuffer(length: 1 << 20, options: [.storageModeShared, .hazardTrackingModeUntracked])!
let bufTracked = dev.makeBuffer(length: 1 << 20, options: [.storageModeShared])!

let N = 400

func run(_ name: String, concurrent: Bool, barrier: Bool, pso: MTLComputePipelineState,
         buffer: MTLBuffer, tgs: Int) {
    var best = Double.infinity
    for _ in 0..<3 {
        let cb = q.makeCommandBuffer()!
        let enc = concurrent
            ? cb.makeComputeCommandEncoder(dispatchType: .concurrent)!
            : cb.makeComputeCommandEncoder()!
        for _ in 0..<N {
            enc.setComputePipelineState(pso)
            enc.setBuffer(buffer, offset: 0, index: 0)
            enc.dispatchThreadgroups(MTLSize(width: tgs, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            if barrier { enc.memoryBarrier(scope: .buffers) }
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let us = (cb.gpuEndTime - cb.gpuStartTime) * 1e6 / Double(N)
        best = min(best, us)
    }
    print(String(format: "%-46s %8.2f µs/dispatch", (name as NSString).utf8String!, best))
}

run("nop serial untracked no-barrier", concurrent: false, barrier: false, pso: nopPSO, buffer: buf, tgs: 1)
run("nop concurrent untracked no-barrier", concurrent: true, barrier: false, pso: nopPSO, buffer: buf, tgs: 1)
run("nop concurrent untracked +barrier", concurrent: true, barrier: true, pso: nopPSO, buffer: buf, tgs: 1)
run("touch(256tg) serial TRACKED no-barrier", concurrent: false, barrier: false, pso: touchPSO, buffer: bufTracked, tgs: 256)
run("touch(256tg) concurrent untracked +barrier", concurrent: true, barrier: true, pso: touchPSO, buffer: buf, tgs: 256)
run("touch(2048tg) concurrent untracked +barrier", concurrent: true, barrier: true, pso: touchPSO, buffer: buf, tgs: 2048)
run("touch(2048tg) concurrent untracked no-barrier", concurrent: true, barrier: false, pso: touchPSO, buffer: buf, tgs: 2048)
