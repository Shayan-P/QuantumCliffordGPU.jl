module QuantumCliffordGPU

using QuantumClifford
using CUDA

include("adapters.jl")
include("apply.jl")

export to_cpu, to_gpu, _apply!
# hide _apply function later. only for internal use
end
