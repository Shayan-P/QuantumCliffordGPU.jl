module QuantumCliffordGPU

using QuantumClifford
using CUDA

include("adapters.jl")
include("apply.jl")
include("pauli_frames.jl");

export to_cpu, to_gpu, _apply!, pftrajectories
# todo hide _apply function later. only for internal use
end
