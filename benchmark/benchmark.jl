using QuantumCliffordGPU
using QuantumClifford
using Plots
using BenchmarkTools
using NVTX
using CUDA

# how to do benchmarks in a more sophisticated way?

function meaningless_random_stabilizer(k, row=k) # generates a meaningless stabilizer with k * k size pauli operators by default. 
    plist = PauliOperator{Array{UInt8, 0}, Vector{UInt64}}[]
    for i in 1:row
        push!(plist, random_pauli(k, realphase = false))
    end
    s = Stabilizer(plist)
    return s
end

N = 20000

s = meaningless_random_stabilizer(N)
s_gpu = to_gpu(s)
op = SingleQubitOperator(sHadamard(40))

@benchmark QuantumClifford._apply!($s, $op)

@benchmark QuantumCliffordGPU._apply!($s_gpu, $op)

function main()
    function multiple_calls(s_gpu)
        for i in 1:10
            my_op = SingleQubitOperator(sHadamard(i))
            NVTX.@range "hadamard $i" QuantumCliffordGPU._apply!(s_gpu, my_op)
        end
    end
    CUDA.@profile begin
        NVTX.@range "first apply" QuantumCliffordGPU._apply!(s_gpu, op)
        NVTX.@range "second apply" QuantumCliffordGPU._apply!(s_gpu, op)
        NVTX.@range "first multiple apply" multiple_calls(s_gpu)
        NVTX.@range "second multiple apply" multiple_calls(s_gpu)
    end
end
