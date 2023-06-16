using QuantumCliffordGPU
using QuantumClifford
using Plots
using BenchmarkTools

# how to do benchmarks in a more sophisticated way?

s = random_stabilizer(500, 500)
s_gpu = to_gpu(s)

s = to_gpu(S"XX XZ")
op = SingleQubitOperator(sHadamard(1))

@benchmark QuantumClifford._apply!($s, $op)

@benchmark QuantumCliffordGPU._apply!($s, $op)
