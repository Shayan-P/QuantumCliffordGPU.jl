using QuantumClifford
using CUDA
using QuantumCliffordGPU
using Test

@testset "QuantumCliffordGPU.jl" begin
    @test begin
        p = P"_IZXY"
        p_gpu = to_gpu(p)
        typeof(p_gpu.xz) <: CUDA.CuArray # this is a bad test because it depends on representation of data in QuantumClifford. Change later...
    end
    @test begin
        s = S"-XX
              +ZZ"
        s_gpu = to_gpu(s)
        typeof(tab(s_gpu).xzs) <: CUDA.CuArray # this is a bad test because it depends on representation of data in QuantumClifford. Change later...
    end

    @test begin
        s = to_gpu(S"XX XZ")
        op = SingleQubitOperator(sHadamard(1))
        QuantumCliffordGPU._apply!(s, op) 
        correct = S"ZX ZZ"
        to_cpu(s) == correct
    end

    @test begin
        s = random_stabilizer(500, 500)
        s_gpu = to_gpu(s)
        op = SingleQubitOperator(sHadamard(5))
        QuantumCliffordGPU._apply!(s_gpu, op) 
        QuantumClifford._apply!(s, op)
        to_cpu(s_gpu) == s
    end

end
