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

end
