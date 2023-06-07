using QuantumClifford
using QuantumCliffordGPU
using CUDA
using Test

@testset "QuantumCliffordGPU.jl" begin
    @test begin
        p = P"_IZXY"
        typeof(p.xz) <: CUDA.CuArray # this is a bad test because it depends on representation of data in QuantumClifford. Change later...
    end
end
