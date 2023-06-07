import QuantumClifford

using CUDA


QuantumClifford.PauliOperator(phase::UInt8, x::AbstractVector{Bool}, z::AbstractVector{Bool}) = begin
    phase = fill(UInt8(phase),())
    nqubits = length(x)
    xz = vcat(reinterpret(UInt,BitVector(x).chunks),
                reinterpret(UInt,BitVector(z).chunks))
    xz = CuArray(xz)
    QuantumClifford.PauliOperator(phase, nqubits, xz)
end
