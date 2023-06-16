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

function QuantumClifford.Tableau(paulis::AbstractVector{QuantumClifford.PauliOperator{Tz,Tv}}) where {Tz<:AbstractArray{UInt8,0},Tve<:Unsigned,Tv<:CuArray{Tve}}
    r = length(paulis)
    n = QuantumClifford.nqubits(paulis[1])
    tab = QuantumClifford.zero(QuantumClifford.Tableau{Vector{UInt8}, CuArray{Tve, 2}},r,n)
    for i in eachindex(paulis)
        tab[i] = paulis[i]
    end
    tab
end

QuantumClifford.Tableau(phases::AbstractVector{UInt8}, xs::AbstractMatrix{Bool}, zs::AbstractMatrix{Bool}) = Tableau(
    phases, size(xs,2),
    CuArray(vcat(hcat((BitArray(xs[i,:]).chunks for i in 1:size(xs,1))...)::Matrix{UInt},
         hcat((BitArray(zs[i,:]).chunks for i in 1:size(zs,1))...)::Matrix{UInt})) # type assertions to help Julia infer types
)


function Base.zero(::Type{QuantumClifford.Tableau{Tzv, Tm}}, r, q) where {Tzv,T<:Unsigned,Tm<:CuArray{T}}
    phases = zeros(UInt8,r) # todo. also make this CuArray
    xzs = zeros(UInt, QuantumClifford._nchunks(q,T), r)
    xzs = CuArray(xzs)
    QuantumClifford.Tableau(phases, q, xzs)::QuantumClifford.Tableau{Vector{UInt8},<:CuArray{<:Unsigned, 2}}
end


function Base.setindex!(tab::QuantumClifford.Tableau{<:AbstractArray, <:CuArray}, 
                        pauli::QuantumClifford.PauliOperator{<:AbstractArray, <:CuArray}, 
                        i)
    CUDA.@allowscalar tab.phases[i] = pauli.phase[]
    #tab.xzs[:,i] = pauli.xz # TODO why is this assigment causing allocations
    for j in 1:length(pauli.xz)
        CUDA.@allowscalar tab.xzs[j,i] = pauli.xz[j] # todo make this vectorized
    end
    tab
end
