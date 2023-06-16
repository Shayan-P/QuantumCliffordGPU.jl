# helper functions to be defined for xzs because in gpu kernel we are not allowed to work with Stabilizer object
# what's the better way not to copy all these functions?
@inline _logsizeof() = 6
@inline _mask() = sizeof(UInt64)*8-1
@inline _div(l) = l >> _logsizeof()
@inline _mod(l) = l & _mask()

@inline getshift(col::Int) = _mod(col-1)
@inline getmask(col::Int) = UInt64(0x1)<<getshift(col)
@inline getbigindex(col::Int) = _div(col-1)+1

Base.@propagate_inbounds function getxbit(xzs, r, c)
    xzs[getbigindex(c),r]&getmask(c)
end
Base.@propagate_inbounds function getzbit(xzs, r, c)
    xzs[end÷2+getbigindex(c),r]&getmask(c)
end
Base.@propagate_inbounds function setxbit(xzs, r, c, x)
    cbig = getbigindex(c)
    xzs[cbig,r] &= ~getmask(c)
    xzs[cbig,r] |= x
end
Base.@propagate_inbounds function setzbit(xzs, r, c, z)
    cbig = getbigindex(c)
    xzs[end÷2+cbig,r] &= ~getmask(c)
    xzs[end÷2+cbig,r] |= z
end

# todo put back the generic types later
# Questions:
# 1- couldn't input tabeulu to gpu kernel
# 2- doesn't support multimodal so I had to write functions one by one
# 3- how to use the getxbit, setxbit functions that are in QuantumClifford? (without having to copy)
# 4- CuArray becomes CuDeviceMatrix in kernel!
function single_qubit_gpu_kernel(xzs::CuDeviceMatrix{UInt64, 1},
                                 phases::CuDeviceVector{UInt8, 1},
                                 op::SingleQubitOperator,
                                 rows::Unsigned)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx > rows
        return nothing
    end
    c = op.q
    r = idx
    Tme = UInt64 #eltype(xzs)
    sh = getshift(c)
    xx,zx,xz,zz = Tme.((op.xx,op.zx,op.xz,op.zz)) .<< sh # maybe do this in parent class?
    anticom = ~iszero((~zz & xz & ~xx & zx) | ( zz & ~xz & xx & zx) | (zz &  xz & xx & ~zx))

    # todo. in future each gpu core can be responsible for multiple rows
    x = getxbit(xzs, r, c)
    z = getzbit(xzs, r, c)
    setxbit(xzs, r, c, (x&xx)⊻(z&zx))
    setzbit(xzs, r, c, (x&xz)⊻(z&zz))

    if op.px && ~iszero(x)
        phases[r] += 0x2
        phases[r] &= 3
    end
    if op.pz && ~iszero(z)
        phases[r] += 0x2
        phases[r] &= 3
    end
    if ~iszero(x&z) && anticom
        phases[r] += 0x2
        phases[r] &= 3
    end
    return nothing
end

function _apply!(stab::QuantumClifford.Stabilizer{QuantumClifford.Tableau{Tz, Tm}},
    op::SingleQubitOperator;
    phases::Val{B}=Val(true)) where {B, Tz<:AbstractArray{<:Unsigned}, Tm<:CuArray{<:Unsigned, 2}}
    # todo how to use phases similar to before in kernel functions??!
    threads_count = 1024 # Change this later
    rows::Unsigned = size(stab, 2)
    blocks_count = ceil(Int, rows/threads_count)
    tab = stab.tab
    CUDA.@sync @cuda threads=threads_count blocks=blocks_count single_qubit_gpu_kernel(tab.xzs, tab.phases, op, rows)
end
