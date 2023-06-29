const InnerGPUType = UInt64;

# helper functions to be defined for xzs because in gpu kernel we are not allowed to work with Stabilizer object
# what's the better way not to copy all these functions?
@inline _logsizeof() = 6
@inline _mask() = sizeof(UInt64)*8-1
@inline _div(l) = l >> _logsizeof()
@inline _mod(l) = l & _mask()

@inline getshift(col::Int) = _mod(col-1)
@inline getmask(col::Int) = UInt64(0x1)<<getshift(col)
@inline getbigindex(col::Int) = _div(col-1)+1

Base.@propagate_inbounds function getxbit(xzs, r::Integer, c::Integer)::InnerGPUType
    xzs[getbigindex(c),r]&getmask(c)
end
Base.@propagate_inbounds function getzbit(xzs, r::Integer, c::Integer)::InnerGPUType
    xzs[end÷2+getbigindex(c),r]&getmask(c)
end
Base.@propagate_inbounds function setxbit(xzs, r::Integer, c::Integer, x::InnerGPUType)
    cbig = getbigindex(c)
    xzs[cbig,r] &= ~getmask(c)
    xzs[cbig,r] |= x
end
Base.@propagate_inbounds function setzbit(xzs, r::Integer, c::Integer, z::InnerGPUType)
    cbig = getbigindex(c)
    xzs[end÷2+cbig,r] &= ~getmask(c)
    xzs[end÷2+cbig,r] |= z
end
Base.@propagate_inbounds setxbit(xzs, r::Integer, c::Integer, x::InnerGPUType, shift::Integer) = setxbit(xzs, r, c, x<<shift)
Base.@propagate_inbounds setzbit(xzs, r::Integer, c::Integer, z::InnerGPUType, shift::Integer) = setzbit(xzs, r, c, z<<shift)

# todo put back the generic types later
# Questions:
# 1- couldn't input tabeulu to gpu kernel
# 2- doesn't support multimodal so I had to write functions one by one
# 3- how to use the getxbit, setxbit functions that are in QuantumClifford? (without having to copy)
# 4- CuArray becomes CuDeviceMatrix in kernel!
function single_qubit_gpu_kernel(xzs::CuDeviceMatrix{UInt64, 1},
                                 phases::CuDeviceVector{UInt8, 1},
                                 op::SingleQubitOperator,
                                 rows::Unsigned,
                                 compute_phases::Bool=true)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx > rows
        return nothing
    end
    c = op.q
    r = idx
    sh = getshift(c)
    xx,zx,xz,zz = InnerGPUType.((op.xx,op.zx,op.xz,op.zz)) .<< sh # maybe do this in parent class?
    anticom = ~iszero((~zz & xz & ~xx & zx) | ( zz & ~xz & xx & zx) | (zz &  xz & xx & ~zx))

    # todo. in future each gpu core can be responsible for multiple rows
    x = getxbit(xzs, r, c)
    z = getzbit(xzs, r, c)
    setxbit(xzs, r, c, (x&xx)⊻(z&zx))
    setzbit(xzs, r, c, (x&xz)⊻(z&zz))

    if compute_phases
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
    end
    return nothing
end

function _apply!(stab::QuantumClifford.Stabilizer{QuantumClifford.Tableau{Tz, Tm}},
    op::QuantumClifford.SingleQubitOperator;
    compute_phases::Bool=true) where {Tz<:CuArray{<:Unsigned, 1}, Tm<:CuArray{<:Unsigned, 2}}
    # todo how to use phases similar to before in kernel functions??!
    threads_count = 1024 # Change this later
    rows::Unsigned = size(stab, 2)
    blocks_count = ceil(Int, rows/threads_count)
    tab = QuantumClifford.tab(stab)
    # todo. why can't I pass compute_phases=compute_phases normally without function call?
    CUDA.@sync @cuda threads=threads_count blocks=blocks_count single_qubit_gpu_kernel(tab.xzs, tab.phases, op, rows, compute_phases)
end

function two_qubit_gpu_kernel(xzs::CuDeviceMatrix{UInt64, 1},
                              phases::CuDeviceVector{UInt8, 1},
                              gate::QuantumClifford.AbstractTwoQubitOperator,
                              rows::Unsigned,
                              compute_phases::Bool=true)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx > rows
        return nothing
    end

    q1 = gate.q1
    q2 = gate.q2
    shift = getshift(q1) - getshift(q2)
    r = idx;

    _x1::InnerGPUType = getxbit(xzs, r, q1)
    _z1::InnerGPUType = getzbit(xzs, r, q1)
    _x2::InnerGPUType = getxbit(xzs, r, q2)<<shift
    _z2::InnerGPUType = getzbit(xzs, r, q2)<<shift
    x1::InnerGPUType,z1::InnerGPUType,x2::InnerGPUType,z2::InnerGPUType,phase::Bool = QuantumClifford.qubit_kernel(gate,_x1,_z1,_x2,_z2) # Most `qubit_kernel` functions are defined by a `qubitop2` macro
    setxbit(xzs, r, q1, x1, 0)
    setzbit(xzs, r, q1, z1, 0)
    setxbit(xzs, r, q2, x2, -shift)
    setzbit(xzs, r, q2, z2, -shift)
    if compute_phases && phase
        phases[r] += 0x2
        phases[r] &= 3
    end
    return nothing
end


function _apply!(stab::QuantumClifford.Stabilizer, 
                 gate::G; 
                 compute_phases::Bool=true) where {G<:QuantumClifford.AbstractTwoQubitOperator}
    threads_count = 1024 # Change this later
    rows::Unsigned = size(stab, 2)
    blocks_count = ceil(Int, rows/threads_count)
    tab = QuantumClifford.tab(stab)
    # todo. why can't I pass compute_phases=compute_phases normally without function call?
    CUDA.@sync @cuda threads=threads_count blocks=blocks_count two_qubit_gpu_kernel(tab.xzs, tab.phases, gate, rows, compute_phases)

    # todo dry this out...!
end
