using CUDA
using NVTX

N = 20000
i = 102
j = 1234
rand_matrix = CUDA.rand(N, N)

function row_swap()
    copy = rand_matrix[i, :]
    rand_matrix[i, :] = rand_matrix[j, :]
    rand_matrix[j, :] = copy
end

function column_swap()
    copy = rand_matrix[:, i]
    rand_matrix[:, i] = rand_matrix[:, j]
    rand_matrix[:, j] = copy
end

function row_swap_kernel(matrix, i, j, N)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    if idx > N
        return nothing
    end
    cp = matrix[i, idx]
    matrix[i, idx] = matrix[j, idx]
    matrix[j, idx] = cp
    return nothing
end

function comparison_main()
    function wrap_kernel()
        CUDA.@sync @cuda threads=1024 blocks=ceil(Int, N/1024) row_swap_kernel(rand_matrix, i, j, N)
    end

    CUDA.@profile begin
        NVTX.@range "first row_swap" row_swap()
        NVTX.@range "second row_swap" row_swap()
        NVTX.@range "first column_swap" column_swap()
        NVTX.@range "second column_swap" column_swap()
        NVTX.@range "first wrap_kernel" wrap_kernel()
        NVTX.@range "second wrap_kernel" wrap_kernel()
    end
end