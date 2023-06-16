# maybe change the format of storing the data in gpu array 
# so that it is more convinient to work with them on gpu?
function to_gpu(tab::QuantumClifford.Tableau)
    QuantumClifford.Tableau(CuArray(tab.phases), tab.nqubits, CuArray(tab.xzs))
end

function to_cpu(tab::QuantumClifford.Tableau)
    QuantumClifford.Tableau(Array(tab.phases), tab.nqubits, Matrix(tab.xzs))
end

function to_gpu(pauli::QuantumClifford.PauliOperator) 
    QuantumClifford.PauliOperator(CuArray(pauli.phase), pauli.nqubits, CuArray(pauli.xz))
end

function to_cpu(pauli::QuantumClifford.PauliOperator)
    QuantumClifford.PauliOperator(Array(pauli.phase), pauli.nqubits, Array(pauli.xz))
end

function to_gpu(stabilizer::QuantumClifford.Stabilizer) 
    Stabilizer(to_gpu(tab(stabilizer)))
end

function to_cpu(stabilizer::QuantumClifford.Stabilizer) 
    Stabilizer(to_cpu(tab(stabilizer)))
end
