
@adjoint *(circuit::QCircuit, x::Union{StateVector,StateVectorBatch,DensityMatrix, DensityMatrixBatch, CuStateVectorBatch, CuDensityMatrixBatch}) = begin
    y = circuit * x
    return y, Δ -> begin
        Δ, grads, y = back_propagate(copy(Δ), circuit, copy(y))
        return grads, Δ
    end
end

@adjoint apply!(circuit::QCircuit, x::Union{StateVector, StateVectorBatch, DensityMatrix, DensityMatrixBatch, CuStateVectorBatch, CuDensityMatrixBatch}) = begin
    y = circuit * x
    return y, Δ -> begin
        Δ, grads, y = back_propagate(copy(Δ), circuit, copy(y))
        return grads, Δ
    end
end


function back_propagate(Δ::AbstractVecOrMat, circuit::QCircuit, y::Union{StateVector,StateVectorBatch,DensityMatrix,DensityMatrixBatch})
    RT = real(eltype(y))
    grads = Vector{RT}[]
    for item in reverse(circuit)
        Δ, ∇θs, y = back_propagate(Δ, item, y)
        !isnothing(∇θs) && push!(grads, ∇θs)
    end

    ∇θs_all = RT[]
    for item in Iterators.reverse(grads)
        append!(∇θs_all, item)
    end
    return Δ, ∇θs_all , y
end

function back_propagate(Δ::CuArray, circuit::QCircuit, y::Union{CuStateVectorBatch, CuDensityMatrixBatch})
    RT = real(eltype(y))
    grads = Vector{RT}[]
    #println(typeof(Δ))
    for item in reverse(circuit)
        #println(typeof(Δ))
        Δ, ∇θs, y = back_propagate(Δ, item, y)
        !isnothing(∇θs) && push!(grads, ∇θs)
    end

    ∇θs_all = RT[]
    for item in Iterators.reverse(grads)
        append!(∇θs_all, item)
    end
    return Δ, ∇θs_all , y
end
 
function back_propagate(Δ::AbstractVector, m::Gate, y::StateVector)
    Δ = StateVector(Δ, nqubits(y))
    Δ = apply!(m', Δ)
    y = apply!(m', y)
    ∇θs = nothing
    if nparameters(m) > 0
        ∇θs = [real(expectation(y, item, Δ)) for item in differentiate(m)]
    end
    return storage(Δ), ∇θs, y
end


function back_propagate(Δ::AbstractVector, m::Union{RxGate,RzGate,RyGate}, y::StateVector)
    Δ = StateVector(Δ, nqubits(y))
    Δ = apply!(m', Δ)
    y = apply!(m', y)
    ∇θs = nothing
    if nparameters(m) > 0
        ∇θs = [-imag(expectation(y, item, Δ))/2 for item in differentiate(m)]
    end
    return storage(Δ), ∇θs, y
end

function back_propagate(Δ::AbstractVector, m::Gate, y::StateVectorBatch)
    Δ = StateVectorBatch(Δ, nqubits(y), nitems(y))
    Δ = apply!(m', Δ)
    y = apply!(m', y)
    ∇θs = nothing
    if nparameters(m) > 0
        ∇θs = [real(expectation(y, item, Δ)) for item in differentiate(m)]
    end
    return storage(Δ), ∇θs, y
end

function back_propagate(Δ::AbstractVector, m::Union{RxGate,RzGate,RyGate}, y::StateVectorBatch)
    Δ = StateVectorBatch(Δ, nqubits(y), nitems(y))
    Δ = apply!(m', Δ)
    y = apply!(m', y)
    ∇θs = nothing
    if nparameters(m) > 0
        ∇θs = [-imag(expectation(y, item, Δ))/2 for item in differentiate(m)]
    end
    return storage(Δ), ∇θs, y
end

function back_propagate(Δ::CuVector, m::Gate, y::CuStateVectorBatch)
    Δ = CuStateVectorBatch(Δ, nqubits(y), nitems(y))
    Δ = apply!(m', Δ)
    y = apply!(m', y)
    ∇θs = nothing
    if nparameters(m) > 0
        ∇θs = [real(expectation(y, item, Δ)) for item in differentiate(m)]
    end
    return storage(Δ), ∇θs, y
end


function back_propagate(Δ::CuVector, m::Union{RxGate,RzGate,RyGate}, y::CuStateVectorBatch)
    Δ = CuStateVectorBatch(Δ, nqubits(y), nitems(y))
    Δ = apply!(m', Δ)
    y = apply!(m', y)
    ∇θs = nothing
    if nparameters(m) > 0
        ∇θs = [-imag(expectation(y, item, Δ))/2 for item in differentiate(m)]
    end
    return storage(Δ), ∇θs, y
end

function back_propagate(Δ::AbstractMatrix, m::Gate, y::DensityMatrix)
    Δ = DensityMatrix(Δ, nqubits(y))
    Δ = apply!(m', Δ)
    y = apply!(m', y)
    ∇θs = nothing
    if nparameters(m) > 0
        ∇θs = [2*real(expectation(y, item, Δ)) for item in differentiate(m)]
    end
    return storage(Δ), ∇θs, y
end

function back_propagate(Δ::AbstractMatrix, m::Union{RxGate,RzGate,RyGate}, y::DensityMatrix)
    Δ = DensityMatrix(Δ, nqubits(y))
    Δ = apply!(m', Δ)
    y = apply!(m', y)
    ∇θs = nothing
    if nparameters(m) > 0
        ∇θs = [-imag(expectation(y, item, Δ)) for item in differentiate(m)]
    end
    return storage(Δ), ∇θs, y
end

function back_propagate(Δ::AbstractMatrix, m::Gate, y::DensityMatrixBatch)
    Δ = DensityMatrixBatch(Δ, nqubits(y), nitems(y))
    Δ = apply!(m', Δ)
    y = apply!(m', y)
    ∇θs = nothing
    if nparameters(m) > 0
        ∇θs = [2*real(expectation(y, item, Δ)) for item in differentiate(m)]
    end
    return storage(Δ), ∇θs, y
end

function back_propagate(Δ::AbstractMatrix, m::Union{RxGate,RzGate,RyGate}, y::DensityMatrixBatch)
    Δ = DensityMatrixBatch(Δ, nqubits(y), nitems(y))
    Δ = apply!(m', Δ)
    y = apply!(m', y)
    ∇θs = nothing
    if nparameters(m) > 0
        ∇θs = [-imag(expectation(y, item, Δ)) for item in differentiate(m)]
    end
    return storage(Δ), ∇θs, y
end

function back_propagate(Δ::CuMatrix, m::Gate, y::CuDensityMatrixBatch)
    Δ = CuDensityMatrixBatch(Δ, nqubits(y), nitems(y))
    Δ = apply!(m', Δ)
    y = apply!(m', y)
    ∇θs = nothing
    if nparameters(m) > 0
        ∇θs = [2*real(expectation(y, item, Δ)) for item in differentiate(m)]
    end
    return storage(Δ), ∇θs, y
end

function back_propagate(Δ::CuMatrix, m::Union{RxGate,RzGate,RyGate}, y::CuDensityMatrixBatch)
    Δ = CuDensityMatrixBatch(Δ, nqubits(y), nitems(y))
    Δ = apply!(m', Δ)
    y = apply!(m', y)
    ∇θs = nothing
    if nparameters(m) > 0
        ∇θs = [-imag(expectation(y, item, Δ)) for item in differentiate(m)]
    end
    return storage(Δ), ∇θs, y
end

function back_propagate(Δ::AbstractMatrix, m::QuantumMap, y::DensityMatrix)
    Δ = DensityMatrix(Δ, nqubits(y))
    Δ = apply_dagger!(m, Δ)
    y = apply_inverse!(m, y) 
    ∇θs = nothing
    return storage(Δ), ∇θs, y
end

function back_propagate(Δ::AbstractMatrix, m::QuantumMap, y::DensityMatrixBatch)
    Δ = DensityMatrixBatch(Δ, nqubits(y),nitems(y))
    Δ = apply_dagger!(m, Δ)
    y = apply_inverse!(m, y) 
    ∇θs = nothing
    return storage(Δ), ∇θs, y
end

function back_propagate(Δ::CuMatrix, m::QuantumMap, y::CuDensityMatrixBatch)
    Δ = CuDensityMatrixBatch(Δ, nqubits(y),nitems(y))
    Δ = apply_dagger!(m, Δ)
    y = apply_inverse!(m, y) 
    ∇θs = nothing
    return storage(Δ), ∇θs, y
end

function expectation(y::DensityMatrix,G::Gate,Δ::DensityMatrix)
    Δ1=copy(Δ)
    _dm_apply_threaded_left!(G,Δ1.data)
    return dot(storage(y)',storage(Δ1))
end

function expectation(y::DensityMatrixBatch,G::Gate,Δ::DensityMatrixBatch)
    Δ1=copy(Δ)
    _dm_apply_threaded_left!(G, Δ1.data, nqubits(y))
    return dot(y.data,Δ1.data)
end

function expectation(y::CuDensityMatrixBatch,G::Gate,Δ::CuDensityMatrixBatch)
    Δ1=copy(Δ)
    _dm_apply_threaded_left!(G, Δ1.data, nqubits(y))
    return dot(y.data,Δ1.data)
end

function expectation(y::StateVectorBatch,G::Gate,Δ::StateVectorBatch)
    Δ1=copy(Δ)
    apply_threaded!(G, Δ1.data, nqubits(Δ1))
    return dot(storage(y),storage(Δ1))
end

function expectation(y::CuStateVectorBatch,G::Gate,Δ::CuStateVectorBatch)
    Δ1=copy(Δ)
    apply_threaded!(G, Δ1.data, nqubits(Δ1))
    return dot(storage(y),storage(Δ1))
end

@adjoint qubit_encoding(::Type{T}, mpsstr::Vector{<:Real}) where {T<:Number} = begin
    y = qubit_encoding(T, mpsstr)
    return y, Δ -> begin
        circuit = QCircuit([RyGate(i, theta*pi, isparas=true) for (i, theta) in enumerate(mpsstr)])
        Δ, grads, y = back_propagate(Δ, circuit, copy(y))
        return nothing, grads .* pi
    end
end