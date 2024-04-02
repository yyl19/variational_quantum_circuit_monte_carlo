

# predefined quantum channels

# see NielsenChuang
AmplitudeDamping(pos::Int; γ::Real) = QuantumMap(pos, [[1 0; 0 sqrt(1-γ)], [0 sqrt(γ); 0 0]])
PhaseDamping(pos::Int; γ::Real) = QuantumMap(pos, [[1 0; 0 sqrt(1-γ)], [0 0; 0 sqrt(γ)]])
Depolarizing(pos::Int; p::Real) = QuantumMap(pos, [sqrt(1-3*p/4) .* I₂, (sqrt(p)/2) .* X, (sqrt(p)/2) .* Y, (sqrt(p)/2) .* Z ])

# function Depolarizing(pos::Tuple{Int,Int};p::Real)
# 	ms=Vector{Matrix{ComplexF64}}()
# 	#push!(ms,sqrt(1-p) .* kron(I₂,I₂))
# 	op_list=(sqrt(p)/4) .*[I₂,X,Y,Z]
# 	for (i,j) in Base.product(op_list,op_list)
# 		push!(ms,kron(i,j))
# 	end
# 	ms[1]=sqrt(1-15*p/16) .* kron(I₂,I₂)
# 	return QuantumMap(pos,ms)
# end

nparameters(x::QuantumMap) = 0

function reset_parameters_util!(x::QuantumMap, p::Vector{<:Real}, pos::Int)
	return pos
end
parameters(x::QuantumMap) = nothing
is_parameters(x::QuantumMap) = nothing