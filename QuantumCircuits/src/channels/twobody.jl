

# predefined quantum channels

# see NielsenChuang
# AmplitudeDamping(pos::Int; γ::Real) = QuantumMap(pos, [[1 0; 0 sqrt(1-γ)], [0 sqrt(γ); 0 0]])
# PhaseDamping(pos::Int; γ::Real) = QuantumMap(pos, [[1 0; 0 sqrt(1-γ)], [0 0; 0 sqrt(γ)]])
# Depolarizing(pos::Tuple{Int,Int}; p::Real) = QuantumMap(pos, [sqrt(1-15*p/16) .* kron(I₂,I₂)])

function Depolarizing(pos::Tuple{Int,Int};p::Real)
	ms=[sqrt(1-p) .* kron(I₂,I₂)]
	op_list=(sqrt(p)/4) .*[I₂,X,Y,Z]
	for i,j in Base.product(op_list,op_list)
		push!(ms,kron(i,j))
	end
	return QuantumMap(pos,ms)
end