abstract type EncodeGate{N}<:Gate{N} end


#RyGate as encode Gate, position, batchnumber
mutable struct ERyGate <: EncodeGate{1}
	pos::Tuple{Int}
	paras::Vector{Float64}
	batch::Tuple{Int}
end
ERyGate(pos::Int, p::Vector{<:Real},batch::Int) = ERyGate((pos,), convert.(Float64, p), (batch,))

mat(x::ERyGate) = [Ry(x.paras[i]) for i in 1:x.batch[1]]
change_positions(s::ERyGate, m::AbstractDict) = ERyGate((m[s.pos[1]],), s.paras, s.batch)
Base.eltype(x::ERyGate) = eltype(Ry(x.paras[1]))

Base.adjoint(x::ERyGate) = ERyGate(x.pos,-x.paras,x.batch)