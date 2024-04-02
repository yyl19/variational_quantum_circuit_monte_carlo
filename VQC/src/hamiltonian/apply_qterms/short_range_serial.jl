

function _apply_gate_2_impl!(key::Int, U::AbstractMatrix, v::AbstractVector, vout::AbstractVector)
    L = length(v)
    pos = 2^(key-1)
    strid = pos * 2
    for i in 0:strid:(L-1)
        @inbounds for j in 0:(pos-1)
            l = i + j + 1
            vi1 = v[l]
            vi2 = v[l + pos]
            @fastmath vo1 = U[1,1] * vi1 + U[1,2] * vi2
            @fastmath vo2 = U[2,1] * vi1 + U[2,2] * vi2
            vout[l] += vo1
            vout[l + pos] += vo2
        end
    end
end

_apply_gate_2!(key::Int, U::AbstractMatrix, v::AbstractVector, vout::AbstractVector) = _apply_gate_2_impl!(
	key, SMatrix{2,2, eltype(v)}(U), v, vout)
_apply_gate_2!(key::Tuple{Int}, U::AbstractMatrix, v::AbstractVector, vout::AbstractVector) = _apply_gate_2!(
    key[1], U, v, vout)


function _apply_gate_2_impl!(key::Tuple{Int, Int}, U::AbstractMatrix, v::AbstractVector, vout::AbstractVector)
    # U = Matrix(transpose(reshape(U, 4, 4)))
    L = length(v)
    q1, q2 = key
    pos1, pos2 = 2^(q1-1), 2^(q2-1)
    stride1, stride2 = 2 * pos1, 2 * pos2
    for i in 0:stride2:(L-1)
        for j in 0:stride1:(pos2-1)
            @inbounds for k in 0:(pos1-1)
                l = i + j + k + 1
                vi = SVector(v[l], v[l + pos1], v[l + pos2], v[l + pos1 + pos2])
                @fastmath vo = U * vi

                vout[l] += vo[1] 
                vout[l + pos1] += vo[2] 
                vout[l + pos2] += vo[3] 
                vout[l + pos1 + pos2] += vo[4]
            end
        end
    end
end

_apply_gate_2!(key::Tuple{Int, Int}, U::AbstractMatrix, v::AbstractVector, vout::AbstractVector) =  _apply_gate_2_impl!(
key, SMatrix{4,4, eltype(v)}(U), v, vout)

function _apply_gate_2_impl!(key::Tuple{Int, Int, Int}, m::AbstractMatrix, v::AbstractVector, vout::AbstractVector)
    L = length(v)
    q1, q2, q3 = key
    pos1, pos2, pos3 = 2^(q1-1), 2^(q2-1), 2^(q3-1)
    stride1, stride2, stride3 = 2 * pos1, 2 * pos2, 2 * pos3
    for h in 0:stride3:(L-1)
        for i in 0:stride2:(pos3-1)
            for j in 0:stride1:(pos2-1)
                @inbounds for k in 0:(pos1-1)
                    l000 = h + i + j + k + 1
                    l100 = l000 + pos1
                    l010 = l000 + pos2
                    l110 = l010 + pos1

                    l001 = l000 + pos3
                    l101 = l001 + pos1
                    l011 = l001 + pos2
                    l111 = l011 + pos1
                    vi = SVector(v[l000], v[l100], v[l010], v[l110], v[l001], v[l101], v[l011], v[l111])
                    
                    @fastmath begin
                        vo = m * vi
                        vout[l000] += vo[1] 
                        vout[l100] += vo[2] 
                        vout[l010] += vo[3] 
                        vout[l110] += vo[4] 
                        vout[l001] += vo[5] 
                        vout[l101] += vo[6] 
                        vout[l011] += vo[7] 
                        vout[l111] += vo[8]
                    end 
                end
            end
        end
    end
end

_apply_gate_2!(key::Tuple{Int, Int, Int}, U::AbstractMatrix, v::AbstractVector, vout::AbstractVector) = _apply_gate_2_impl!(
key, SMatrix{8,8, eltype(v)}(U), v, vout)

function _apply_serial_util!(m::QubitsOperator, v::AbstractVector, vout::AbstractVector)
    for (k, bond) in m.data
        _apply_gate_2!(k, _get_mat(length(k), bond), v, vout)
    end
    return vout
end


