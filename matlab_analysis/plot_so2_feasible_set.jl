using JuMP, Mosek, MosekTools
using Plots
using Graphs
using GraphPlot, Compose
using LinearAlgebra

NUM_DIMS = 2
n = NUM_DIMS + 1 # we also include 1

# Unit vectors
function e_i(i)
    v = zeros(n)
    v[i] = 1
    return v
end

function get_sol(θ)
    model = Model(Mosek.Optimizer)

    @variable(model, X[1:n, 1:n], PSD)

    x = X[2:end, 1]
    @constraint(model, X[1,1] == 1)

    Q_eq = -e_i(2) * e_i(2)' - e_i(3) * e_i(3)'
    Q_eq[1,1] = 1
    @show(Q_eq)
    @constraint(model, tr(Q_eq * X') == 0)

    lin_cost = [cos(θ); sin(θ)]
    @objective(model, Min, lin_cost' * x)

    # Solve the problem
    optimize!(model)
    x_val = value.(X)[2:end,1]
    return x_val
end

θs = 0:0.01:2*π

extreme_points = vcat([get_sol(θ)' for θ ∈ θs]...)

plot(extreme_points[:,1], extreme_points[:,2], fillrange = 0, fillalpha = 0.35, c = 1, aspect_ratio=1, label="Projection of SDP unit-circle constraint onto R^2")

