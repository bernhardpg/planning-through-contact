using JuMP, Mosek, MosekTools
using Plots
using Graphs
using GraphPlot, Compose
using LinearAlgebra

NUM_DIMS = 2
NUM_CTRL_POINTS = 3
n = NUM_CTRL_POINTS * NUM_DIMS + 1 # we also include 1

θ_0 = 0;
θ_T = π / 2;

r_0 = [cos(θ_0); sin(θ_0)];
r_T = [cos(θ_T); sin(θ_T)];

# Unit vectors
function e_i(i)
    v = zeros(n)
    v[i] = 1
    return v
end

function get_sol(θ)
    model = Model(Mosek.Optimizer)

    @variable(model, X[1:n, 1:n], PSD)

    @constraint(model, X[1,1] == 1)
    x = X[2:end, 1]

    ## SO(2) constraints 
    for i ∈ 1:2:n-1
        Q_eq = -e_i(i+1) * e_i(i+1)' - e_i(i+2) * e_i(i+2)';
        Q_eq[1,1] = 1;
        @constraint(model, tr(Q_eq * X') == 0)
    end

    # Initial and final condition
    A_0 = zeros(NUM_DIMS,n-1);
    A_0[1,1] = 1
    A_0[2,2] = 1
    A_T = zeros(NUM_DIMS,n-1);
    A_T[1,end-1] = 1
    A_T[2,end] = 1
    A_eq = [-r_0 A_0;
            -r_T A_T];
    @constraint(model, A_eq * X .== 0)

    # Cost
    lin_cost = [cos(θ); sin(θ)]
    second_control_point = X[4:5,1]
    @objective(model, Min, lin_cost' * second_control_point)

    # Solve the problem
    optimize!(model)

    if termination_status(model) != MOI.OPTIMAL
        print("No solution found")
        return
    end

    second_control_point_val = value.(second_control_point)
    return second_control_point_val
end

θs = 0:0.01:2*π

extreme_points = vcat([get_sol(θ)' for θ ∈ θs]...)

plot(extreme_points[:,1], extreme_points[:,2], fillrange = 0, fillalpha = 0.35, c = 1, aspect_ratio=1, label="Projection of SDP unit-circle constraint onto R^2")

