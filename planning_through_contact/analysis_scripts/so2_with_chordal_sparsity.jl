using COSMO, JuMP, LinearAlgebra

NUM_CTRL_POINTS = 8
NUM_DIMS = 2

n = NUM_CTRL_POINTS * NUM_DIMS + 1 # we also include 1

# Cost
function create_Q_i(i)
    Q_i = zeros(n, n);
    temp =  [I(NUM_DIMS) -I(NUM_DIMS);
             -I(NUM_DIMS) I(NUM_DIMS)];
    start_index = i * NUM_DIMS;
    Q_i[start_index:start_index+NUM_DIMS*2-1, start_index:start_index+NUM_DIMS*2-1] .= temp;
    return Q_i
end

# Unit vectors
function e_i(i)
    v = zeros(n)
    v[i] = 1
    return v
end

# Create problem data
Q = sum([create_Q_i(i) for i ∈ 1:NUM_CTRL_POINTS-1]);

ϵ = 1e-7
θ_0 = 0;
θ_T = π - ϵ;

r_0 = [cos(θ_0); sin(θ_0)];
r_T = [cos(θ_T); sin(θ_T)];

A_0 = zeros(NUM_DIMS,n-1);
A_0[1,1] = 1
A_0[2,2] = 1

A_T = zeros(NUM_DIMS,n-1);
A_T[1,end-1] = 1
A_T[2,end] = 1

A_eq = [-r_0 A_0;
        -r_T A_T];

# Formulate the problem

model = JuMP.Model(COSMO.Optimizer);
set_optimizer_attribute(model, "decompose", true)
set_optimizer_attribute(model, "merge_strategy", COSMO.NoMerge)

@variable(model, X[1:n, 1:n], PSD)
@constraint(model, X[1,1] == 1)

@objective(model, Min, tr(Q * X'))

@constraint(model, A_eq * X .== 0)

## SO(2) constraints 
for i ∈ 1:2:n-1
    Q_eq = -e_i(i+1) * e_i(i+1)' - e_i(i+2) * e_i(i+2)';
    Q_eq[1,1] = 1;
    @constraint(model, tr(Q_eq * X') == 0)
end

JuMP.optimize!(model);

status = JuMP.termination_status(model)
X_sol = JuMP.value.(X)
obj_value = JuMP.objective_value(model)

x_val = value.(x)[2:end]

pts = reshape(x_val, (NUM_DIMS, NUM_CTRL_POINTS))'

using Plots
# Plot the trajectory
plot(pts[:,1], pts[:,2], label="", ylims=(-1.5, 1.5), xlims=(-1.5, 1.5), aspect_ratio=1)
scatter!(pts[:,1], pts[:,2], label="Ctrl points")

ϵ = 0.1
for i ∈ 1:NUM_CTRL_POINTS
    annotate!.(
        pts[i,1] + ϵ , pts[i,2] + ϵ,
        string(i)
    )
end

# Plot the unit circle
θs = 0:0.01:2*π
plot!(cos.(θs), sin.(θs), color=:grey, label="")


