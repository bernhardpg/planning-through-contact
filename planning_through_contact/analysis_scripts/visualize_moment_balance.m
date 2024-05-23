%% Visualize moment balance
clc; clear; close all;

% Quantities defined in the Box frame
BOX_WIDTH = 0.3;
BOX_HEIGHT = 0.2;

MASS = 1; % kg
G = 9.81; % m/s^2
f_g_W = [0; -MASS * G];

theta = (pi / 2) * 0.5;
R_WB = [cos(theta) -sin(theta);
        sin(theta)  cos(theta)];
f_g = R_WB' * f_g_W;

p_m1 = [-BOX_WIDTH/2; BOX_HEIGHT/2];
p_m2 = [BOX_WIDTH/2; BOX_HEIGHT/2];
p_m3 = [BOX_WIDTH/2; -BOX_HEIGHT/2];
p_m4 = [-BOX_WIDTH/2; -BOX_HEIGHT/2];

corners = [p_m1 p_m2 p_m3 p_m4];

p_c1 = p_m4;

syms lam f_c1_x f_c1_y f_c2_x f_c2_y
f_c1 = [f_c1_x; f_c1_y];
f_c2 = [f_c2_x; f_c2_y];
p_c2 = (1 - lam) * p_m2 + lam * p_m3;

cross = @(v1, v2) v1(1) * v2(2) - v1(2) * v2(1);

torque_balance = cross(p_c1, f_c1) + cross(p_c2, f_c2) == 0;
force_balance = f_c1 + f_c2 + f_g == 0;

% Solve force and moment balance to get an expression for f_c2
f_c1_x_as_func = solve(torque_balance, f_c1_x);
force_balance_as_func = subs(force_balance, f_c1_x, f_c1_x_as_func);
temp = solve(force_balance_as_func, f_c2);
f_c2_as_func = [temp.f_c2_x; temp.f_c2_y];

f_c1_y_val = 6;

f_c2_by_lam = subs(f_c2_as_func, f_c1_y, f_c1_y_val);
f_c1_x_as_func = solve(force_balance, f_c1).f_c1_x;

lams = linspace(0,1,100);

for k = 1:numel(lams)
    clf
    p_c2_val = subs(p_c2, lam, lams(k));
    f_c2_val = subs(f_c2_by_lam, lam, lams(k));

    f_c1_x_val = subs(f_c1_x_as_func, f_c2_x, f_c2_val(2));
    f_c1_val = [f_c1_x_val; f_c1_y_val];

    plot(polyshape(corners(1,:), corners(2,:))); hold on
    plot_vector(p_c1, f_c1_val); hold on
    plot_vector(p_c2_val, f_c2_val);
    ylim([-0.35, 0.35])
    xlim([-0.35, 0.35])
    pause(0.01)
end


function plot_vector(pos, vec)
    SCALE = 0.05;
    quiver(pos(1), pos(2), vec(2) * SCALE, vec(2) * SCALE, "LineWidth", 2);
end
