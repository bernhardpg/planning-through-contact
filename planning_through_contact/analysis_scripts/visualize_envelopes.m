sdpvar x;
xL = -1;
xU = 2;
constraints =[(xU-x)*(x-xL)>=0,(xU-x)*(xU-x)>=0,(x-xL)*(x-xL)>=0]

plot(constraints,[x;x^2],[],[],sdpsettings('relax',1))
t = (-1:0.01:2);
hold on;plot(t,t.^2);

%%

sdpvar x;
xL = 0;
xU = 1;
sdpvar y;
yL = 0;
yU = 1;
sdpvar w;

constraints = [...
                (xU-x)*(x-xL)>=0, ...
                (xU-x)*(xU-x)>=0, ...
                (x-xL)*(x-xL)>=0, ...
                (yU-y)*(y-yL)>=0, ...
                (yU-y)*(yU-y)>=0, ...
                (y-yL)*(y-yL)>=0, ...
               ];
w_constraints = [x >= xL, x <= xU, y >= yL, y <= yU, w == x^2 + y^2 - 1,w == 0];

plot([constraints, w_constraints],[x; y; w],[],[],sdpsettings('relax',1));
hold on;
plot(w_constraints,[x; y; w],[],[])
tx = (xL:0.01:xU);
ty = (yL:0.01:yU);
%hold on;
%plot(tx, tx.^2 + ty.^2 - 1, 'g');
%plot3(tx, ty, tx.^2 + ty.^2 - 1, 'g');

%%

%%

sdpvar x;
xL = 0;
xU = 1;
sdpvar y;
yL = 0;
yU = 1;
sdpvar w;

constraints = [...
                (xU-x)*(x-xL)>=0, ...
                (xU-x)*(xU-x)>=0, ...
                (x-xL)*(x-xL)>=0, ...
                (yU-y)*(y-yL)>=0, ...
                (yU-y)*(yU-y)>=0, ...
                (y-yL)*(y-yL)>=0, ...
               ];
w_constraints = [w == x^2 + y^2 - 1, w == 0];

plot([constraints, w_constraints],[x; y; w],[],[],sdpsettings('relax',1)); hold on
fimplicit(@(x,y) x.^2 + y.^2 - 1, [0 1 0 1])
ylim([0 1])
