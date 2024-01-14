from typing import List, Optional

import numpy as np
import numpy.typing as npt
from pydrake.math import eq, le
from pydrake.solvers import (
    Binding,
    BoundingBoxConstraint,
    LinearConstraint,
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
)

from planning_through_contact.convex_relaxation.sdp import (
    linear_bindings_to_homogenuous_form,
)
from planning_through_contact.tools.types import NpExpressionArray, NpVariableArray


class BandSparseSemidefiniteRelaxation:
    def __init__(self, num_groups: int) -> None:
        self.prog = MathematicalProgram()
        self.relaxed_prog = None
        self.Xs = None
        self.num_groups = num_groups

        self.groups = {idx: [] for idx in range(num_groups)}
        self.linear_costs = {idx: [] for idx in range(num_groups)}
        self.linear_inequality_constraints = {idx: [] for idx in range(num_groups)}
        self.linear_equality_constraints = {idx: [] for idx in range(num_groups)}

        # Quadratic constraints and costs is what couples the groups
        self.quadratic_constraints = {
            (i, j): [] for i, j in zip(range(num_groups - 1), range(1, num_groups))
        }
        self.quadratic_costs = {
            (i, j): [] for i, j in zip(range(num_groups - 1), range(1, num_groups))
        }

        # Constraints that will just be added directly to SDP and will not be multiplied
        self.independent_constraints = []

        # Costs that will just be added directly to SDP
        self.independent_costs = []
        self.l2_norm_costs = []

        # Include quadratic terms
        for i in range(num_groups):
            self.quadratic_constraints[i, i] = []
            self.quadratic_costs[i, i] = []

    def _find_numbered_idx(self, idx: int) -> int:
        assert idx < self.num_groups
        if idx == -1:
            return self.num_groups - 1
        else:
            return idx

    def new_variables(self, group_idx: int, *args):
        group_idx = self._find_numbered_idx(group_idx)
        vars = self.prog.NewContinuousVariables(*args)
        self.groups[group_idx].append(vars)

        return vars

    def add_linear_cost(self, group_idx: int, *args):
        group_idx = self._find_numbered_idx(group_idx)
        constraint = self.prog.AddLinearCost(*args)
        self.linear_costs[group_idx].append(constraint)

        return constraint

    def add_bounding_box_constraint(self, group_idx: int, *args):
        group_idx = self._find_numbered_idx(group_idx)
        constraint = self.prog.AddBoundingBoxConstraint(*args)
        self.linear_inequality_constraints[group_idx].append(constraint)

        return constraint

    def add_linear_inequality_constraint(self, group_idx: int, *args):
        group_idx = self._find_numbered_idx(group_idx)
        constraint = self.prog.AddLinearConstraint(*args)
        self.linear_inequality_constraints[group_idx].append(constraint)

        return constraint

    def add_linear_equality_constraint(self, group_idx: int, *args):
        group_idx = self._find_numbered_idx(group_idx)
        constraint = self.prog.AddLinearEqualityConstraint(*args)
        self.linear_equality_constraints[group_idx].append(constraint)

        return constraint

    def add_quadratic_constraint(self, group_idx_1: int, group_idx_2: int, *args):
        group_idx_1 = self._find_numbered_idx(group_idx_1)
        group_idx_2 = self._find_numbered_idx(group_idx_2)

        constraint = self.prog.AddQuadraticConstraint(*args)
        self.quadratic_constraints[(group_idx_1, group_idx_2)].append(constraint)

        return constraint

    def add_quadratic_cost(self, group_idx_1: int, group_idx_2: int, *args):
        group_idx_1 = self._find_numbered_idx(group_idx_1)
        group_idx_2 = self._find_numbered_idx(group_idx_2)

        cost = self.prog.AddQuadraticCost(*args)
        self.quadratic_costs[(group_idx_1, group_idx_2)].append(cost)

        return cost

    def add_independent_constraint(self, *args):
        constraint = self.prog.AddConstraint(*args)
        self.independent_constraints.append(constraint)

    def add_l2_norm_cost(self, A, b, vars):
        """
        Add a cost that is just added to the relaxation without any tightening constraints
        """
        cost = self.prog.AddL2NormCost(A, b, vars)
        self.l2_norm_costs.append(cost)

        return cost

    def add_independent_cost(self, *args):
        """
        Add a cost that is just added to the relaxation without any tightening constraints
        """
        cost = self.prog.AddCost(*args)
        self.independent_costs.append(cost)

        return cost

    # TODO(bernhardpg): Temporary function to make it possible to use my old code
    def _make_bbox_to_expr(
        self, bounding_box_bindings: List[Binding[BoundingBoxConstraint]]  # type: ignore
    ) -> NpExpressionArray:
        bounding_box_constraints = []
        for b in bounding_box_bindings:
            x = b.variables()
            b_upper = b.evaluator().upper_bound()
            b_lower = b.evaluator().lower_bound()

            for x_i, b_u, b_l in zip(x, b_upper, b_lower):
                if b_u == b_l:  # eq constraint
                    # TODO: Remove this part
                    raise ValueError("Bounding box equalities are not supported!")
                    bounding_box_constraints.append((x_i - b_u, ConstraintType.EQ))
                else:
                    if not np.isinf(b_u):
                        bounding_box_constraints.append(b_u - x_i)
                    if not np.isinf(b_l):
                        bounding_box_constraints.append(x_i - b_l)

        return np.array(bounding_box_constraints)

    def make_relaxation(
        self, trace_cost: Optional[float] = None, add_l2_norm_cost: bool = False
    ) -> MathematicalProgram:
        relaxed_prog = MathematicalProgram()

        # First gather variables
        self.xs = {
            idx: np.concatenate(self.groups[idx]).reshape((-1, 1))
            for idx in range(self.num_groups)
        }

        # Form smaller PSD cones
        self.Xs = {}
        self.Ys = {}
        for idx in range(self.num_groups - 1):
            x = self.xs[idx]
            x_next = self.xs[idx + 1]

            relaxed_prog.AddDecisionVariables(x)
            relaxed_prog.AddDecisionVariables(x_next)
            num_vars = len(x) + len(x_next)
            X = relaxed_prog.NewSymmetricContinuousVariables(num_vars, f"X_{idx}")
            self.Xs[idx] = X

            # Y = [1; x; x_next] [1; x'; x_next']'
            y = np.concatenate((x, x_next))
            # fmt: off
            Y = np.block([[1, y.T],
                          [y, X]])
            # fmt: on
            relaxed_prog.AddPositiveSemidefiniteConstraint(Y)
            self.Ys[idx] = Y

        # Cones must intersect
        for idx in range(self.num_groups - 2):
            X = self.Xs[idx]
            X_next = self.Xs[idx + 1]

            # X = [x; y][x' y']
            # X = [xx' xy'
            #      yx' yy']
            # X_next = [yy' zy'
            #           zy' zz']
            # We enforce yy' == yy'
            y_len = len(self.xs[idx + 1])
            const = X[-y_len:, -y_len:] - X_next[:y_len, :y_len]
            for c in const.flatten():
                relaxed_prog.AddLinearEqualityConstraint(c, 0)

        # We add independent constraints without any fancy relaxation machinery
        for const in self.independent_constraints:
            relaxed_prog.AddConstraint(const.evaluator(), const.variables())

        # We add independent costs without any fancy relaxation machinery
        for cost in self.independent_costs:
            relaxed_prog.AddCost(cost.evaluator(), cost.variables())

        if add_l2_norm_cost:
            for cost in self.l2_norm_costs:
                A = cost.evaluator().A()
                b = cost.evaluator().b()
                vars = cost.variables()
                relaxed_prog.AddL2NormCostUsingConicConstraint(A, b, vars)

        # Add all linear constraints directly
        for idx in range(self.num_groups):
            for const in self.linear_inequality_constraints[idx]:
                relaxed_prog.AddConstraint(const.evaluator(), const.variables())
            for const in self.linear_equality_constraints[idx]:
                relaxed_prog.AddConstraint(const.evaluator(), const.variables())

        # Add all linear costs directly
        for idx in range(self.num_groups):
            for const in self.linear_costs[idx]:
                relaxed_prog.AddCost(const.evaluator(), const.variables())

        # Add constraints implied by linear equality constraints
        for idx in range(self.num_groups - 1):
            x = self.xs[idx]
            y = self.xs[idx + 1]
            eqs_i = self.linear_equality_constraints[idx]
            eqs_j = self.linear_equality_constraints[idx + 1]

            for c in eqs_i + eqs_j:
                if isinstance(c, Binding[BoundingBoxConstraint]):
                    # NOTE: I am not sure if I am handling these yet!
                    # breakpoint()
                    ...

            # [b A][1; x] = 0
            # TODO: what about BBOX constraints?
            A = linear_bindings_to_homogenuous_form(eqs_i, np.array([]), x)
            B = linear_bindings_to_homogenuous_form(eqs_j, np.array([]), y)

            x_len = len(x)
            y_len = len(y)

            # X = [xx' xy'
            #      yx' yy']
            X = self.Xs[idx]
            xxT = X[:x_len, :x_len]
            xyT = X[:x_len, -y_len:]
            yyT = X[-y_len:, -y_len:]

            x_bar_x_bar_T = np.block([[1, x.T], [x, xxT]])
            x_bar_y_bar_T = np.block([[1, y.T], [x, xyT]])
            y_bar_y_bar_T = np.block([[1, y.T], [y, yyT]])

            # TODO: Will be adding some constraints multiple times here!

            # A[1; x] = 0 => A[1; x][1; x]' = 0 and A[1; x][1; y]' = 0
            for c in A.dot(x_bar_x_bar_T).flatten():
                relaxed_prog.AddLinearEqualityConstraint(c, 0)
            for c in A.dot(x_bar_y_bar_T).flatten():
                relaxed_prog.AddLinearEqualityConstraint(c, 0)

            # B[1; y] = 0 => B[1; y][1; y]' = 0 and B[1; y][1; x]' = 0
            for c in B.dot(y_bar_y_bar_T).flatten():
                relaxed_prog.AddLinearEqualityConstraint(c, 0)
            for c in B.dot(x_bar_y_bar_T.T).flatten():
                relaxed_prog.AddLinearEqualityConstraint(c, 0)

        # Add constraints implied by linear equality constraints
        for idx in range(self.num_groups - 1):
            x = self.xs[idx]
            y = self.xs[idx + 1]
            ineqs_i = [
                ineq
                for ineq in self.linear_inequality_constraints[idx]
                if isinstance(ineq, Binding[LinearConstraint])
            ]
            bboxs_i = [
                ineq
                for ineq in self.linear_inequality_constraints[idx]
                if isinstance(ineq, Binding[BoundingBoxConstraint])
            ]
            ineqs_j = [
                ineq
                for ineq in self.linear_inequality_constraints[idx + 1]
                if isinstance(ineq, Binding[LinearConstraint])
            ]
            bboxs_j = [
                ineq
                for ineq in self.linear_inequality_constraints[idx + 1]
                if isinstance(ineq, Binding[BoundingBoxConstraint])
            ]

            # [b A][1; x] >= 0
            A = linear_bindings_to_homogenuous_form(
                ineqs_i, self._make_bbox_to_expr(bboxs_i), x
            )
            B = linear_bindings_to_homogenuous_form(
                ineqs_j, self._make_bbox_to_expr(bboxs_j), y
            )

            x_len = len(x)
            y_len = len(y)

            # X = [xx' xy'
            #      yx' yy']
            X = self.Xs[idx]
            xxT = X[:x_len, :x_len]
            xyT = X[:x_len, -y_len:]
            yyT = X[-y_len:, -y_len:]

            x_bar_x_bar_T = np.block([[1, x.T], [x, xxT]])
            x_bar_y_bar_T = np.block([[1, y.T], [x, xyT]])
            y_bar_y_bar_T = np.block([[1, y.T], [y, yyT]])

            # TODO: Will be adding some constraints multiple times here!
            # A[1; x] >= 0 and B[1; y] >= 0
            # => A[1; x][1; x]'A' >= 0 and A[1; x][1; y]'B' >= 0 and B[1; y][1; y]'B' >=
            for c in (A @ x_bar_x_bar_T @ A.T).flatten():
                relaxed_prog.AddLinearConstraint(c, 0, np.inf)
            for c in (A @ x_bar_y_bar_T @ B.T).flatten():
                relaxed_prog.AddLinearConstraint(c, 0, np.inf)
            for c in (B @ y_bar_y_bar_T @ B.T).flatten():
                relaxed_prog.AddLinearConstraint(c, 0, np.inf)

        # Quadratic constraints
        for (i, j), consts in self.quadratic_constraints.items():
            # Only constraint coupling subsequent groups is supported
            assert i == i or i + 1 == j

            # We add each quadratic constraint separately
            for c in consts:
                eval = c.evaluator()
                vars = c.variables()

                bTx = eval.b().T @ vars
                Q = eval.Q()

                # get the variables in the monomials described by Q
                Q_idxs = list(zip(*np.where(Q != 0)))

                # get the idxs of these variables in our variable groups
                if i == self.num_groups - 1:  # Last group
                    assert i == j  # for the last group, we can not have j bigger than i
                    x = np.concatenate((self.xs[i - 1], self.xs[i]))
                elif i + 1 == j:
                    x = np.concatenate((self.xs[i], self.xs[j]))
                    # TODO(bernhardpg): Why is the lower and upper bound changing when we only have quadratic terms?
                    # TODO(bernhardpg): Seems like a bug that the lower and upper bound is scaled by 2?
                    SCALE = 1
                elif i == j:  # only squared terms
                    x = self.xs[i]
                    # TODO(bernhardpg): Why is the lower and upper bound changing when we only have quadratic terms?
                    # TODO(bernhardpg): Seems like a bug that the lower and upper bound is scaled by 2?
                    SCALE = 1.0

                idxs_map = {
                    k: l
                    for k, var_k in enumerate(vars)
                    for l, var_l in enumerate(x.flatten())
                    if var_k.EqualTo(var_l)
                }
                var_idxs = [(idxs_map[i], idxs_map[j]) for i, j in Q_idxs]

                if i == self.num_groups - 1:  # Last group
                    assert i == j  # for the last group, we can not have j bigger than i
                    X = self.Xs[
                        i - 1
                    ]  # for the last group we need to pick the last part of the previous X
                else:
                    X = self.Xs[i]
                X_vars = np.array([X[k, l] for (k, l) in var_idxs])
                coeffs = np.array([Q[k, l] for (k, l) in Q_idxs])

                # 0.5 x'Qx + b'x
                const = 0.5 * coeffs.T @ X_vars + bTx
                relaxed_prog.AddLinearConstraint(
                    const,
                    SCALE * eval.lower_bound(),
                    SCALE * eval.upper_bound(),
                )

        # Quadratic costs
        for (i, j), costs in self.quadratic_costs.items():
            if len(costs) == 0:
                continue

            # Only constraint coupling subsequent groups is supported
            assert i == i or i + 1 == j

            # We add each quadratic constraint separately
            for c in costs:
                eval = c.evaluator()
                vars = c.variables()

                bTx = eval.b().T @ vars
                Q = eval.Q()

                # get the variables in the monomials described by Q
                Q_idxs = list(zip(*np.where(Q != 0)))

                # get the idxs of these variables in our variable groups
                if i == self.num_groups - 1:  # Last group
                    assert i == j  # for the last group, we can not have j bigger than i
                    x = np.concatenate((self.xs[i - 1], self.xs[i]))
                elif i + 1 == j:
                    x = np.concatenate((self.xs[i], self.xs[j]))
                elif i == j:  # only squared terms
                    x = self.xs[i]

                idxs_map = {
                    k: l
                    for k, var_k in enumerate(vars)
                    for l, var_l in enumerate(x.flatten())
                    if var_k.EqualTo(var_l)
                }
                var_idxs = [(idxs_map[i], idxs_map[j]) for i, j in Q_idxs]

                if i == self.num_groups - 1:  # Last group
                    assert i == j  # for the last group, we can not have j bigger than i
                    X = self.Xs[
                        i - 1
                    ]  # for the last group we need to pick the last part of the previous X
                else:
                    X = self.Xs[i]

                X_vars = np.array([X[k, l] for (k, l) in var_idxs])
                coeffs = np.array([Q[k, l] for (k, l) in Q_idxs])

                # 0.5 x'Qx + b'x
                cost = 0.5 * coeffs.T @ X_vars + bTx + eval.c()
                # TODO(bernhardpg): Seems like a bug that the lower and upper bound is scaled by 2?
                relaxed_prog.AddLinearCost(cost)

        if trace_cost is not None:
            for X in self.Xs.values():
                relaxed_prog.AddLinearCost(trace_cost * np.trace(X))

        self.relaxed_prog = relaxed_prog
        return self.relaxed_prog

    def get_full_X(self) -> NpVariableArray:
        """
        Gets the full X = [xx'] after using the band sparse relaxation.
        Fills in 0 for all terms that are not present in sparse relaxation.

        NOTE: Does not get the full X with normal relaxation!
        """
        assert self.relaxed_prog is not None
        assert self.Xs is not None
        assert self.xs is not None

        # y = [1; x]
        num_vars = len(self.prog.decision_variables()) + 1
        big_X = np.zeros((num_vars, num_vars), dtype=object)
        big_X[0, 0] = 1

        # Fill in first row and column
        count = 1
        for idx in range(self.num_groups - 1):
            x = self.xs[idx].flatten()
            big_X[count : count + len(x), 0] = x
            big_X[0, count : count + len(x)] = x
            count += len(x)
        big_X[count : count + len(x), 0] = self.xs[self.num_groups - 1].flatten()
        big_X[0, count : count + len(x)] = self.xs[self.num_groups - 1].flatten()

        # Fill in blocks
        count = 1
        for idx in range(self.num_groups - 1):
            x = self.xs[idx].flatten()
            X = self.Xs[idx]
            big_X[count : count + len(x), count : count + len(x)] = X[
                : len(x), : len(x)
            ]
            count += len(x)

        x = self.xs[self.num_groups - 1].flatten()
        big_X[count : count + len(x), count : count + len(x)] = self.Xs[
            self.num_groups - 2
        ][-len(x) :, -len(x) :]

        return big_X

    # TODO: This should not really be a part of this class
    def make_full_relaxation(self, trace_cost: Optional[float]) -> MathematicalProgram:
        self.relaxed_prog = MakeSemidefiniteRelaxation(self.prog)

        if trace_cost is not None:
            assert (
                len(self.relaxed_prog.positive_semidefinite_constraints()) == 1
            )  # should only be one big X
            X = self.relaxed_prog.positive_semidefinite_constraints()[0].variables()
            N = np.sqrt(len(X))
            assert int(N) == N
            X = X.reshape((int(N), int(N)))
            self.relaxed_prog.AddLinearCost(trace_cost * np.trace(X))
        return self.relaxed_prog
