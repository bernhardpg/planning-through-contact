from typing import List, Optional

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.math import eq

from geometry.contact import CollisionPair, ModeConfig

# flake8: noqa


class GraphBuilder:
    def __init__(
        self,
        collision_pairs: List[CollisionPair],  # TODO for now I define this manually
        unactuated_bodies: List[str],  # TODO make part of RigidBody
        external_forces: List[sym.Expression],
        additional_constraints: Optional[List[sym.Formula]],
        allow_sliding: bool = False,
    ) -> None:

        self.collision_pairs = collision_pairs
        self.all_bodies = self._collect_all_rigid_bodies(collision_pairs)
        unactuated_dofs = self._get_unactuated_dofs(
            unactuated_bodies, self.all_bodies, self.dim
        )
        force_balance_constraints = self.construct_force_balance(
            collision_pairs,
            self.all_bodies,
            external_forces,
            unactuated_dofs,
        )
        for p in self.collision_pairs:
            p.add_force_balance(force_balance_constraints)
        for p in self.collision_pairs:
            p.add_constraint_to_all_modes(additional_constraints)

        # 1. Build source and target nodes (with force balance)
        # 2. Start with source node:
        #   - change one contact mode at a time to obtain the frontier.
        # 3. Add node edge if it is reachable
        # 4. Repeat until we 'hit' target node:
        #   - Repeat until we can actually make an edge to the target node
        #
        # How to deal with repeated visits to a node? For now we just make graph repeated after building it

    @property
    def dim(self) -> int:
        return self.collision_pairs[0].body_a.dim

    @staticmethod
    def _collect_all_rigid_bodies(pairs: List[CollisionPair]) -> List[str]:
        all_body_names = sorted(
            list(set(sum([[p.body_a.name, p.body_b.name] for p in pairs], [])))
        )
        return all_body_names

    def _collect_all_pos_vars(
        self, pairs: List[CollisionPair]
    ) -> npt.NDArray[sym.Variable]:
        all_pos_vars = np.concatenate(
            [np.concatenate((p.body_a.pos.x, p.body_b.pos.x)).flatten() for p in pairs]
        )
        unique_pos_vars = set(all_pos_vars)
        sorted_unique_pos_vars = np.array(
            sorted(list(unique_pos_vars)),
            key=lambda x: x.get_name(),
        )
        return sorted_unique_pos_vars

    def _collect_all_decision_vars(
        self, pairs: List[CollisionPair]
    ) -> npt.NDArray[sym.Variable]:
        all_pos_vars = self._collect_all_pos_vars(pairs)
        all_normal_force_vars = np.concatenate([p.lam_n for p in pairs]).flatten()
        all_friction_force_vars = np.concatenate([p.lam_f for p in pairs]).flatten()
        all_vars = np.concatenate(
            [all_pos_vars, all_normal_force_vars, all_friction_force_vars]
        )
        return all_vars

    def _get_unactuated_dofs(
        self, unactuated_bodies: List[str], all_bodies: List[str], dim: int
    ) -> npt.NDArray[np.int32]:
        unactuated_idxs = [all_bodies.index(b) * dim for b in unactuated_bodies]
        unactuated_dofs = np.concatenate(
            [np.arange(idx, idx + dim) for idx in unactuated_idxs]
        )
        return unactuated_dofs

    # TODO move to contact module
    @staticmethod
    def construct_force_balance(
        collision_pairs: List[CollisionPair],
        bodies: List[str],
        external_forces: npt.NDArray[sym.Expression],
        unactuated_dofs: npt.NDArray[np.int32],
    ) -> List[sym.Formula]:
        normal_jacobians = np.vstack(
            [p.get_normal_jacobian_for_bodies(bodies) for p in collision_pairs]
        )
        tangential_jacobians = np.vstack(
            [p.get_tangential_jacobian_for_bodies(bodies) for p in collision_pairs]
        )

        normal_forces = np.concatenate([p.lam_n for p in collision_pairs])
        friction_forces = np.concatenate([p.lam_f for p in collision_pairs])

        all_force_balances = eq(
            normal_jacobians.T.dot(normal_forces)
            + tangential_jacobians.T.dot(friction_forces)
            + external_forces,
            0,
        )
        force_balance = all_force_balances[unactuated_dofs, :]
        return force_balance

    def add_source(self, mode: ModeConfig) -> None:
        self.source = mode

    def add_target(self, mode: ModeConfig) -> None:
        self.target = mode
