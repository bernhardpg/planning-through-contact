from dataclasses import dataclass
from itertools import product
from typing import List, Optional

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.math import eq

from geometry.contact import CollisionPair, RigidBody

# flake8: noqa


@dataclass
class GraphBuilder:
    source_constraints: List[sym.Formula]
    target_constraints: List[sym.Formula]
    rigid_bodies: List[RigidBody]
    unactuated_bodies: List[str]  # TODO make part of RigidBody
    additional_constraints: Optional[List[sym.Formula]]

    def __post_init__(self) -> None:
        rigid_bodies = self.rigid_bodies
        body_1 = rigid_bodies[0]
        body_2 = rigid_bodies[2]
        collision_pairs = product(
            body_1.collision_geometries, body_2.collision_geometries
        )

        breakpoint()
        self.all_bodies = self._collect_all_rigid_bodies(self.collision_pairs)
        self.unactuated_dofs = self._get_unactuated_dofs(
            self.unactuated_bodies, self.all_bodies, self.dim
        )
        breakpoint()

        # 1. Build source and target nodes (with force balance)
        # 2. Start with source node:
        #   - change one contact mode at a time to obtain the frontier.
        # 3. Add node edge if it is reachable
        # 4. Repeat until we 'hit' target node:
        #   - Repeat until we can actually make an edge to the target node
        #
        # How to deal with repeated visits to a node? For now we just make graph repeated after building it

    def _collect_all_rigid_bodies(self, pairs: List[CollisionPair]) -> List[str]:
        all_body_names = sorted(
            list(set(sum([[p.body_a.name, p.body_b.name] for p in pairs], [])))
        )
        return all_body_names

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
