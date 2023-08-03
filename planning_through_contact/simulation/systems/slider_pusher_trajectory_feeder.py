from typing import Dict, List

import pydrake.geometry.optimization as opt
from pydrake.solvers import MathematicalProgramResult
from pydrake.systems.framework import BasicVector, Context, LeafSystem

from planning_through_contact.geometry.planar.abstract_mode import AbstractModeVariables
from planning_through_contact.geometry.planar.non_collision_subgraph import (
    VertexModePair,
)
from planning_through_contact.geometry.planar.planar_pushing_path import (
    PlanarPushingPath,
)

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge


class SliderPusherTrajectoryFeeder(LeafSystem):
    def __init__(self, path: List[AbstractModeVariables]) -> None:
        super().__init__()

        NUM_STATE_VARS = 4
        self.DeclareVectorOutputPort("state", NUM_STATE_VARS, self.CalcStateOutput)

        NUM_INPUT_VARS = 3
        self.DeclareVectorOutputPort("input", NUM_INPUT_VARS, self.CalcInputOutput)

    def CalcStateOutput(self, context: Context, output: BasicVector):
        output.SetFromVector([0, 0, 0, 0])

    def CalcInputOutput(self, context: Context, output: BasicVector):
        breakpoint()
        output.SetFromVector([0, 0, 0, 0])

    @classmethod
    def from_result(
        cls,
        result: MathematicalProgramResult,
        gcs: opt.GraphOfConvexSets,
        source_vertex: GcsVertex,
        target_vertex: GcsVertex,
        pairs: Dict[str, VertexModePair],
    ):
        path = PlanarPushingPath.from_result(
            gcs, result, source_vertex, target_vertex, pairs
        )
        return cls(path.get_rounded_vars())
