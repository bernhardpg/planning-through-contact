from pydrake.all import LeafSystem, QueryObject, AbstractValue
from typing import List


class ContactDetectionSystem(LeafSystem):
    def __init__(self, geometry_A_name: str, geometry_B_names: List[str]):
        LeafSystem.__init__(self)
        self._geometry_A_name = geometry_A_name
        self._geometry_B_names = geometry_B_names
        self._query_object = self.DeclareAbstractInputPort(
            "query_object", AbstractValue.Make(QueryObject())
        )
        self.DeclareVectorOutputPort("contact_detected", 1, self.DoCalcOutput)
        # print(f"ContactDetectionSystem: geometry_A_name {geometry_A_name}, geometry_B_names {geometry_B_names}")

    def DoCalcOutput(self, context, output):
        EPS = 1e-3
        query_object = self._query_object.Eval(context)
        inspector = query_object.inspector()
        bodyB_geometry_ids = []
        for geometry_id in inspector.GetAllGeometryIds():
            frame_id = inspector.GetFrameId(geometry_id)
            name = inspector.GetName(geometry_id)
            # print(f"Geometry id {geometry_id}, frame_id {frame_id}, name {name}")
            if name == self._geometry_A_name:
                bodyA_geometry_id = geometry_id
            if name in self._geometry_B_names:
                bodyB_geometry_ids.append(geometry_id)
        signed_dists = []
        for bodyB_geometry_id in bodyB_geometry_ids:
            signed_dists.append(
                query_object.ComputeSignedDistancePairClosestPoints(
                    bodyA_geometry_id, bodyB_geometry_id
                ).distance
            )
        signed_dist = min(signed_dists)
        output.set_value([signed_dist <= EPS])
