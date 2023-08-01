import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pydot
from pydrake.systems.all import ConstantVectorSource
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import (
    Context,
    ContinuousState,
    ContinuousState_,
    DiagramBuilder,
    LeafSystem,
)
from pydrake.systems.primitives import (
    ConstantVectorSource_,
    LogVectorOutput,
    VectorLogSink,
)

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
)


class SliderPusherSystem(LeafSystem):
    """
    Implements the quasi-dynamic slider-pusher system, as described in
    the paper:

    [1] F. R. Hogan, E. R. Grau, and A. Rodriguez,
    “Reactive Planar Manipulation with Convex Hybrid MPC,”
    in 2018 IEEE International Conference on Robotics and
    Automation (ICRA), May 2018, pp. 247–253.
    doi: 10.1109/ICRA.2018.8461175.
    """

    def __init__(
        self,
        slider_geometry: CollisionGeometry,
        num_contact_points: int = 1,
    ) -> None:
        LeafSystem.__init__(self)

        self.slider_geometry = slider_geometry

        if not num_contact_points == 1:
            raise NotImplementedError("Currently only one contact point is supported")

        NUM_SLIDER_STATES = 3  # x, y, theta
        state_index = self.DeclareContinuousState(
            NUM_SLIDER_STATES + num_contact_points
        )  # x, y, theta, phi
        self.y = self.DeclareStateOutputPort("y", state_index)  # y = x

        NUM_INPUTS = 3  # f_n, f_t, phi_dot
        self.u = self.DeclareVectorInputPort("u", NUM_INPUTS)

        self.A = np.diag([0.1, 0.1, 0.1])  # TODO: change

    def _get_wrench(
        self, x: npt.NDArray[np.float64], u: npt.NDArray
    ) -> npt.NDArray[np.float64]:
        c_n, c_f, _ = u
        phi = x[3]  # need phi to determine which face we are on
        breakpoint()

    def _get_twist(
        self, x: npt.NDArray[np.float64], u: npt.NDArray
    ) -> npt.NDArray[np.float64]:
        w = self._get_wrench(x, u)
        return self.A.dot(w)

    def DoCalcTimeDerivatives(
        self, context: Context, derivatives: ContinuousState
    ) -> None:
        x = context.get_continuous_state_vector().value()
        derivatives.get_mutable_vector().set_value(x)
