from abc import ABC, abstractmethod
from enum import Enum
from typing import List, NamedTuple, Tuple

import numpy as np
import numpy.typing as npt
from pydrake.geometry import Shape as DrakeShape

from planning_through_contact.geometry.hyperplane import Hyperplane


class ContactLocation(Enum):
    FACE = 1
    VERTEX = 2


class ContactMode(Enum):
    ROLLING = 1
    SLIDING_LEFT = 2
    SLIDING_RIGHT = 3


# TODO: move this?
class PolytopeContactLocation(NamedTuple):
    pos: ContactLocation
    idx: int

    def __str__(self) -> str:
        return f"{self.pos.name}_{self.idx}"


class CollisionGeometry(ABC):
    """
    Abstract class for all of the collision geometries supported by the contact planner,
    with all of the helper functions required by the planner.
    """

    @property
    @abstractmethod
    def vertices(self) -> List[npt.NDArray[np.float64]]:
        ...

    @property
    @abstractmethod
    def faces(self) -> List[Hyperplane]:
        ...

    @abstractmethod
    def get_proximate_vertices_from_location(
        self, location: PolytopeContactLocation
    ) -> List[npt.NDArray[np.float64]]:
        pass

    @abstractmethod
    def get_neighbouring_vertices(
        self, location: PolytopeContactLocation
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        pass

    @abstractmethod
    def get_hyperplane_from_location(
        self, location: PolytopeContactLocation
    ) -> Hyperplane:
        pass

    @abstractmethod
    def get_norm_and_tang_vecs_from_location(
        self, location: PolytopeContactLocation
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        pass

    @property
    @abstractmethod
    def vertices_for_plotting(self) -> npt.NDArray[np.float64]:
        pass

    @property
    @abstractmethod
    def contact_locations(self) -> List[PolytopeContactLocation]:
        pass

    @abstractmethod
    def get_face_length(self, location: PolytopeContactLocation) -> float:
        pass

    @classmethod
    @abstractmethod
    def from_drake(cls, drake_shape: DrakeShape):
        pass

    def get_shortest_vec_from_com_to_loc(
        self, location: PolytopeContactLocation
    ) -> npt.NDArray[np.float64]:
        v1, v2 = self.get_proximate_vertices_from_location(location)
        vec = (v1 + v2) / 2
        return vec

    @property
    @abstractmethod
    def num_collision_free_regions(self) -> int:
        pass

    @abstractmethod
    def get_collision_free_region_for_loc_idx(self, loc_idx: int) -> int:
        pass

    @abstractmethod
    def get_planes_for_collision_free_region(self, idx: int) -> List[Hyperplane]:
        """
        Returns the hyperplanes defining the collision-free region, but not the plane
        of the contact face.
        """

    @abstractmethod
    def get_contact_planes(self, idx: int) -> List[Hyperplane]:
        """
        Return the contact plane(s) that define the current collision-free region
        """

    def get_lam_from_p_BP(
        self,
        p_BP: npt.NDArray[np.float64],
        loc: PolytopeContactLocation,
        radius: float,
    ) -> npt.NDArray[np.float64]:
        assert loc.pos == ContactLocation.FACE
        assert p_BP.shape == (2, 1)
        pv1, pv2 = self.get_proximate_vertices_from_location(loc)

        n, t = self.get_norm_and_tang_vecs_from_location(loc)
        radius_offset = -n * radius
        point_on_surface = p_BP - radius_offset

        # project p_BP onto vector from v1 to v2 to find lam
        u1 = point_on_surface - pv2
        u2 = pv1 - pv2
        lam = u1.T.dot(u2).item() / np.linalg.norm(u2) ** 2
        return lam

    def get_p_Bc_from_lam(
        self, lam: float, loc: PolytopeContactLocation
    ) -> npt.NDArray[np.float64]:
        """
        Get the position of the contact point in the body frame.
        """
        assert loc.pos == ContactLocation.FACE
        pv1, pv2 = self.get_proximate_vertices_from_location(loc)
        p_Bc = lam * pv1 + (1 - lam) * pv2
        return p_Bc

    def get_p_BP_from_lam(
        self, lam: float, loc: PolytopeContactLocation, radius: float
    ) -> npt.NDArray[np.float64]:
        """
        Get the position of the pusher in the body frame (note: requires the
        radius to compute the position!)
        """
        p_Bc = self.get_p_Bc_from_lam(lam, loc)
        n, _ = self.get_norm_and_tang_vecs_from_location(loc)
        radius_offset = -n * radius

        p_BP = radius_offset + p_Bc
        return p_BP

    def get_force_comps_from_f_c_B(
        self, f_c_B, loc: PolytopeContactLocation
    ) -> Tuple[float, float]:
        n, t = self.get_norm_and_tang_vecs_from_location(loc)
        c_n = f_c_B.T.dot(n).item()
        c_f = f_c_B.T.dot(t).item()
        return c_n, c_f

    @property
    def max_dist_from_com(self) -> float:
        dists = [np.linalg.norm(v) for v in self.vertices]
        return np.max(dists)  # type: ignore
