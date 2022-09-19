import matplotlib.pyplot as plt
import cdd

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import List, Literal

import math
from pydrake.math import le, ge, eq
import pydrake.symbolic as sym
import pydrake.geometry.optimization as opt

from pydrake.geometry.optimization import GraphOfConvexSets
from pydrake.solvers import MathematicalProgram, Solve, MathematicalProgramResult

# TODO: Replace with VPolytope
class Polyhedron(opt.HPolyhedron):
    def get_vertices(self) -> npt.NDArray[np.float64]:  # [N, 2]
        A = self.A()
        b = self.b().reshape((-1, 1))
        dim = A.shape[0]
        cdd_matrix = cdd.Matrix(np.hstack((b, -A)))
        cdd_matrix.rep_type = cdd.RepType.INEQUALITY
        cdd_poly = cdd.Polyhedron(cdd_matrix)
        generators = np.array(cdd_poly.get_generators())
        # cdd specific, see https://pycddlib.readthedocs.io/en/latest/polyhedron.html
        vertices = generators[:, 1 : 1 + dim]
        return vertices

    @classmethod
    def from_vertices(cls, vertices: npt.NDArray[np.float64]) -> "Polyhedron":
        ones = np.ones((vertices.shape[0], 1))
        cdd_matrix = cdd.Matrix(np.hstack((ones, vertices)))
        cdd_matrix.rep_type = cdd.RepType.GENERATOR
        cdd_poly = cdd.Polyhedron(cdd_matrix)
        inequalities = np.array(cdd_poly.get_inequalities())
        b = inequalities[:, 0:1]
        A = -inequalities[:, 1:]

        return cls(A, b)
