from typing import Callable, List, Optional, Dict, Any, Union
from dataclasses import dataclass
import pickle

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.solvers import MathematicalProgramResult
from scipy.spatial.transform import Rotation as R

from planning_through_contact.tools.types import NpExpressionArray, NpFormulaArray


def convert_formula_to_lhs_expression(form: sym.Formula) -> sym.Expression:
    lhs, rhs = form.Unapply()[1]  # type: ignore
    expr = lhs - rhs
    return expr


convert_np_formulas_array_to_lhs_expressions: Callable[
    [NpFormulaArray], NpExpressionArray
] = np.vectorize(convert_formula_to_lhs_expression)

convert_np_exprs_array_to_floats: Callable[
    [NpExpressionArray], npt.NDArray[np.float64]
] = np.vectorize(lambda expr: expr.Evaluate())


def evaluate_np_formulas_array(
    formulas: NpFormulaArray, result: MathematicalProgramResult
) -> npt.NDArray[np.float64]:
    expressions = convert_np_formulas_array_to_lhs_expressions(formulas)
    evaluated_expressions = convert_np_exprs_array_to_floats(
        result.GetSolution(expressions)
    )
    return evaluated_expressions


def evaluate_np_expressions_array(
    expr: NpExpressionArray, result: MathematicalProgramResult
) -> npt.NDArray[np.float64]:
    from_expr_to_float = np.vectorize(lambda expr: expr.Evaluate())
    solutions = from_expr_to_float(result.GetSolution(expr))
    return solutions


def calc_displacements(
    vars, dt: Optional[float] = None
) -> List[npt.NDArray[np.float64]]:
    if dt is not None:
        scale = 1 / dt
    else:
        scale = 1

    displacements = [
        (var_next - var_curr) * scale for var_curr, var_next in zip(vars[:-1], vars[1:])  # type: ignore
    ]
    return displacements


def skew_symmetric_so2(a):
    return np.array([[0, -a], [a, 0]])


def approx_exponential_map(omega_hat, num_dims: int = 2):
    # Approximates the exponential map (matrix exponential) by truncating terms of higher degree than 2
    return np.eye(num_dims) + omega_hat + 0.5 * omega_hat @ omega_hat

@dataclass
class PhysicalProperties:
    """
    A dataclass for physical properties.

    See https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html for
    more information about these properties.
    """

    mass: float
    """Mass in kg."""
    inertia: np.ndarray
    """Moment of inertia of shape (3,3)."""
    center_of_mass: np.ndarray
    """The center of mass that the inertia is about of shape (3,)."""
    is_compliant: bool
    """Whether the object is compliant or rigid in case of a Hydroelastic contact model.
    If compliant, the compliant Hydroelastic arguments are required."""
    hydroelastic_modulus: Union[float, None] = None
    """This is the measure of how stiff the material is. It directly defines how much
    pressure is exerted given a certain amount of penetration. More pressure leads to
    greater forces. Larger values create stiffer objects."""
    hunt_crossley_dissipation: Union[float, None] = None
    """A non-negative real value. This gives the contact an energy-damping property."""
    mu_dynamic: Union[float, None] = None
    """Dynamic coefficient of friction."""
    mu_static: Union[float, None] = None
    """Static coefficient of friction. Not used in discrete systems."""
    mesh_resolution_hint: Union[float, None] = None
    """A positive real value in meters. Most shapes (capsules, cylinders, ellipsoids,
    spheres) need to be tessellated into meshes. The resolution hint controls the
    fineness of the meshes. It is a no-op for mesh geometries and is consequently not
    required for mesh geometries."""

def load_primitive_info(primitive_info_file: str) -> List[Dict[str, Any]]:
    with open(primitive_info_file, "rb") as f:
        primitive_info = pickle.load(f)
    return primitive_info

def construct_drake_proximity_properties_sdf_str(
    physical_properties: PhysicalProperties, is_hydroelastic: bool
) -> str:
    """
    Constructs a Drake proximity properties SDF string using the proximity properties
    contained in `physical_properties`. Only adds the Hydroelastic properties if
    `is_hydroelastic` is true.
    """
    proximity_properties_str = """
            <drake:proximity_properties>
        """
    if is_hydroelastic:
        if physical_properties.is_compliant:
            assert (
                physical_properties.hydroelastic_modulus is not None
            ), "Require a Hydroelastic modulus for compliant Hydroelastic objects!"
            proximity_properties_str += f"""
                        <drake:compliant_hydroelastic/>
                        <drake:hydroelastic_modulus>
                            {physical_properties.hydroelastic_modulus}
                        </drake:hydroelastic_modulus>
                """
        else:
            proximity_properties_str += """
                    <drake:rigid_hydroelastic/>
            """
        if physical_properties.mesh_resolution_hint is not None:
            proximity_properties_str += f"""
                    <drake:mesh_resolution_hint>
                        {physical_properties.mesh_resolution_hint}
                    </drake:mesh_resolution_hint>
            """
    if physical_properties.hunt_crossley_dissipation is not None:
        proximity_properties_str += f"""
                    <drake:hunt_crossley_dissipation>
                        {physical_properties.hunt_crossley_dissipation}
                    </drake:hunt_crossley_dissipation>
            """
    if physical_properties.mu_dynamic is not None:
        proximity_properties_str += f"""
                    <drake:mu_dynamic>
                        {physical_properties.mu_dynamic}
                    </drake:mu_dynamic>
            """
    if physical_properties.mu_static is not None:
        proximity_properties_str += f"""
                    <drake:mu_static>
                        {physical_properties.mu_static}
                    </drake:mu_static>
            """
    proximity_properties_str += """
            </drake:proximity_properties>
        """
    return proximity_properties_str

def get_primitive_geometry_str(primitive_geometry: Dict[str,Any]) -> str:
    if primitive_geometry["name"] == "ellipsoid":
        radii = primitive_geometry["radii"]
        geometry = f"""
            <ellipsoid>
                <radii>{radii[0]} {radii[1]} {radii[2]}</radii>
            </ellipsoid>
        """
    elif primitive_geometry["name"] == "sphere":
        radius = primitive_geometry["radius"]
        geometry = f"""
            <sphere>
                <radius>{radius}</radius>
            </sphere>
        """
    elif primitive_geometry["name"] == "box":
        size = primitive_geometry["size"]
        geometry = f"""
            <box>
                <size>{size[0]} {size[1]} {size[2]}</size>
            </box>
        """
    elif primitive_geometry["name"] == "cylinder":
        height = primitive_geometry["height"]
        radius = primitive_geometry["radius"]
        geometry = f"""
            <cylinder>
                <radius>{radius}</radius>
                <length>{height}</length>
            </cylinder>
        """
    else:
        raise RuntimeError(f"Unsupported primitive type: {primitive_geometry['name']}")
    
    return geometry

def create_processed_mesh_primitive_sdf_file(
    primitive_info: List[Dict[str, Any]],
    physical_properties: PhysicalProperties,
    global_translation: np.ndarray,
    output_file_path: str,
    model_name: str,
    base_link_name: str,
    is_hydroelastic: bool,
    visual_mesh_file_path: Optional[str] = None,
    rgba: Optional[List[float]] = None,
) -> None:
    """
    Creates and saves an,rocessed mesh consisting of primitive
    geometries.

    :param primitive_info: A list of dicts containing primitive params. Each dict must
        contain "name" which can for example be sphere, ellipsoid, box, etc. and
        "transform" which is a homogenous transformation matrix. The other params are
        primitive dependent but must be sufficient to construct that primitive.
    :param physical_properties: The physical properties.
    :param global_translation: The translation of the processed mesh.
    :param output_file_path: The path to save the processed mesh SDF file.
    :param is_hydroelastic: Whether to make the body rigid hydroelastic.
    :param visual_mesh_file_path: The path to the mesh to use for the visual geometry.
    :param rgba: The color of the visual geometry. Only used if visual_mesh_file_path is
        None.
    """
    com = physical_properties.center_of_mass
    procesed_mesh_sdf_str = f"""
        <?xml version="1.0"?>
        <sdf version="1.7">
            <model name="{model_name}">
                <link name="{base_link_name}">
                    <inertial>
                        <inertia>
                            <ixx>{physical_properties.inertia[0,0]}</ixx>
                            <ixy>{physical_properties.inertia[0,1]}</ixy>
                            <ixz>{physical_properties.inertia[0,2]}</ixz>
                            <iyy>{physical_properties.inertia[1,1]}</iyy>
                            <iyz>{physical_properties.inertia[1,2]}</iyz>
                            <izz>{physical_properties.inertia[2,2]}</izz>
                        </inertia>
                        <mass>{physical_properties.mass}</mass>
                        <pose>{com[0]} {com[1]} {com[2]} 0 0 0</pose>
                    </inertial>
        """

    if visual_mesh_file_path is not None:
        procesed_mesh_sdf_str += f"""
                    <visual name="visual">
                        <pose>0 0 0 0 0 0</pose>
                        <geometry>
                            <mesh>
                                <uri>{visual_mesh_file_path}</uri>
                            </mesh>
                        </geometry>
                    </visual>
            """
    else:
        # Use primitives for the visual geometry.
        for i, info in enumerate(primitive_info):
            transform = info["transform"]
            translation = transform[:3, 3] + global_translation
            rotation = R.from_matrix(transform[:3, :3]).as_euler("XYZ")
            geometry = get_primitive_geometry_str(info)

            procesed_mesh_sdf_str += f"""
                <visual name="visual_{i}">
                    <pose>
                        {translation[0]} {translation[1]} {translation[2]} {rotation[0]} {rotation[1]} {rotation[2]}
                    </pose>
                    <geometry>
                        {geometry}
                    </geometry>
            """
            if rgba is not None:
                procesed_mesh_sdf_str += f"""
                    <material>
                        <diffuse> {rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]} </diffuse>
                    </material>
                """
            procesed_mesh_sdf_str += """
                </visual>
            """

    # Add the primitives
    for i, info in enumerate(primitive_info):
        transform = info["transform"]
        translation = transform[:3, 3] + global_translation
        rotation = R.from_matrix(transform[:3, :3]).as_euler("XYZ")
        geometry = get_primitive_geometry_str(info)

        procesed_mesh_sdf_str += f"""
            <collision name="collision_{i}">
                <pose>
                    {translation[0]} {translation[1]} {translation[2]} {rotation[0]} {rotation[1]} {rotation[2]}
                </pose>
                <geometry>
                    {geometry}
                </geometry>
            """

        assert (
            not is_hydroelastic or physical_properties.mesh_resolution_hint is not None
        ), "Require a mesh resolution hint for Hydroelastic primitive collision geometries!"
        procesed_mesh_sdf_str += construct_drake_proximity_properties_sdf_str(
            physical_properties, is_hydroelastic
        )

        procesed_mesh_sdf_str += """
                </collision>
            """

    procesed_mesh_sdf_str += """
                    </link>
                </model>
            </sdf>
        """

    with open(output_file_path, "w") as f:
        f.write(procesed_mesh_sdf_str)
