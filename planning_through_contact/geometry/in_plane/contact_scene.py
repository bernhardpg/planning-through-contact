from dataclasses import dataclass
from typing import List

from planning_through_contact.geometry.rigid_body import RigidBody


@dataclass
class ContactSceneDefinition:
    rigid_bodies: List[RigidBody]
    contact_pairs: List[ContactPairDefinition]
    body_anchored_to_W: RigidBody2d

    def create_instance(
        self,
        contact_pair_modes: Dict[str, ContactMode],
        instance_postfix: Optional[str] = None,
    ) -> "ContactSceneInstance":
        """
        Instantiates an instance of each contact pair with new sets of variables and constraints.
        """
        contact_pair_instances = [
            pair.create_instance(contact_pair_modes[pair.name], instance_postfix)
            for pair in self.contact_pairs
        ]
        return ContactSceneInstance(
            self.rigid_bodies, contact_pair_instances, self.body_anchored_to_W
        )

    # TODO duplicated code
    @property
    def unactuated_bodies(self) -> List[RigidBody2d]:
        bodies = [body for body in self.rigid_bodies if not body.actuated]
        return bodies
