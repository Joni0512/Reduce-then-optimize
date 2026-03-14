from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Node:
    """
    Position of a node in the street network defined by latitude and longitude.
    """
    lat: float
    lon: float
    node_id: Optional[int] = None  # uncertain what this id is for, seems to be stored in NetworkHandler and not as part of the node itself

    @classmethod
    def from_dict(cls, loc_dict) -> Node:
        return cls(loc_dict["lat"], loc_dict["lon"], loc_dict.get("node_id"))
    
    def to_dict(self) -> dict:
        node_dict = asdict(self)
        if self.node_id is not None:
            node_dict["node_id"] = self.node_id
        return node_dict
    
    def __str__(self):
        return f"<Node: lat: {self.lat}, lon: {self.lon}, id: {self.node_id}>"
    
    def __repr__(self):
        return f"Node(lat={self.lat}, lon={self.lon}, id={self.node_id})"

    def copy(self) -> Node:
        return Node(self.lat, self.lon, self.node_id)
    
    def __eq__(self, other: Node) -> bool:
        if not isinstance(other, Node):
            return False
        return self.lat == other.lat and self.lon == other.lon

    @staticmethod
    def get_node_from_manifest_location(location: dict[str, float], node_id: int =None) -> Node:
        if 'node_id' in location and node_id is None:
            node_id = location['node_id']
        return Node(location["lat"], location["lon"], node_id)