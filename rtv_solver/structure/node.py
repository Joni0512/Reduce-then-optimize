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
    id: Optional[int] = None # uncertain what this id is for, seems to be stored in NetworkHandler and not as part of the node itself
    
    @classmethod
    def from_dict(cls, loc_dict) -> Node:
        return cls(loc_dict["lat"], loc_dict["lon"], loc_dict.get("node_id"))
    
    def to_dict(self) -> dict:
        node_dict = asdict(self)
        if self.id is not None:
            node_dict["node_id"] = self.id
        return node_dict
    
    def __str__(self):
        return f"<Node: lat: {self.lat}, lon: {self.lon}, id: {self.id}>"
    
    def __repr__(self):
        return f"Node(lat={self.lat}, lon={self.lon}, id={self.id})"

    def copy(self) -> Node:
        return Node(self.lat, self.lon, self.id)
    
    def __eq__(self, other: Node) -> bool:
        if not isinstance(other, Node):
            return False
        return self.lat == other.lat and self.lon == other.lon