from __future__ import annotations

class Node:
    """
    Position of a node in the street network defined by latitude and longitude.
    """
    def __init__(self, lat: float, lon: float, identifier = None):
        self.lat = lat
        self.lon = lon
        self.id = identifier # uncertain what this id is for, seems to be stored in NetworkHandler and not as part of the node itself
    
    @classmethod
    def from_dict(self, loc_dict) -> Node:
        return Node(loc_dict['lat'], loc_dict['lon'])
    
    def __str__(self):
        return f"<Node: lat: {self.lat}, lon: {self.lon}, id: {self.id}>"

    def copy(self) -> Node:
        return Node(self.lat, self.lon, self.id)