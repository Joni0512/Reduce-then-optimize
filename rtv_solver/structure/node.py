class Node:
    """
    Position of a node in the street network defined by latitude and longitude.
    """
    def __init__(self, lat, lon, id = None):
        self.lat = lat
        self.lon = lon
        self.id = id # uncertain what this id is for, seems to be stored in NetworkHandler and not as part of the node itself

    def __str__(self):
        return "<Node: {lat: {0}, lon: {1}, id: {2}}>".format(self.lat,self.lon,self.id)
