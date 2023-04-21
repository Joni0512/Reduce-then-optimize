class Node:
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def __str__(self):
        return "{{lat: {0}, lon: {1}}}".format(self.lat,self.lon)
