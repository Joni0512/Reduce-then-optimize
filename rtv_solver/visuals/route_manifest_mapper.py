import json
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt

from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.structure.node import Node
from rtv_solver.structure.config import Config
from rtv_solver.structure.vehicle_stop import VehicleStop

from rtv_solver.visuals.map_icons import MakiIcon

class RouteManifestMapper():
    """
    Creates a GeoJSON FeatureCollection from a JSON data payload incl. all initial requests, the final vehicles, the complete manifest and the assignment development

    Upload .geojson file to the following website <https://geojson.io/#map=12.74/35.73105/-77.89457> with 'Open' to see the results of the routing.
    For GeoJSON the following icons are available


    TODO add `to_folium_map()` that consumes FeatureCollection to build an interactive map
    """
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self._network_initialized = False
        self._init_network()
        self.vehicle_colors = self._get_route_colors(2)

    def _init_network(self) -> None:
        if self._network_initialized:
            return
        NetworkHandler.init(True, self.config.server_url)
        self._network_initialized = True
    
    def _get_route_colors(self, veh_count: int) -> dict[int, str]:
        """Choose colormap from <https://matplotlib.org/stable/users/explain/colors/colormaps.html>"""
        cmap = plt.get_cmap("plasma")  # matplotlib colormap for oranges
        colors = [plt.cm.colors.to_hex(cmap(i / veh_count)) for i in range(0, veh_count)]
    
        return {i: color for i, color in enumerate(colors)}

    def manifest_to_geojson(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Convert payload["manifest"] into a GeoJSON FeatureCollection.
        """
        driver_runs = payload.get(PayloadParser.DRIVERS, [])
        depot = payload.get(PayloadParser.DEPOT)

        features: List[Dict[str, Any]] = []
        features.append(self._build_depot_feature(depot))
        for run in driver_runs:
            route = []
            state = run[PayloadParser.DRIVER_STATE]
            manifest = run[PayloadParser.DRIVER_MANIFEST]

            last_loc = Node.from_dict(depot[PayloadParser.DEPOT_PT]) # assumes all vehicles start in depot
            for stop in manifest:
                stop_feature = self._build_stop_feature(stop)
                features.append(stop_feature)

                next_loc = Node.from_dict(stop[PayloadParser.MANIFEST_LOC])
                route_part = self._build_simple_route_feature(last_loc, next_loc)
                route.append(route_part)

                # update location for right connections
                last_loc = next_loc

            route_feature = self._merge_routeparts_features(state[PayloadParser.DRIVER_STATE_RUN_ID], route)
            features.append(route_feature)
        return self._feature_collection(features)

    def save_geojson(self, geojson: Dict[str, Any], filepath: str) -> None:
        """Save GeoJSON to disk"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)

    def _build_stop_feature(self, stop: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build a Point feature for a stop action."""
        loc = stop.get(PayloadParser.MANIFEST_LOC, {})
        lon = loc.get("lon")
        lat = loc.get("lat")
        if lon is None or lat is None:
            return None
        
        design_props = self._build_stop_design_properties(stop)  
        properties = stop | design_props    
        return self._feature(geometry={
                                "type": "Point",
                                "coordinates": [float(lon), float(lat)],},
                             properties = properties) # dict with all information
    
    @staticmethod
    def _build_stop_design_properties(stop: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        action = stop.get(PayloadParser.MANIFEST_ACTION, "")
        run_id = stop.get(PayloadParser.MANIFEST_RUN_ID, "")
        order = stop.get(PayloadParser.MANIFEST_ORDER, "")
        color, icon = "#000000", MakiIcon.CIRCLE
        if action == VehicleStop.ACT_PICKUP:
            color = "#7d8aff"
            icon = MakiIcon.STAR
        elif action == VehicleStop.ACT_DROPOFF:
            color = "#c8cdff"
            icon = MakiIcon.TRIANGLE
        
        return {"title": f"{run_id}-{order}",
                "marker-color": color,
                "marker-size": "medium",
                "marker-symbol": icon
                }
    
    def _build_depot_feature(self, depot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build a Point feature for a stop action."""
        loc = depot.get(PayloadParser.DEPOT_PT, {})
        lon = loc.get("lon")
        lat = loc.get("lat")
        if lon is None or lat is None:
            return None
        return self._feature(geometry={
                                "type": "Point",
                                "coordinates": [float(lon), float(lat)],},
                             properties = {
                                "name": "depot",
                                "marker-color": "#ff0000",
                                "marker-size": "medium",
                                "marker-symbol": MakiIcon.BUS}) # dict with all information
    
    def _build_simple_route_feature(self, last_loc, next_loc):
        """simple route as we do not handle any complexities but rather re-create the street network route between two locations"""
        geometry = self._get_GEOJSON_route(last_loc, next_loc)
        return self._feature(geometry=geometry,
                            properties={"feature_type": "run_route"},) # TODO add more details to route
    
    def _merge_routeparts_features(self, run_id: int, route_parts: list[dict]) -> Dict[str, Any]:
        lines = [f["geometry"]["coordinates"] for f in route_parts if f["geometry"]["type"] == "LineString"]
        properties = {"feature_type": "merged_route",
                      "run-id": run_id,
                      "stroke": self.vehicle_colors[run_id], # design props
                      "stroke-width": "2",
                      "stroke-opacity": "0.6"}
        return self._feature(geometry={"type": "MultiLineString", "coordinates": lines},
                             properties=properties)

    @staticmethod
    def _get_GEOJSON_route(source, dest):
        response = NetworkHandler.get_detailed_route_reponse(source, dest)
        return response['routes'][0]['geometry']

    @staticmethod
    def _feature_collection(features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"type": "FeatureCollection", "features": features}

    @staticmethod
    def _feature(geometry: Dict[str, Any], properties: Dict[str, Any]) -> Dict[str, Any]:
        return {"type": "Feature", "geometry": geometry, "properties": properties}
   
if __name__ == '__main__':
    """
    debug test for visualizer
    """
    # load data from file and update to canonical format for the entire system
    filename = 'rtv-solver/rtv_solver/visuals/debug_output.json'
    with open(filename, 'r') as json_file:
        loaded_data = json.load(json_file)

    mapper = RouteManifestMapper()
    geojson = mapper.manifest_to_geojson(loaded_data)
    mapper.save_geojson(geojson, 'rtv-solver/rtv_solver/visuals/route_manifest.geojson')