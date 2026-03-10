import json
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
from pathlib import Path

from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.network_handler import NetworkHandler

from rtv_solver.structure.node import Node
from rtv_solver.structure.config import Config
from rtv_solver.structure.vehicle_stop import VehicleStop

from rtv_solver.visuals.map_icons import MakiIcon

from rtv_solver.util.helper import load_json

class RouteManifestMapper():
    """
    Creates a GeoJSON FeatureCollection from a JSON data payload incl. all initial requests, the final vehicles, the complete manifest and the assignment development

    Upload .geojson file to the following website <https://geojson.io/#map=12.74/35.73105/-77.89457> with 'Open' to see the results of the routing.
    For GeoJSON the following icons are available


    TODO add `to_folium_map()` that consumes FeatureCollection to build an interactive map
    """
    def __init__(self, config: Config | None = None):
        self.config = config
        self._network_initialized = False
        self._init_network()

    def _init_network(self) -> None:
        if self._network_initialized:
            return
        NetworkHandler.init_from_source(server_url=self.config.SERVER_URL)
        self._network_initialized = True
    
    def _get_route_colors(self, veh_count: int) -> dict[int, str]:
        """Choose colormap from <https://matplotlib.org/stable/users/explain/colors/colormaps.html>"""
        cmap = plt.get_cmap("plasma")  # matplotlib colormap
        colors = [plt.cm.colors.to_hex(cmap(i / veh_count)) for i in range(0, veh_count)]
    
        return {i: color for i, color in enumerate(colors)}

    def manifest_to_geojson(self, payload: dict[str, Any], vehicle_count: int = 18) -> dict[str, Any]:
        """
        Convert payload["manifest"] into a GeoJSON FeatureCollection.

        :param int veh_count: default 18 as we use that many vehicles normally with the standard file
        """
        driver_runs = payload.get(PayloadKeys.DRIVERS, [])
        depot = payload.get(PayloadKeys.DEPOT)
        self.vehicle_colors = self._get_route_colors(vehicle_count)

        features: List[Dict[str, Any]] = []
        features.append(self._build_depot_feature(depot))
        for run in driver_runs:
            route = []
            state = run[PayloadKeys.DRIVER_STATE]
            manifest = run[PayloadKeys.IVER_MANIFEST]

            last_loc = Node.from_dict(depot[PayloadKeys.DEPOT_PT]) # assumes all vehicles start in depot
            for stop in manifest:
                stop_feature = self._build_stop_feature(stop)
                features.append(stop_feature)

                next_loc = Node.from_dict(stop[PayloadKeys.NIFEST_LOC])
                route_part = self._build_simple_route_feature(last_loc, next_loc)
                route.append(route_part)

                # update location for right connections
                last_loc = next_loc

            # add depot stop if it has not been added yet as final stop (applies to online routing)
            if stop.get(PayloadKeys.MANIFEST_ACTION) != VehicleStop.ACT_DEPOT:
                new_stop = {
                    PayloadKeys.MANIFEST_ACTION: VehicleStop.ACT_DEPOT, 
                    PayloadKeys.MANIFEST_LOC: depot[PayloadKeys.DEPOT_PT],
                    PayloadKeys.MANIFEST_ORDER: len(manifest) + 1}
                stop_feature = self._build_stop_feature(new_stop)
                features.append(stop_feature)
                next_loc = Node.from_dict(depot[PayloadKeys.DEPOT_PT])
            
                route_part = self._build_simple_route_feature(last_loc, next_loc)
                route.append(route_part)

            route_feature = self._merge_routeparts_features(state[PayloadKeys.DRIVER_STATE_RUN_ID], route)
            features.append(route_feature)

        return self._feature_collection(features)

    def save_geojson(self, geojson: Dict[str, Any], filepath: Path | str) -> None:
        """Save GeoJSON to disk"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)

    def _build_stop_feature(self, stop: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build a Point feature for a stop action."""
        loc = stop.get(PayloadKeys.MANIFEST_LOC, {})
        lon = loc.get("lon")
        lat = loc.get("lat")
        if lon is None or lat is None:
            return None
        
        design_props = self._build_stop_design_properties(stop)  
        properties = stop | design_props   # append two dicts 
        return self._feature(geometry={
                                "type": "Point",
                                "coordinates": [float(lon), float(lat)],},
                             properties = properties) # dict with all information
    
    @staticmethod
    def _build_stop_design_properties(stop: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        action = stop.get(PayloadKeys.MANIFEST_ACTION, "n/a")
        run_id = stop.get(PayloadKeys.MANIFEST_RUN_ID, "n/a")
        order = stop.get(PayloadKeys.MANIFEST_ORDER, "n/a")
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
        loc = depot.get(PayloadKeys.DEPOT_PT, {})
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
                      "stroke": self.vehicle_colors[run_id], # design props, check how many vehicles participate
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
    folder = Path("/Users/jw/Desktop/master_thesis/mt_sourcecode/rtv-solver/outputs/storage/comp_v1/run_20260223_164646_optimal?") 
    
    with open(folder / "result_driver_runs.json", 'r') as driver_runs_file:
        loaded_data = json.load(driver_runs_file)

    config_file = load_json(folder / "config.json")
    config = Config.from_dict(config_file["config_dict"])

    mapper = RouteManifestMapper(config)
    geojson = mapper.manifest_to_geojson(loaded_data, 18)
    mapper.save_geojson(geojson, folder / "route_manifest_v2.geojson")