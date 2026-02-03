import json

from rtv_solver.visuals.route_manifest_mapper import RouteManifestMapper

ALLOWED_GEOMETRY_TYPES = {
    "Point", "MultiPoint",
    "LineString", "MultiLineString",
    "Polygon", "MultiPolygon"
}

def test_visual_geoJsonCreation():
    """check if the structure of the output is valid GeoJson"""
    # initialize data
    filename = 'rtv-solver/rtv_solver/visuals/debug_output.json'
    with open(filename, 'r') as json_file:
        loaded_data = json.load(json_file)

    mapper = RouteManifestMapper()
    result = mapper.manifest_to_geojson(loaded_data)

    # --- parse JSON if needed ---
    if isinstance(result, str):
        geojson = json.loads(result)
    else:
        geojson = result

    assert isinstance(geojson, dict)
    assert geojson.get("type") == "FeatureCollection"
    assert "features" in geojson
    assert isinstance(geojson["features"], list)

    for feature in geojson["features"]:
        assert isinstance(feature, dict)
        assert feature.get("type") == "Feature"

        # properties must exist (can be empty dict)
        assert "properties" in feature
        assert isinstance(feature["properties"], dict)

        # geometry must exist (can be None, but usually dict)
        assert "geometry" in feature
        geometry = feature["geometry"]

        assert geometry is None or isinstance(geometry, dict)

        if geometry is not None:
            assert geometry.get("type") in ALLOWED_GEOMETRY_TYPES
            assert "coordinates" in geometry


