# rolling-horizen-RTV

## Running

- Set up an osrm server. Follow https://github.com/Project-OSRM/osrm-backend
- `cd` into the src folder.
- run `python main.py --server_url "" --input_file "" --out_put_dir "" --interval 300 --rh_factor 0 --max_cardinality 4`
- Required parameters:
    - server_url: Url of the OSRM server (ex: "http://127.0.0.1:5000/")
    - input_file: path to the payload.pkl file
    - out_put_dir: directory to record outputs
    - interval: interval for the rolling horizon and batching
    - rh_factor: rolling horizon factor
    - max_cardinality: meximum size of shared trips
