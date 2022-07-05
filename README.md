# taxi-public-transit-integration

## Initialize

Download [this](https://drive.google.com/file/d/1n9TeLRGiP5fh7ziVTvO8QWReTO5bQwjo/view?usp=sharing) and extract to the main directory.

## Running

- `mkdir output`
- `cd` into the src folder.
- Execute the code `python main.py --allow_bus --allow_bus_transfer --out_put_dir "../output/" --max_number_of_vehicles $NOVEH --max_capacity $CAP --max_cardinality 1`.
- Replace '$NOVEH' with the fleet size and '$CAP' with the maximum allowed capacity in any MoD vehicle.
