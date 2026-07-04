import json
import sys
from pathlib import Path

print("instance,serviced,total_requests,service_rate,vmt,avg_wait,avg_detour,avg_lateness,runtime")

for path in sys.argv[1:]:
    p = Path(path)
    data = json.loads(p.read_text())
    s = data["stats"]

    instance = p.parts[-3].split("_")[0]
    serviced = s["serviced"]
    total = s["total_requests"]
    rate = serviced / total * 100

    print(
        f"{instance},"
        f"{serviced},"
        f"{total},"
        f"{rate:.1f},"
        f"{s['vmt']:.2f},"
        f"{s['average_wait_time']:.2f},"
        f"{s['average_detour']:.2f},"
        f"{s['average_dropoff_goal_lateness']:.2f},"
        f"{s['total_time']:.2f}"
    )