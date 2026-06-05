from pathlib import Path
from collections import Counter
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.schema.payload_keys import PayloadKeys

for payload_path in sorted(Path("inputs").rglob("*.pkl")):
    try:
        payload = PayloadParser.load_input_data(payload_path)
        req_count = len(payload.get(PayloadKeys.REQUESTS, []))
        print(payload_path, "=>", req_count)
    except Exception as e:
        print(payload_path, "=> ERROR:", e)

requests = payload[PayloadKeys.REQUESTS]

print("=" * 80)
print("PAYLOAD FILE:", payload_path)
print("PAYLOAD KEYS:", list(payload.keys()))
print("REQUEST COUNT:", len(requests))
print("DRIVER COUNT:", len(payload.get(PayloadKeys.DRIVERS, [])))
print("=" * 80)

if len(requests) == 0:
    print("No requests found.")
    raise SystemExit

booking_ids = [req[PayloadKeys.REQ_BOOKING_ID] for req in requests]

print("FIRST 10 BOOKING IDS:")
print(booking_ids[:10])

print("\nLAST 10 BOOKING IDS:")
print(booking_ids[-10:])

print("\nUNIQUE BOOKING IDS:", len(set(booking_ids)))
print("DUPLICATE BOOKING IDS:", len(booking_ids) - len(set(booking_ids)))

print("\nFIRST REQUEST:")
print(requests[0])

print("\nTIME WINDOW SUMMARY:")
pickup_starts = [req[PayloadKeys.REQ_PICKUP_WINDOW_START] for req in requests]
pickup_ends = [req[PayloadKeys.REQ_PICKUP_WINDOW_END] for req in requests]

print("Earliest pickup start:", min(pickup_starts))
print("Latest pickup start:", max(pickup_starts))
print("Earliest pickup end:", min(pickup_ends))
print("Latest pickup end:", max(pickup_ends))

print("\nWHEELCHAIR COUNTS:")
wc_counts = Counter(req[PayloadKeys.REQ_WHEELCHAIR] for req in requests)
print(wc_counts)

print("\nAMBULATORY COUNTS:")
am_counts = Counter(req[PayloadKeys.REQ_AMBULATORY] for req in requests)
print(am_counts)