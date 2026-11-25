"""
2026-11-24: LKH-3 as the per-step optimizer inside RHO (user request "LKH im RHO-Schritt", rejection variant R1).

Replaces OnlineRTVSolver.solve_pdptw_rtv (RTV trips + Gurobi ILP) by one static PDPTW per rolling-horizon step,
solved with the LKH-3 binary (c4v4 fork, LKH-3.0.6 base, external_repos/LKH3). Same inputs/outputs as
solve_pdptw_rtv, so OfflineRTVSolver's loop, simulate_manifest and StatsParser are unchanged.

Mapping of the RHO step onto LKH's PDPTW (verified in LKH3/SRC/Penalty_PDPTW.c + ReadProblem.c: one depot, every
route starts at time 0 at the depot, every node must be visited, one capacity, one asymmetric cost matrix = travel time):
- vehicle k at (state.loc, state.location_dt_seconds): start node s_k, arc depot->s_k = 0, every other arc into s_k
  = BIG, time window [t_k, t_k]  -> each of the K routes starts with exactly one s_k.
- request already on board of vehicle k: dummy pickup at k's location (only reachable from s_k or another dummy of
  k, arc cost 0) paired with the real delivery -> load, precedence and vehicle are kept.
- active (accepted, not picked up) and new requests: normal pickup/delivery pairs.
- route end: arc to depot = travel time if RETURN_DEPOT (depot latest = state.end_time), else 0.
- integer costs: times scaled by TIME_SCALE; travel/service/earliest rounded UP, latest rounded DOWN, so an
  LKH-feasible plan is also feasible in real-valued time (checked again in Python anyway).

Rejection (LKH has none, every node must be visited) = R1 "iterativ streichen": solve with all requests, check the
plan in real time, drop the NEW requests involved in violations, re-solve; at most MAX_DROP_ITERS rounds. If only
active/boarded requests are violated, LKH fails, or the rounds run out -> fallback to the normal RTV-ILP step
(user decision 2026-11-24), counted in LKHStepStats.
"""
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field

from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.structure.vehicle_stop import VehicleStop

import logging
from rtv_solver.util.logger import BASIC_LOGGER

console_logger = logging.getLogger(BASIC_LOGGER)

# 2026-11-24: default binary built by the user from github.com/c4v4/LKH3 (make with -DCAVA_CUSTOM); override via env LKH_BIN
DEFAULT_LKH_BIN = "/Users/joni/Desktop/Masterarbeit/external_repos/LKH3/LKH"
TIME_SCALE = 100          # 0.01 time-unit resolution; C = cost * PRECISION(100) must stay < 2^31 in LKH
BIG = 1_000_000           # forbidden arc (scaled); > any Li&Lim horizon (<= ~3400 * 100) -> always a time-window violation
NO_LATEST = 1_000_000_000
EPS = 1e-6


@dataclass
class LKHStepStats:
    """2026-11-24: per-run counters for the LKH step (written by the driver script)."""
    steps: int = 0
    steps_lkh_ok: int = 0
    steps_fallback_ilp: int = 0
    fallback_reasons: dict = field(default_factory=dict)
    lkh_calls: int = 0
    lkh_time_s: float = 0.0
    dropped_requests: int = 0
    drop_iters_hist: dict = field(default_factory=dict)


class LKHFailure(Exception):
    pass


def _ceil(x):
    return int(math.ceil(x * TIME_SCALE - 1e-9))


def _floor(x):
    return int(math.floor(x * TIME_SCALE + 1e-9))


class LKHStepSolver:
    def __init__(self, config, ilp_step_fn, runs=1, time_limit_s=10, max_drop_iters=10, seed=1, lkh_bin=None, keep_files_dir=None,
                 max_trials=None, special=False):
        self.config = config
        # 2026-11-24: LKH "SPECIAL" (ReadParameters.c: GAIN23=NO, KICKS=1, KICK_TYPE=4, MAX_SWAPS=0, MOVE_TYPE=5 SPECIAL,
        # POPULATION_SIZE=10). Without it lr208 steps kept split pickup/delivery pairs (penalty 2-5) even with 60 s;
        # with it all 5 failing steps were feasible in 10 s (outputs/rho_lkh/diag93_files). User go 2026-11-24.
        self.special = special
        # 2026-11-24: LKH default MAX_TRIALS = #nodes stopped before TIME_LIMIT and left precedence penalties
        # (lc101 smoke test, 7/34 steps fallback); user go for MAX_TRIALS=10000 so TIME_LIMIT binds. None = LKH default.
        self.max_trials = max_trials
        self.ilp_step_fn = ilp_step_fn          # original OnlineRTVSolver.solve_pdptw_rtv (bound), used as fallback
        self.runs = runs
        self.time_limit_s = time_limit_s
        self.max_drop_iters = max_drop_iters
        self.seed = seed
        self.lkh_bin = lkh_bin or os.environ.get("LKH_BIN", DEFAULT_LKH_BIN)
        self.keep_files_dir = keep_files_dir
        self.stats = LKHStepStats()
        if not os.path.isfile(self.lkh_bin):
            raise FileNotFoundError(f"LKH binary not found: {self.lkh_bin}")

    # ------------------------------------------------------------------ public
    def solve_pdptw_rtv(self, payload, iteration=0, current_time=None):
        self.stats.steps += 1
        try:
            result = self._solve_with_lkh(payload, iteration)
        except LKHFailure as e:
            reason = str(e).split(":")[0]
            self.stats.fallback_reasons[reason] = self.stats.fallback_reasons.get(reason, 0) + 1
            self.stats.steps_fallback_ilp += 1
            console_logger.warning(f"[LKH step] it={iteration} t={current_time}: fallback to ILP ({e})")
            return self.ilp_step_fn(payload, iteration, current_time=current_time)
        self.stats.steps_lkh_ok += 1
        return result

    # ------------------------------------------------------------------ core
    def _solve_with_lkh(self, payload, iteration):
        tt = payload.get(PayloadKeys.TIME_MATRIX)
        if tt is None:
            raise LKHFailure("no_time_matrix: LKH step needs a precomputed travel_time_matrix")
        cfg = self.config
        # 2026-11-24: get_payload_object needs an initialized NetworkHandler (same call as solve_pdptw_rtv)
        NetworkHandler.init_from_payload(payload=payload, server_url=cfg.SERVER_URL)
        pobj =PayloadParser.get_payload_object(payload, dwell_pickup_default=cfg.DWELL_PICKUP,
                                                dwell_alight_default=cfg.DWELL_ALIGHT, online=True)
        boarded_ids = set(pobj.boarded_requests_keys)
        active_ids = set(pobj.active_requests_keys)
        requests = {r[PayloadKeys.REQ_BOOKING_ID]: r for r in pobj.requests}
        new_ids = [rid for rid in requests if rid not in boarded_ids and rid not in active_ids]

        vehicles = self._vehicle_states(pobj.driver_runs, requests, boarded_ids)
        depot_loc = int(payload[PayloadKeys.DEPOT]["node_id"])

        kept_new = list(new_ids)
        for drop_iter in range(self.max_drop_iters + 1):
            routes = self._run_lkh(tt, depot_loc, vehicles, requests, active_ids, kept_new)
            violated = self._check_routes(tt, depot_loc, vehicles, requests, routes)
            if not violated:
                self.stats.drop_iters_hist[drop_iter] = self.stats.drop_iters_hist.get(drop_iter, 0) + 1
                break
            to_drop = self._requests_to_drop(violated, routes, set(kept_new))
            if not to_drop:
                raise LKHFailure(f"committed_violation: only active/boarded requests violated ({sorted(violated)[:5]})")
            kept_new = [r for r in kept_new if r not in to_drop]
            self.stats.dropped_requests += len(to_drop)
        else:
            raise LKHFailure(f"drop_iters_exhausted: still infeasible after {self.max_drop_iters} rounds")

        return self._build_result(payload, pobj, tt, vehicles, requests, routes, new_ids, active_ids)

    # ------------------------------------------------------------------ vehicles
    def _vehicle_states(self, driver_runs, requests, boarded_ids):
        vehicles = []
        for dr in driver_runs:
            st = dr[PayloadKeys.DRIVER_STATE]
            serviced = st[PayloadKeys.DRIVER_STATE_LOC_SERV]
            onboard = []
            for stop in dr[PayloadKeys.DRIVER_MANIFEST]:
                bid = stop[PayloadKeys.MANIFEST_BOOKING_ID]
                if stop[PayloadKeys.MANIFEST_ACTION] == VehicleStop.ACT_PICKUP and stop[PayloadKeys.MANIFEST_ORDER] <= serviced and bid in boarded_ids:
                    onboard.append(bid)
            if st.get(PayloadKeys.DRIVER_STATE_WC_CAP, 0):
                raise LKHFailure("wc_capacity: LKH mapping supports one capacity dimension (am) only")
            vehicles.append({
                "run_id": int(st[PayloadKeys.DRIVER_STATE_RUN_ID]),
                "loc": int(st[PayloadKeys.DRIVER_STATE_LOC]["node_id"]),
                "t": float(st[PayloadKeys.DRIVER_STATE_DT_SEC]),
                "cap": int(st[PayloadKeys.DRIVER_STATE_AM_CAP]),
                "end": float(st[PayloadKeys.DRIVER_STATE_END_TIME]),
                "onboard": onboard,
            })
        caps = {v["cap"] for v in vehicles}
        if len(caps) != 1:
            raise LKHFailure(f"heterogeneous_capacity: {caps}")
        for r in requests.values():
            if r.get(PayloadKeys.REQ_WHEELCHAIR, 0):
                raise LKHFailure("wc_demand: wheelchair demand not supported")
        return vehicles

    # ------------------------------------------------------------------ LKH I/O
    def _run_lkh(self, tt, depot_loc, vehicles, requests, active_ids, kept_new):
        K = len(vehicles)
        nodes = [{"kind": "depot", "loc": depot_loc}]                                   # LKH id 1
        for k, v in enumerate(vehicles):
            nodes.append({"kind": "start", "veh": k, "loc": v["loc"]})
        for k, v in enumerate(vehicles):
            for bid in v["onboard"]:
                r = requests[bid]
                nodes.append({"kind": "dummy", "veh": k, "loc": v["loc"], "req": bid})
                nodes.append({"kind": "del", "req": bid, "loc": int(r[PayloadKeys.REQ_DROPOFF_PT]["node_id"])})
        for bid in list(active_ids) + list(kept_new):
            r = requests[bid]
            nodes.append({"kind": "pick", "req": bid, "loc": int(r[PayloadKeys.REQ_PICKUP_PT]["node_id"])})
            nodes.append({"kind": "del", "req": bid, "loc": int(r[PayloadKeys.REQ_DROPOFF_PT]["node_id"])})
        n = len(nodes)
        pick_idx, del_idx = {}, {}
        for i, nd in enumerate(nodes, start=1):
            if nd["kind"] in ("pick", "dummy"):
                pick_idx[nd["req"]] = i
            elif nd["kind"] == "del":
                del_idx[nd["req"]] = i

        ret = self.config.RETURN_DEPOT
        ends = {v["end"] for v in vehicles}
        depot_latest = _floor(min(ends)) if ret else NO_LATEST

        rows = []
        for i, a in enumerate(nodes):
            row = []
            for j, b in enumerate(nodes):
                if i == j:
                    c = 0
                elif b["kind"] == "depot":
                    c = _ceil(tt[a["loc"]][depot_loc]) if (ret and a["kind"] != "depot") else 0
                elif b["kind"] == "start":
                    c = 0 if a["kind"] == "depot" else BIG
                elif b["kind"] == "dummy":
                    c = 0 if (a["kind"] in ("start", "dummy") and a.get("veh") == b["veh"]) else BIG
                elif a["kind"] == "depot":
                    c = BIG
                else:
                    c = _ceil(tt[a["loc"]][b["loc"]])
                row.append(c)
            rows.append(row)

        pd_lines = []
        for i, nd in enumerate(nodes, start=1):
            kind = nd["kind"]
            if kind == "depot":
                pd_lines.append(f"{i} 0 0 {depot_latest} 0 0 0")
            elif kind == "start":
                t = _ceil(vehicles[nd["veh"]]["t"])
                pd_lines.append(f"{i} 0 {t} {t} 0 0 0")
            elif kind == "dummy":
                r = requests[nd["req"]]
                t = _ceil(vehicles[nd["veh"]]["t"])
                pd_lines.append(f"{i} {int(r[PayloadKeys.REQ_AMBULATORY])} 0 {t} 0 0 {del_idx[nd['req']]}")
            elif kind == "pick":
                r = requests[nd["req"]]
                pd_lines.append(f"{i} {int(r[PayloadKeys.REQ_AMBULATORY])} {_ceil(r[PayloadKeys.REQ_PICKUP_WINDOW_START])} "
                                f"{_floor(r[PayloadKeys.REQ_PICKUP_WINDOW_END])} {_ceil(r[PayloadKeys.REQ_DWELL_PICKUP])} 0 {del_idx[nd['req']]}")
            else:  # del
                r = requests[nd["req"]]
                pd_lines.append(f"{i} {-int(r[PayloadKeys.REQ_AMBULATORY])} {_ceil(r[PayloadKeys.REQ_DROPOFF_WINDOW_START])} "
                                f"{_floor(r[PayloadKeys.REQ_DROPOFF_WINDOW_END])} {_ceil(r[PayloadKeys.REQ_DWELL_ALIGHT])} {pick_idx[nd['req']]} 0")

        workdir = tempfile.mkdtemp(prefix="lkh_step_")
        prob, par, tour = (os.path.join(workdir, f) for f in ("step.pdptw", "step.par", "step.tour"))
        with open(prob, "w") as f:
            f.write(f"NAME : rho_step\nTYPE : PDPTW\nDIMENSION : {n}\nVEHICLES : {K}\nCAPACITY : {vehicles[0]['cap']}\n"
                    "EDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n")
            for row in rows:
                f.write(" ".join(map(str, row)) + "\n")
            f.write("PICKUP_AND_DELIVERY_SECTION\n" + "\n".join(pd_lines) + "\nDEPOT_SECTION\n1\n-1\nEOF\n")
        with open(par, "w") as f:
            f.write(f"PROBLEM_FILE = {prob}\nMTSP_SOLUTION_FILE = {tour}\nRUNS = {self.runs}\nTIME_LIMIT = {self.time_limit_s}\n"
                    f"SEED = {self.seed}\nTRACE_LEVEL = 0\n")
            if self.max_trials is not None:
                f.write(f"MAX_TRIALS = {self.max_trials}\n")
            if self.special:
                f.write("SPECIAL\n")
        t0 = time.time()
        try:
            proc = subprocess.run([self.lkh_bin, par], capture_output=True, text=True, timeout=self.time_limit_s * 10 + 120)
        except subprocess.TimeoutExpired:
            raise LKHFailure("lkh_timeout: subprocess timeout")
        finally:
            self.stats.lkh_calls += 1
            self.stats.lkh_time_s += time.time() - t0
        if proc.returncode != 0 or not os.path.isfile(tour):
            tail = (proc.stdout + proc.stderr)[-500:]
            if self.keep_files_dir is None:
                shutil.rmtree(workdir, ignore_errors=True)
            raise LKHFailure(f"lkh_error: rc={proc.returncode} {tail}")
        # 2026-11-24: read MTSP_SOLUTION_FILE, not TOUR_FILE - WriteTour.c skips writing when the best tour still has
        # penalty > 0 (and MTSP_OBJECTIVE unset), but R1 needs exactly those infeasible tours to find what to drop.
        # Format (MTSP_WriteSolution.c): header line, salesmen line, then one route per line "1 a b ... 1 (#size) Cost: c".
        ids = []
        with open(tour) as f:
            lines = f.read().splitlines()[2:]
        for line in lines:
            route_part = line.split("(#")[0].split()
            if not route_part:
                continue
            ids.extend(int(x) for x in route_part[:-1])  # drop the closing depot id; next line starts with it
        if self.keep_files_dir:
            os.makedirs(self.keep_files_dir, exist_ok=True)
            dst = os.path.join(self.keep_files_dir, f"step_{self.stats.lkh_calls:04d}")
            shutil.copytree(workdir, dst, dirs_exist_ok=True)
        shutil.rmtree(workdir, ignore_errors=True)
        return self._split_routes(ids, nodes, K)

    @staticmethod
    def _split_routes(ids, nodes, K):
        """Tour over ids 1..n plus depot copies n+1..n+K-1 -> {vehicle k: [node dicts after s_k]}."""
        n = len(nodes)
        is_depot = lambda x: x == 1 or x > n
        if not ids or not any(is_depot(x) for x in ids):
            raise LKHFailure("bad_tour: no depot in tour")
        start = next(i for i, x in enumerate(ids) if is_depot(x))
        seq = ids[start:] + ids[:start]
        raw, cur = [], None
        for x in seq:
            if is_depot(x):
                if cur is not None:
                    raw.append(cur)
                cur = []
            else:
                cur.append(nodes[x - 1])
        raw.append(cur)
        raw = [r for r in raw if r]
        # orientation: every route must start with its start node; if they all end with one, the tour is reversed
        if raw and all(r[-1]["kind"] == "start" for r in raw) and not all(r[0]["kind"] == "start" for r in raw):
            raw = [list(reversed(r)) for r in raw]
        routes = {}
        for r in raw:
            if r[0]["kind"] != "start" or sum(nd["kind"] == "start" for nd in r) != 1:
                raise LKHFailure("bad_tour: route does not start with exactly one vehicle start node")
            routes[r[0]["veh"]] = r[1:]
        if len(routes) != K:
            raise LKHFailure(f"bad_tour: {len(routes)} routes for {K} vehicles")
        return routes

    # ------------------------------------------------------------------ real-time check
    def _check_routes(self, tt, depot_loc, vehicles, requests, routes):
        """Re-evaluate each route in real-valued time. Returns set of request ids involved in a violation."""
        violated = set()
        for k, route in routes.items():
            v = vehicles[k]
            t, loc = v["t"], v["loc"]
            load = sum(int(requests[b][PayloadKeys.REQ_AMBULATORY]) for b in v["onboard"])
            seen_pick = set(v["onboard"])
            for nd in route:
                if nd["kind"] == "dummy":
                    if nd["veh"] != k or loc != v["loc"]:
                        violated.add(nd["req"])
                    continue
                if nd["kind"] == "start" or nd["kind"] == "depot":
                    violated.add(-1)
                    continue
                r = requests[nd["req"]]
                t += tt[loc][nd["loc"]]
                loc = nd["loc"]
                if nd["kind"] == "pick":
                    t = max(t, r[PayloadKeys.REQ_PICKUP_WINDOW_START])
                    if t > r[PayloadKeys.REQ_PICKUP_WINDOW_END] + EPS:
                        violated.add(nd["req"])
                    t += r[PayloadKeys.REQ_DWELL_PICKUP]
                    load += int(r[PayloadKeys.REQ_AMBULATORY])
                    seen_pick.add(nd["req"])
                    if load > v["cap"]:
                        violated.add(nd["req"])
                else:
                    t = max(t, r[PayloadKeys.REQ_DROPOFF_WINDOW_START])
                    if t > r[PayloadKeys.REQ_DROPOFF_WINDOW_END] + EPS or nd["req"] not in seen_pick:
                        violated.add(nd["req"])
                    t += r[PayloadKeys.REQ_DWELL_ALIGHT]
                    load -= int(r[PayloadKeys.REQ_AMBULATORY])
            if self.config.RETURN_DEPOT and route and t + tt[loc][depot_loc] > v["end"] + EPS:
                violated.add(route[-1].get("req", -1))
        return violated

    @staticmethod
    def _requests_to_drop(violated, routes, new_set):
        drop = {r for r in violated if r in new_set}
        if drop:
            return drop
        # only committed requests violated: drop the new requests that share a route with a violation
        for route in routes.values():
            reqs = {nd.get("req") for nd in route}
            if reqs & violated:
                drop |= (reqs & new_set)
        return drop

    # ------------------------------------------------------------------ manifests
    def _build_result(self, payload, pobj, tt, vehicles, requests, routes, new_ids, active_ids):
        from rtv_solver.online_rtv_solver import OnlineRTVSolver
        assignment = {}
        updated = []
        by_run = {v["run_id"]: k for k, v in enumerate(vehicles)}
        for dr in pobj.driver_runs:
            st = dr[PayloadKeys.DRIVER_STATE]
            k = by_run[int(st[PayloadKeys.DRIVER_STATE_RUN_ID])]
            v = vehicles[k]
            serviced = st[PayloadKeys.DRIVER_STATE_LOC_SERV]
            manifest = list(dr[PayloadKeys.DRIVER_MANIFEST][:serviced])
            order = serviced
            t, loc = v["t"], v["loc"]
            for nd in routes[k]:
                if nd["kind"] == "dummy":
                    continue
                r = requests[nd["req"]]
                is_pick = nd["kind"] == "pick"
                tws = r[PayloadKeys.REQ_PICKUP_WINDOW_START] if is_pick else r[PayloadKeys.REQ_DROPOFF_WINDOW_START]
                twe = r[PayloadKeys.REQ_PICKUP_WINDOW_END] if is_pick else r[PayloadKeys.REQ_DROPOFF_WINDOW_END]
                dwell = r[PayloadKeys.REQ_DWELL_PICKUP] if is_pick else r[PayloadKeys.REQ_DWELL_ALIGHT]
                pt = r[PayloadKeys.REQ_PICKUP_PT] if is_pick else r[PayloadKeys.REQ_DROPOFF_PT]
                t = max(t + tt[loc][nd["loc"]], tws)   # same rule as VehicleHandler.get_manifest
                loc = nd["loc"]
                order += 1
                manifest.append({
                    PayloadKeys.MANIFEST_RUN_ID: v["run_id"],
                    PayloadKeys.MANIFEST_BOOKING_ID: nd["req"],
                    PayloadKeys.MANIFEST_ORDER: order,
                    PayloadKeys.MANIFEST_ACTION: VehicleStop.ACT_PICKUP if is_pick else VehicleStop.ACT_DROPOFF,
                    PayloadKeys.MANIFEST_LOC: {"lat": pt["lat"], "lon": pt["lon"], "node_id": pt["node_id"]},
                    PayloadKeys.MANIFEST_SCHED_TIME: t,
                    PayloadKeys.MANIFEST_AMBULATORY: r[PayloadKeys.REQ_AMBULATORY],
                    PayloadKeys.MANIFEST_WHEELCHAIR: r.get(PayloadKeys.REQ_WHEELCHAIR, 0),
                    PayloadKeys.MANIFEST_TIME_WINDOW_START: tws,
                    PayloadKeys.MANIFEST_TIME_WINDOW_END: twe,
                    PayloadKeys.MANIFEST_DWELL: dwell,
                })
                t += dwell
                if is_pick:
                    assignment[nd["req"]] = v["run_id"]
            new_state = dict(st)
            new_state[PayloadKeys.DRIVER_STATE_T_LOCS] = len(manifest)
            updated.append({PayloadKeys.DRIVER_STATE: new_state, PayloadKeys.DRIVER_MANIFEST: manifest})

        unserved = set(new_ids) | set(active_ids)
        unserved -= set(assignment)
        OnlineRTVSolver._check_consistency_of_manifests(payload[PayloadKeys.DRIVERS], updated, unserved,
                                                        payload[PayloadKeys.REQUESTS],
                                                        keep_active=self.config.KEEP_ACTIVE,
                                                        return_depot=self.config.RETURN_DEPOT)
        return updated, {PayloadKeys.STATS_ASSIGNED: assignment, PayloadKeys.STATS_UNSERVED: list(unserved)}
