from __future__ import annotations

from typing import Sequence

from rtv_solver.structure.request import Request


class MonteCarloReturnBuilder:
    """
    Computes the Monte Carlo return G_t for each rolling-horizon iteration
    of one finished episode.

    2026-08-15: G_t is a negative penalty count, to be maximized (closer to
    0 = better), not a ratio:

        G_t = -(relevant requests at t that are NOT served by episode end)
            = -(|relevant_t| - |relevant_t intersect serviced|)

    This comes from summing "-1 per request due in this window that never
    gets served" over every future window from t to the episode's end.
    Windows partition all remaining time (every still-relevant request's
    deadline falls into exactly one of them), so that per-window sum
    telescopes down to the single-set-difference formula above - no need to
    loop over individual windows or worry about their exact boundaries.

    "Relevant at t" means the request's pickup deadline has not passed yet
    as of t (latest_pickup_time >= t) - a plain filter on the raw request
    data, independent of which action was taken or how the rollout actually
    played out. Only which of those end up served needs the real simulation
    outcome (one StatsParser call at episode end, see EpisodeBuffer).

    "Serviced" only means a request got picked up and dropped off - it does
    NOT check punctuality on its own. That is fine here: the pickup deadline
    is already enforced as a hard feasibility constraint further upstream,
    in both of VehicleHandler's stop-sequencing paths (vehicle_handler.py -
    get_exact_stop_sequence line ~493, evaluate_stop_sequence line ~645) -
    a request can never end up "serviced" with a late pickup in the first
    place, so this reward can't accidentally reward a late one.

    G_t's scale is not fixed - it depends on how many requests are still
    relevant at t (naturally larger early in an episode, smaller near the
    end). Not a problem in principle since the critic also sees current_time
    as a feature, but noted here in case early training looks odd.

    Also time-blind within a request's own deadline: serving it immediately
    vs. barely before the deadline count the same, no reward for being fast.
    Rewarding speed specifically would need a genuinely different, discounted
    or continuously time-sensitive signal - deferred, not built here.

    2026-08-14: earlier version of this class computed a [0,1] ratio instead
    (kept here as a documented alternative, not deleted):

        G_t = |relevant_t intersect serviced| / |relevant_t|

    Same two underlying sets, just divided instead of subtracted. Bounded to
    [0,1] regardless of episode position (unlike the penalty count above),
    but doesn't preserve the "how many were missed" magnitude - trades scale
    stability for information content. Switch back if the unbounded penalty
    scale turns out to hurt training in practice.

    2026-08-18: build_local() is a second, genuinely different target (not
    just another cumulative variant) - user's proposal, promoted from a
    diagnostic-only value to an actual training target on request. Per-step
    local reward: -1 exactly in the window where a request's deadline passes
    without it ever being served, 0 otherwise (not a cumulative return - only
    counts requests that become permanently unservable IN that window, does
    not carry forward requests still missing from earlier windows).

        r_t = G_t - G_{t+1}   (with G_n := 0 for the step after the last one)

    where G_t is the same cumulative penalty count build() computes - r_t is
    literally the newly-added part of G_t at each step, so implemented here
    by calling build() and taking consecutive differences, not by
    re-deriving it from scratch.

    User's own caveat when proposing this (2026-08-18): "die Definition ist
    denke ich zu streng, außer sie funktioniert gut" - explicitly flagged as
    possibly too myopic (a request due in window 3 that only gets served in
    window 7 contributes nothing to r_t, since it was never permanently
    missed) - being tried empirically, not adopted as clearly correct.
    """

    def __init__(self, requests: Sequence[Request]) -> None:
        self.requests = requests

    def relevant_request_ids_at(self, current_time: float) -> set[int]:
        # deadline not passed yet = still worth serving from here on
        return {int(r.id) for r in self.requests if r.latest_pickup_time >= current_time}

    def build(
        self,
        iteration_times: Sequence[float],
        serviced_request_ids: Sequence[int],
    ) -> list[float]:
        """
        iteration_times: current_time of each rolling-horizon iteration in
            the episode, in order.
        serviced_request_ids: StatsParser.evaluate() result for the whole
            episode (pickup AND dropoff both completed) - call this once
            at episode end, not per iteration.

        Returns one G_t per entry in iteration_times, same order.
        """
        serviced = {int(rid) for rid in serviced_request_ids}
        returns: list[float] = []

        for current_time in iteration_times:
            relevant = self.relevant_request_ids_at(current_time)
            if not relevant:
                # nothing with an open deadline left at this point - no signal to measure
                returns.append(0.0)
                continue

            missed = relevant - serviced
            returns.append(-float(len(missed)))

            # old [0,1]-ratio alternative (see class docstring) - not used:
            # served = relevant & serviced
            # returns.append(len(served) / len(relevant))

        return returns

    def build_local(
        self,
        iteration_times: Sequence[float],
        serviced_request_ids: Sequence[int],
    ) -> list[float]:
        """
        2026-08-18: alternative training target (see class docstring) - -1
        only in the window where a request's deadline permanently passes
        unserved, 0 otherwise. Same iteration_times/serviced_request_ids
        contract as build().
        """
        cumulative = self.build(iteration_times, serviced_request_ids)
        # G_n := 0 (nothing left to miss after the episode's last iteration)
        shifted = cumulative[1:] + [0.0]
        return [g_t - g_next for g_t, g_next in zip(cumulative, shifted)]
