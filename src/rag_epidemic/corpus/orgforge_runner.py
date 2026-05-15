"""OrgForge-style prose wrapper around deterministic SimEvents.

For each event, we synthesise channel-appropriate prose (Slack message,
JIRA ticket, Confluence page, e-mail). We use a small set of templates
that are *strictly grounded* in the event payload so that hallucinations
cannot enter via the corpus generation step itself.

The Patient Zero injects extra events of kind 'fake_capacity_update'
which look like ordinary channel chatter but assert false capacity.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ground_truth import SimEvent


@dataclass
class CorpusChunk:
    chunk_id: str
    text: str
    channel: str
    sim_event_id: str
    author_agent: str
    t: int
    is_poisoned: bool = False


_TEMPLATES = {
    "demand_forecast": {
        "slack": "[demand] Warehouse {warehouse}: forecast demand {demand} units; nominal capacity {capacity}.",
        "email": "Subject: Demand forecast {warehouse}\n\nThe forecast for warehouse {warehouse} this superstep is {demand} units against a nominal capacity of {capacity}.",
        "jira": "DEMAND-{warehouse}: forecast={demand}, capacity={capacity}.",
        "confluence": "Demand snapshot for {warehouse}: forecast {demand}, capacity {capacity}.",
    },
    "vehicle_assignment": {
        "slack": "[ops] Vehicle {vehicle} assigned to {warehouse}.",
        "email": "Subject: Vehicle dispatch\n\nVehicle {vehicle} has been assigned to warehouse {warehouse}.",
        "jira": "DISPATCH-{vehicle}: warehouse={warehouse}.",
        "confluence": "Dispatch log: {vehicle} → {warehouse}.",
    },
    "incident": {
        "slack": "[incident] {warehouse} severity-{severity} incident reported.",
        "email": "Subject: Incident at {warehouse}\n\nA severity-{severity} incident has been reported at warehouse {warehouse}.",
        "jira": "INCIDENT-{warehouse}: severity={severity}.",
        "confluence": "Incident at {warehouse}: severity {severity}; investigation pending.",
    },
}


def event_to_chunk(event: SimEvent, author_agent: str = "system") -> CorpusChunk:
    kind = event.kind
    tmpl = _TEMPLATES.get(kind, {}).get(event.channel)
    if tmpl is None:
        text = f"[{kind}] {event.payload}"
    else:
        try:
            text = tmpl.format(**event.payload)
        except KeyError:
            text = f"[{kind}] {event.payload}"
    return CorpusChunk(
        chunk_id=f"chk_{event.event_id}",
        text=text,
        channel=event.channel,
        sim_event_id=event.event_id,
        author_agent=author_agent,
        t=event.t,
        is_poisoned=not event.is_ground_truth,
    )


def events_to_chunks(events: list[SimEvent], author_agent: str = "system") -> list[CorpusChunk]:
    return [event_to_chunk(e, author_agent=author_agent) for e in events]
