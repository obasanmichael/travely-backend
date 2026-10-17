from __future__ import annotations

from api.v1.schemas import DestinationOut, PhotoOut
from engine.ranker import Destination
from engine.snapshot import Snapshot

TOP_TAGS = 4


def destination_out(destination: Destination, snapshot: Snapshot) -> DestinationOut:
    details = snapshot.details.get(destination.id)
    photo = details.photo if details else None
    return DestinationOut(
        id=destination.id,
        slug=destination.slug,
        name=destination.name,
        state=destination.state_name,
        state_iso=destination.place.state_iso,
        latitude=destination.place.lat,
        longitude=destination.place.lon,
        description=details.description if details else None,
        wikipedia_url=details.wikipedia_url if details else None,
        photo=PhotoOut(**photo.__dict__) if photo else None,
        tags=[t for t, _ in sorted(destination.tags.items(), key=lambda kv: -kv[1])[:TOP_TAGS]],
        best_months=list(details.best_months) if details else [],
        month_scores=list(details.month_scores) if details else [],
    )
