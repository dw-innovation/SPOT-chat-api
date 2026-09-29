"""Client for SPOT's tag search API (the same search the SPOT pipeline uses).

    GET https://tags.findthatspot.io/api/search?word=<name>&limit=5&detail=false

returns the best matching bundles, each with the matched descriptor ("name"),
all descriptors of the bundle, a score and the OSM filter ("imr"). It is used
to tell the model how SPOT will read an entity name that is not an exact
descriptor. Configure with SPOT_TAG_SEARCH_URL (empty string disables it).
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import lru_cache

DEFAULT_URL = "https://tags.findthatspot.io/api/search"


@dataclass(frozen=True)
class Match:
    name: str                      # the descriptor that matched
    descriptors: tuple[str, ...]   # all descriptors of that bundle
    bundle_id: str
    score: float


class TagSearch:
    RETRY_AFTER = 120  # seconds to wait before trying the API again after a failure

    def __init__(self, url: str, timeout: float = 4.0, limit: int = 5):
        self.url, self.timeout, self.limit = url, timeout, limit
        self._cache: dict[str, tuple[Match, ...]] = {}
        self._down_until = 0.0

    def search(self, word: str) -> tuple[Match, ...] | None:
        """Matches, best first (one per bundle), or None if the API is unreachable."""
        key = " ".join(word.lower().split())
        if key in self._cache:
            return self._cache[key]
        if time.time() < self._down_until:
            return None
        query = urllib.parse.urlencode({"word": key, "limit": self.limit, "detail": "false"})
        try:
            with urllib.request.urlopen(f"{self.url}?{query}", timeout=self.timeout) as response:
                matches = parse_results(json.load(response))
        except Exception as exc:  # network down, blocked, bad JSON: use the local check for now
            print(f"Tag search unavailable ({exc!r}); using the local descriptor check.")
            self._down_until = time.time() + self.RETRY_AFTER
            return None
        if len(self._cache) < 10_000:
            self._cache[key] = matches
        return matches


def parse_results(results) -> tuple[Match, ...]:
    matches, seen = [], set()
    for item in results or []:
        bundle = item.get("cluster_id") or item.get("name")
        if bundle in seen:
            continue
        seen.add(bundle)
        matches.append(Match(
            name=str(item.get("name", "")),
            descriptors=tuple(str(d) for d in item.get("descriptors") or []),
            bundle_id=str(bundle),
            score=float(item.get("score") or 0.0),
        ))
    return tuple(matches)


@lru_cache(maxsize=1)
def get_tag_search() -> TagSearch | None:
    url = os.getenv("SPOT_TAG_SEARCH_URL", DEFAULT_URL).strip()
    return TagSearch(url, timeout=float(os.getenv("SPOT_TAG_SEARCH_TIMEOUT", "4"))) if url else None
