"""
Long-term industry thesis memory.

Persists sector/industry-level investment theses, catalysts, risks, and trends
across multiple analysis sessions. Each industry has its own JSON record that
accumulates knowledge over time, enabling the Industry Analyst to build on
prior observations rather than starting fresh each session.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class IndustryThesisMemory:
    """
    Persistent memory store for long-term industry/sector tracking.

    Stores per-industry records containing:
    - key_thesis: Core investment narrative for the sector
    - structural_tailwinds: Long-term growth drivers
    - structural_headwinds: Persistent risks and challenges
    - recent_catalysts: Near-term events/triggers (updated each session)
    - trend_log: Timestamped history of analyst observations
    - last_updated: Date of most recent update
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize memory store.

        Args:
            config: TradingAgents config dict (uses project_dir for storage path)
        """
        results_dir = config.get("results_dir", "./results")
        self.storage_dir = Path(results_dir) / "industry_memory"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_filepath(self, industry_key: str) -> Path:
        """Return JSON file path for a given industry key."""
        safe_key = industry_key.replace("/", "_").replace(" ", "_").lower()
        return self.storage_dir / f"{safe_key}.json"

    def _load(self, industry_key: str) -> Dict[str, Any]:
        """Load industry memory from disk. Returns empty structure if not found."""
        filepath = self._get_filepath(industry_key)
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "industry": industry_key,
            "key_thesis": "",
            "structural_tailwinds": [],
            "structural_headwinds": [],
            "recent_catalysts": [],
            "trend_log": [],
            "last_updated": None,
        }

    def _save(self, industry_key: str, data: Dict[str, Any]) -> None:
        """Persist industry memory to disk."""
        filepath = self._get_filepath(industry_key)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_thesis(self, industry_key: str) -> str:
        """
        Retrieve the stored thesis for an industry as a formatted string
        for injection into the analyst's prompt.

        Args:
            industry_key: Industry or sector name (e.g., "Semiconductors", "Technology")

        Returns:
            Formatted string summary of existing thesis, or empty string if none.
        """
        data = self._load(industry_key)

        if not data["last_updated"]:
            return ""

        tailwinds = "\n".join(f"  - {t}" for t in data["structural_tailwinds"]) or "  (none recorded)"
        headwinds = "\n".join(f"  - {h}" for h in data["structural_headwinds"]) or "  (none recorded)"
        catalysts = "\n".join(f"  - {c}" for c in data["recent_catalysts"][-5:]) or "  (none recorded)"

        # Recent trend log (last 5 entries)
        trend_log = data["trend_log"][-5:]
        trend_summary = "\n".join(
            f"  [{e['date']}] {e['observation']}"
            for e in trend_log
        ) or "  (no prior observations)"

        return f"""## Historical Industry Thesis: {industry_key}
Last Updated: {data['last_updated']}

### Core Thesis
{data['key_thesis'] or '(not yet established)'}

### Structural Tailwinds (Long-term Growth Drivers)
{tailwinds}

### Structural Headwinds (Persistent Risks)
{headwinds}

### Recent Catalysts (last 5)
{catalysts}

### Prior Analyst Observations
{trend_summary}
"""

    def update_thesis(
        self,
        industry_key: str,
        key_thesis: Optional[str] = None,
        structural_tailwinds: Optional[list] = None,
        structural_headwinds: Optional[list] = None,
        recent_catalysts: Optional[list] = None,
        observation: Optional[str] = None,
        trade_date: Optional[str] = None,
    ) -> None:
        """
        Update the stored thesis for an industry.

        Args:
            industry_key: Industry or sector name
            key_thesis: Updated core investment narrative
            structural_tailwinds: List of long-term growth drivers (replaces existing)
            structural_headwinds: List of persistent risks (replaces existing)
            recent_catalysts: List of near-term catalysts (replaces existing)
            observation: A brief observation to append to the trend log
            trade_date: Date of this update (yyyy-mm-dd). Defaults to today.
        """
        data = self._load(industry_key)
        today = trade_date or datetime.now().strftime("%Y-%m-%d")

        if key_thesis is not None:
            data["key_thesis"] = key_thesis
        if structural_tailwinds is not None:
            data["structural_tailwinds"] = structural_tailwinds
        if structural_headwinds is not None:
            data["structural_headwinds"] = structural_headwinds
        if recent_catalysts is not None:
            data["recent_catalysts"] = recent_catalysts
        if observation:
            data["trend_log"].append({
                "date": today,
                "observation": observation,
            })

        data["last_updated"] = today
        self._save(industry_key, data)

    def list_tracked_industries(self) -> list:
        """Return a list of all industry keys that have stored memory."""
        return [
            f.stem.replace("_", " ")
            for f in self.storage_dir.glob("*.json")
        ]
