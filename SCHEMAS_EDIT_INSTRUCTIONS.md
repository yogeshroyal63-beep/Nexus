# schemas.py — 2 manual edits needed

Apply these to the `backend/app/models/schemas.py` you already have from
`nexus-complete-latest4.zip`. Both are exact find-and-replace — the "find"
text is unique in the file.

---

## Edit 1 — near the top of the file, the typing import

FIND:
```python
from typing import Optional
```

REPLACE WITH:
```python
from typing import Literal, Optional
```

---

## Edit 2 — the `confidence` field on `RootCauseReport`

FIND:
```python
    confidence: str = Field(description="'low' | 'moderate' | 'high', based on intervention_delta magnitude & confounding")
```

REPLACE WITH:
```python
    confidence: Literal["low", "moderate", "high"] = Field(
        description=(
            "Based on intervention_delta magnitude & confounding. Tightened "
            "from an unconstrained str after finding the Planner's ONLY "
            "low-confidence safety gate is an exact string comparison "
            "(report.confidence == 'low' — see agents/planner.py). Every "
            "other enum-like field in this schema (RiskLevel, "
            "RemediationActionType, DriftSeverity, DriftMethod, NodeType) is "
            "a proper closed type; this was the one exception, relying "
            "entirely on llm/reasoning.py's _enforce_grounding to normalize "
            "case and fail closed to 'low' for anything off-vocabulary — a "
            "caller-level convention, not a type-system guarantee. Any "
            "future code path constructing a RootCauseReport directly, "
            "without going through that normalization, could have silently "
            "reintroduced the exact bypass already found and fixed once. "
            "A Literal here makes an invalid value impossible to construct "
            "at all — and pairs naturally with reasoning.py's own retry "
            "logic, which already treats a pydantic.ValidationError as "
            "retry-worthy rather than a hard crash."
        )
    )
```

---

## After applying

```bash
cd backend
python -m pytest tests/ -q
```

Should show **220 passed** (the 213 you already had, plus these 7 new
`test_schemas.py` tests once you also drop in the files from the recovery
zip).
