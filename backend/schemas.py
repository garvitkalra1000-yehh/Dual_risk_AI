from __future__ import annotations

from datetime import date
from pydantic import BaseModel, Field


class InterventionIn(BaseModel):
    student_id: str
    intervention_type: str
    owner: str
    status: str = Field(default="Started")


class InterventionRecord(InterventionIn):
    start_date: date
    simulated_30_day_followup: str
    simulated_60_day_followup: str
