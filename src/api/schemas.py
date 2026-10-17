"""Pydantic request/response models, built from the same column lists the model uses (no drift)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Month = Literal["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VisitorType = Literal["Returning_Visitor", "New_Visitor", "Other"]


class SessionIn(BaseModel):
    """One shopping session as observed so far. PageValues is deliberately NOT accepted (see model card)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "Administrative": 2, "Administrative_Duration": 40.0, "Informational": 0, "Informational_Duration": 0.0,
        "ProductRelated": 25, "ProductRelated_Duration": 900.0, "BounceRates": 0.01, "ExitRates": 0.03,
        "SpecialDay": 0.0, "Month": "Nov", "OperatingSystems": 2, "Browser": 2, "Region": 1, "TrafficType": 2,
        "VisitorType": "Returning_Visitor", "Weekend": False}})

    Administrative: int = Field(ge=0, le=200)
    Administrative_Duration: float = Field(ge=0)
    Informational: int = Field(ge=0, le=200)
    Informational_Duration: float = Field(ge=0)
    ProductRelated: int = Field(ge=0, le=2000)
    ProductRelated_Duration: float = Field(ge=0)
    BounceRates: float = Field(ge=0, le=1)
    ExitRates: float = Field(ge=0, le=1)
    SpecialDay: float = Field(ge=0, le=1)
    Month: Month
    OperatingSystems: int = Field(ge=1)
    Browser: int = Field(ge=1)
    Region: int = Field(ge=1)
    TrafficType: int = Field(ge=1)
    VisitorType: VisitorType
    Weekend: bool


class Contributor(BaseModel):
    feature: str
    value: float | int | str | bool
    effect: float = Field(description="Change in purchase probability attributed to this feature (Shapley, probability units)")
    direction: Literal["raises", "lowers"]


class PredictionOut(BaseModel):
    probability: float
    prediction: int
    decision: Literal["high_intent", "low_intent"]
    threshold: float
    confidence: Literal["high", "medium", "low"]
    model: str | None = None
    baseline_probability: float | None = None
    top_contributors: list[Contributor] = []


class HealthOut(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model: str | None = None
