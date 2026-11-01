"""Pydantic request/response schemas for the API.

The applicant schema accepts the base fields a caller would realistically supply
for a new application; derived ratios are computed server-side. Extra keys (e.g.
``applicant_id`` from demo files) are ignored, not rejected.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class Applicant(BaseModel):
    model_config = ConfigDict(extra="ignore")

    AMT_INCOME_TOTAL: float = Field(..., gt=0, description="Annual income")
    AMT_CREDIT: float = Field(..., gt=0, description="Loan amount requested")
    AMT_ANNUITY: float = Field(..., gt=0, description="Loan annuity (installment)")
    AMT_GOODS_PRICE: float = Field(..., gt=0, description="Price of financed goods")
    AGE_YEARS: float = Field(..., ge=18, le=100)
    YEARS_EMPLOYED: float = Field(..., ge=0, le=60)
    CNT_CHILDREN: int = Field(0, ge=0, le=20)
    CNT_FAM_MEMBERS: float = Field(1, ge=1, le=20)
    EXT_SOURCE_1: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_2: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_3: Optional[float] = Field(None, ge=0, le=1)
    NAME_CONTRACT_TYPE: str = "Cash loans"
    NAME_EDUCATION_TYPE: str = "Secondary / secondary special"
    NAME_FAMILY_STATUS: str = "Married"
    NAME_INCOME_TYPE: str = "Working"
    NAME_HOUSING_TYPE: str = "House / apartment"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class PredictRequest(BaseModel):
    applicant: Applicant
    include_explanation: bool = True
    include_counterfactual: bool = True
    include_rag: bool = True
    include_llm: bool = True
    persist: bool = True


class ApplicantOrAudit(BaseModel):
    """Endpoints accept either a fresh applicant or an existing audit_id."""
    applicant: Optional[Applicant] = None
    audit_id: Optional[str] = None


class StressRequest(BaseModel):
    applicant: Optional[Applicant] = None
    audit_id: Optional[str] = None
    portfolio_sample: int = Field(0, ge=0, le=5000,
                                  description="If >0, also run a portfolio-level "
                                              "scenario analysis on this many "
                                              "rows from the test split.")


class RagQueryRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int = Field(4, ge=1, le=20)
