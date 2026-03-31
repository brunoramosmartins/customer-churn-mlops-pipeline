"""Pandera schema for the IBM / Kaggle Telco Customer Churn CSV (raw layout)."""

from __future__ import annotations

import pandera.pandas as pa

# Typical filename from Kaggle / IBM sample
EXPECTED_RAW_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

_YES_NO = ("Yes", "No")
_GENDER = ("Male", "Female")
_MULTIPLE_LINES = ("Yes", "No", "No phone service")
_INTERNET = ("DSL", "Fiber optic", "No")
_TRIPLE_SERVICE = ("Yes", "No", "No internet service")
_CONTRACT = ("Month-to-month", "One year", "Two year")
_PAYMENT = (
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
)


def telco_raw_schema() -> pa.DataFrameSchema:
    """Schema matching `pd.read_csv` on the standard Telco file (no dtype overrides)."""
    c = pa.Check
    col = pa.Column
    return pa.DataFrameSchema(
        {
            "customerID": col(str),
            "gender": col(str, c.isin(_GENDER)),
            "SeniorCitizen": col(int, c.isin((0, 1))),
            "Partner": col(str, c.isin(_YES_NO)),
            "Dependents": col(str, c.isin(_YES_NO)),
            "tenure": col(int, c.ge(0)),
            "PhoneService": col(str, c.isin(_YES_NO)),
            "MultipleLines": col(str, c.isin(_MULTIPLE_LINES)),
            "InternetService": col(str, c.isin(_INTERNET)),
            "OnlineSecurity": col(str, c.isin(_TRIPLE_SERVICE)),
            "OnlineBackup": col(str, c.isin(_TRIPLE_SERVICE)),
            "DeviceProtection": col(str, c.isin(_TRIPLE_SERVICE)),
            "TechSupport": col(str, c.isin(_TRIPLE_SERVICE)),
            "StreamingTV": col(str, c.isin(_TRIPLE_SERVICE)),
            "StreamingMovies": col(str, c.isin(_TRIPLE_SERVICE)),
            "Contract": col(str, c.isin(_CONTRACT)),
            "PaperlessBilling": col(str, c.isin(_YES_NO)),
            "PaymentMethod": col(str, c.isin(_PAYMENT)),
            "MonthlyCharges": col(float, checks=c.ge(0)),
            "TotalCharges": col(float, nullable=True, checks=c.ge(0)),
            "Churn": col(str, c.isin(_YES_NO)),
        },
        strict=True,
        coerce=False,
    )
