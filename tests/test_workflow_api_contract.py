import pytest
from pydantic import ValidationError

from scripts.workflow_api import MandateSubmission


def _submission() -> dict:
    return {
        "name": "Northstar",
        "instruments": [
            {"name": f"Asset {index}", "ticker": f"T{index}", "group": "Equity"}
            for index in range(5)
        ],
        "maxWeight": 20,
        "riskPreference": "balanced",
        "approximateTradingCapital": 1_000_000,
    }


@pytest.mark.parametrize(
    "untrusted_field",
    ["id", "quoteAmount", "state", "stage", "submittedAt", "paymentState"],
)
def test_submission_rejects_server_owned_fields(untrusted_field: str) -> None:
    payload = _submission()
    payload[untrusted_field] = "client-controlled"
    with pytest.raises(ValidationError):
        MandateSubmission.model_validate(payload)


def test_submission_accepts_only_investor_mandate_inputs() -> None:
    parsed = MandateSubmission.model_validate(_submission())
    assert parsed.name == "Northstar"
    assert len(parsed.instruments) == 5
