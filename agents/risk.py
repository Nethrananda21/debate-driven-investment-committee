import re

from .common import ask_json, make_agent


class RiskAgent:
    def __init__(self, service, max_net_debt_to_ebitda):
        self.agent = make_agent(
            service,
            "RiskAgent",
            "Find a stated net debt to EBITDA ratio in the supplied sources. "
            'Return JSON only: {"value": number or null, "source_id": "S1" or null}. '
            "Do not infer a ratio from other figures.",
        )
        self.threshold = max_net_debt_to_ebitda

    async def run(self, asset, passages):
        result = (
            await ask_json(self.agent, f"Asset: {asset}\nSources:\n{passages}")
            if passages else {"value": None, "source_id": None}
        )
        source = next((p for p in passages if p["id"] == result["source_id"]), None)
        value = result["value"]
        # The cited passage must explicitly contain both the metric and the value.
        verified = bool(
            source
            and value is not None
            and re.search(r"net.debt.to.EBITDA", source["text"], re.I)
            and re.search(rf"(?<!\d){re.escape(str(value))}(?!\d)", source["text"])
        )
        return {
            "metric": "net debt to EBITDA",
            "value": value if verified else None,
            "threshold": self.threshold,
            "source_id": source["id"] if verified else None,
            "veto": bool(verified and float(value) > self.threshold),
        }
