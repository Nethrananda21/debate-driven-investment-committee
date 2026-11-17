from .common import ask_json, make_agent


class BullAgent:
    def __init__(self, service):
        self.agent = make_agent(
            service,
            "BullAgent",
            "Argue for BUY using only supplied source passages. Put [Sx] next to "
            "each factual claim and list those IDs in citations. Never describe a "
            "trend, forecast, or industry norm that the passages do not establish. "
            "One year of growth does not prove acceleration. Do not call a valuation "
            "attractive without a comparison in the sources. "
            "Return JSON only with keys "
            "strongest_argument, argument, citations (array of source IDs).",
        )

    async def argue(self, asset, passages):
        return await ask_json(self.agent, f"Asset: {asset}\nSources:\n{passages}")

    async def rebut(self, asset, passages, bear_case):
        return await ask_json(
            self.agent,
            f"Asset: {asset}\nSources:\n{passages}\nBear case:\n{bear_case}\n"
            "Respond once to the bear's strongest point. Explain why that point "
            "does not settle the decision, using supported counterevidence. "
            "Put [Sx] next to each factual claim. Return JSON only with "
            "keys response and citations (array of source IDs).",
        )
