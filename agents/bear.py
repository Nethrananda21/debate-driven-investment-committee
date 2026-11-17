from .common import ask_json, make_agent


class BearAgent:
    def __init__(self, service):
        self.agent = make_agent(
            service,
            "BearAgent",
            "Argue for AVOID using only supplied source passages. Put [Sx] next to "
            "each factual claim and list those IDs in citations. Never describe a "
            "trend, forecast, or industry norm that the passages do not establish. "
            "Lead with the strongest concrete downside in the supplied passages, "
            "not a generic statement that information is incomplete. "
            "Return JSON only with keys "
            "strongest_argument, argument, citations (array of source IDs).",
        )

    async def argue(self, asset, passages):
        return await ask_json(self.agent, f"Asset: {asset}\nSources:\n{passages}")

    async def rebut(self, asset, passages, bull_case):
        return await ask_json(
            self.agent,
            f"Asset: {asset}\nSources:\n{passages}\nBull case:\n{bull_case}\n"
            "Respond once to the bull's strongest point. Explain why it does not "
            "settle the decision, using supported counterevidence. "
            "Put [Sx] next to each factual claim. Return JSON only with "
            "keys response and citations (array of source IDs).",
        )
