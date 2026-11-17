from .common import ask_json, make_agent


class JudgeAgent:
    def __init__(self, service):
        self.agent = make_agent(
            service,
            "JudgeAgent",
            "Weigh both cases and both rebuttals. Use only the cited source "
            "passages supplied in this message. Disregard any unsupported claim. "
            "Put [Sx] next to each factual claim in the rationale. "
            "Return JSON only with recommendation (BUY, HOLD, or AVOID), "
            "rationale, and deciding_citations (array of source IDs).",
        )

    async def run(self, asset, cited_passages, bull, bear, bull_rebuttal, bear_rebuttal):
        return await ask_json(
            self.agent,
            f"Asset: {asset}\nCited evidence only:\n{cited_passages}\n"
            f"Bull: {bull}\nBear: {bear}\n"
            f"Bull rebuttal: {bull_rebuttal}\nBear rebuttal: {bear_rebuttal}",
        )
