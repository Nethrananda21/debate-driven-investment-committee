from .common import ask_json, make_agent


class ResearchAgent:
    def __init__(self, service):
        self.agent = make_agent(
            service,
            "ResearchAgent",
            "Select the most decision-relevant source passages. Use only the supplied "
            'passage IDs. Return JSON only: {"source_ids":["S1","S2"]}. '
            "Select at most 8 passages; include upside, downside, and financial risk.",
        )

    async def run(self, asset, passages):
        prompt = f"Asset: {asset}\n\nRetrieved passages:\n{passages}"
        result = await ask_json(self.agent, prompt)
        available = {p["id"] for p in passages}
        return [x for x in result["source_ids"] if x in available][:8]
