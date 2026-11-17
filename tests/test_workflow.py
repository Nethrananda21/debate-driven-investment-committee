import asyncio
import unittest
from unittest.mock import patch

import main
from agents.risk import RiskAgent
from retrieval import load_passages, retrieve


class WorkflowTest(unittest.TestCase):
    def test_retrieval_finds_debt_fact(self):
        passages = load_passages("documents/asteria")
        found = retrieve(passages, "net debt to EBITDA", limit=2)
        self.assertTrue(any("3.4x" in p["text"] for p in found))

    def test_cited_only_judge_and_risk_veto(self):
        seen = {}

        class Research:
            def __init__(self, service):
                pass

            async def run(self, asset, passages):
                return [p["id"] for p in passages]

        class Bull:
            def __init__(self, service):
                pass

            async def argue(self, asset, passages):
                return {"strongest_argument": "Growth", "argument": "Growth", "citations": ["S6"]}

            async def rebut(self, asset, passages, bear):
                return {"response": "Cash flow", "citations": ["S6"]}

        class Bear:
            def __init__(self, service):
                pass

            async def argue(self, asset, passages):
                return {"strongest_argument": "Debt", "argument": "Debt", "citations": ["S8"]}

            async def rebut(self, asset, passages, bull):
                return {"response": "Valuation", "citations": ["S8"]}

        class Judge:
            def __init__(self, service):
                pass

            async def run(self, asset, passages, bull, bear, bull_rebuttal, bear_rebuttal):
                seen["judge_ids"] = {p["id"] for p in passages}
                return {"recommendation": "BUY", "rationale": "Growth wins", "deciding_citations": ["S6"]}

        class Risk:
            def __init__(self, service, threshold):
                pass

            async def run(self, asset, passages):
                seen["risk_text"] = " ".join(p["text"] for p in passages)
                return {"veto": True, "value": 3.4, "threshold": 3.0, "source_id": "S8"}

        with patch.multiple(
            main,
            ResearchAgent=Research,
            BullAgent=Bull,
            BearAgent=Bear,
            JudgeAgent=Judge,
            RiskAgent=Risk,
        ):
            services = {role: object() for role in main.AGENT_MODELS}
            result = asyncio.run(main.committee("Asteria Systems", "documents/asteria", 3.0, services))

        self.assertEqual(seen["judge_ids"], {"S6", "S8"})
        self.assertIn("3.4x", seen["risk_text"])
        self.assertEqual(result["judge_recommendation"], "BUY")
        self.assertEqual(result["recommendation"], "HOLD")

    def test_risk_requires_cited_metric_and_value(self):
        source = {"id": "S8", "text": "At year end, net debt to EBITDA was 3.4x."}
        with patch("agents.risk.make_agent", return_value=object()), patch(
            "agents.risk.ask_json", return_value={"value": 3.4, "source_id": "S8"}
        ):
            result = asyncio.run(RiskAgent(object(), 3.0).run("Asteria", [source]))
        self.assertTrue(result["veto"])

        with patch("agents.risk.make_agent", return_value=object()), patch(
            "agents.risk.ask_json", return_value={"value": 4.9, "source_id": "S8"}
        ):
            result = asyncio.run(RiskAgent(object(), 3.0).run("Asteria", [source]))
        self.assertFalse(result["veto"])

        with patch("agents.risk.make_agent", return_value=object()), patch(
            "agents.risk.ask_json", return_value={"value": 3.4, "source_id": "S8"}
        ):
            result = asyncio.run(RiskAgent(object(), 4.0).run("Asteria", [source]))
        self.assertFalse(result["veto"])


if __name__ == "__main__":
    unittest.main()
