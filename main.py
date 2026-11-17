import argparse
import asyncio
import json
import os
import re
from time import perf_counter

from dotenv import load_dotenv
from openai import AsyncOpenAI
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

from agents.bear import BearAgent
from agents.bull import BullAgent
from agents.judge import JudgeAgent
from agents.research import ResearchAgent
from agents.risk import RiskAgent
from retrieval import load_passages, retrieve


AGENT_MODELS = {
    "research": ("nvidia", "nvidia/nemotron-3.5-bolt-e4b"),
    "bull": ("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b"),
    "bear": ("groq", "openai/gpt-oss-20b"),
    "judge": ("groq", "openai/gpt-oss-120b"),
    "risk": ("groq", "qwen/qwen3.8-27b"),
}


def cited(case, allowed):
    case["citations"] = list(dict.fromkeys(x for x in case["citations"] if x in allowed))
    if not case["citations"]:
        raise ValueError("A debate turn supplied no valid source citations")
    return case


async def committee(asset, folder, threshold, services, on_event=None):
    started = perf_counter()
    turn_times = {}

    async def turn(role, stage, task):
        if on_event:
            on_event(role, stage, None, 0)
        turn_started = perf_counter()
        answer = await task
        seconds = round(perf_counter() - turn_started, 2)
        turn_times[f"{role} {stage}"] = seconds
        if on_event:
            on_event(role, stage, answer, seconds)
        return answer

    passages = load_passages(folder)
    if not passages:
        raise ValueError("No .md, .txt, or .pdf asset documents found")
    retrieved = retrieve(passages, f"{asset} revenue growth earnings cash flow valuation risks debt EBITDA")
    research = ResearchAgent(services["research"])
    selected_ids = await turn("research", "Selected passages", research.run(asset, retrieved))
    evidence = [p for p in retrieved if p["id"] in selected_ids]
    if not evidence:
        raise ValueError("Research selected no valid passages")
    allowed = {p["id"] for p in evidence}

    bull_agent = BullAgent(services["bull"])
    bear_agent = BearAgent(services["bear"])
    bull, bear = await asyncio.gather(
        turn("bull", "Opening case", bull_agent.argue(asset, evidence)),
        turn("bear", "Opening case", bear_agent.argue(asset, evidence)),
    )
    bull, bear = cited(bull, allowed), cited(bear, allowed)
    bull_rebuttal, bear_rebuttal = await asyncio.gather(
        turn("bull", "Rebuttal", bull_agent.rebut(asset, evidence, bear)),
        turn("bear", "Rebuttal", bear_agent.rebut(asset, evidence, bull)),
    )
    bull_rebuttal = cited(bull_rebuttal, allowed)
    bear_rebuttal = cited(bear_rebuttal, allowed)

    used_ids = set().union(*(set(x["citations"]) for x in (bull, bear, bull_rebuttal, bear_rebuttal)))
    cited_evidence = [p for p in evidence if p["id"] in used_ids]
    judge = await turn(
        "judge", "Decision",
        JudgeAgent(services["judge"]).run(asset, cited_evidence, bull, bear, bull_rebuttal, bear_rebuttal),
    )
    judge["recommendation"] = judge["recommendation"].upper()
    if judge["recommendation"] not in {"BUY", "HOLD", "AVOID"}:
        raise ValueError("Judge gave an invalid recommendation")
    judge["deciding_citations"] = [x for x in judge["deciding_citations"] if x in used_ids]
    if not judge["deciding_citations"]:
        raise ValueError("Judge gave no valid deciding citations")

    risk_passages = [p for p in passages if re.search(r"net.debt.to.ebitda", p["text"], re.I)]
    if risk_passages:
        risk_passages = retrieve(risk_passages, f"{asset} latest net debt to EBITDA", limit=8)
    risk = await turn("risk", "Veto check", RiskAgent(services["risk"], threshold).run(asset, risk_passages))
    final = "HOLD" if risk["veto"] else judge["recommendation"]
    cited_sources = [p for p in passages if p["id"] in used_ids or p["id"] == risk["source_id"]]
    return {
        "asset": asset,
        "research_selected_ids": selected_ids,
        "total_seconds": round(perf_counter() - started, 2),
        "turn_seconds": turn_times,
        "agent_models": {role: model for role, (_, model) in AGENT_MODELS.items()},
        "agent_providers": {role: provider for role, (provider, _) in AGENT_MODELS.items()},
        "recommendation": final,
        "judge_recommendation": judge["recommendation"],
        "bull_strongest_argument": bull["strongest_argument"],
        "bear_strongest_argument": bear["strongest_argument"],
        "bull_case": bull,
        "bear_case": bear,
        "bull_rebuttal": bull_rebuttal,
        "bear_rebuttal": bear_rebuttal,
        "judge_rationale": judge["rationale"],
        "deciding_citations": judge["deciding_citations"],
        "risk": risk,
        "sources": cited_sources,
    }


def create_services():
    load_dotenv()
    nvidia_key, groq_key = os.getenv("NVIDIA_API_KEY"), os.getenv("GROQ_API_KEY")
    if not nvidia_key or not groq_key:
        raise ValueError("Set NVIDIA_API_KEY and GROQ_API_KEY in the .env file")
    clients = {
        "nvidia": AsyncOpenAI(api_key=nvidia_key, base_url="https://integrate.api.nvidia.com/v1"),
        "groq": AsyncOpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1"),
    }
    return {
        role: OpenAIChatCompletion(ai_model_id=model, async_client=clients[provider])
        for role, (provider, model) in AGENT_MODELS.items()
    }


def main():
    parser = argparse.ArgumentParser(description="Debate-driven investment committee")
    parser.add_argument("asset", help="Asset name")
    parser.add_argument("documents", help="Folder containing .md, .txt, or .pdf documents")
    parser.add_argument("--max-net-debt-ebitda", type=float, default=3.0)
    args = parser.parse_args()
    services = create_services()
    result = asyncio.run(committee(args.asset, args.documents, args.max_net_debt_ebitda, services))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
