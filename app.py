"""Simple live dashboard for the investment committee."""

import asyncio
import json
from pathlib import Path

import streamlit as st

from main import AGENT_MODELS, committee, create_services

OUTPUT_FILE = Path(__file__).parent / "output.json"


def display_answer(role, stage, answer):
    if role == "research":
        return "Selected source passages: " + ", ".join(answer)
    if role == "risk":
        value = "not stated" if answer["value"] is None else f'{answer["value"]}x'
        verdict = "VETO → HOLD" if answer["veto"] else "No veto"
        return f'**{verdict}**\n\nNet debt to EBITDA: {value}  \nLimit: {answer["threshold"]}x  \nSource: {answer["source_id"]}'
    if role == "judge":
        return f'**{answer["recommendation"]}**\n\n{answer["rationale"]}\n\nCitations: {", ".join(answer["deciding_citations"])}'
    if stage == "Opening case":
        return f'**Strongest point:** {answer["strongest_argument"]}\n\n{answer["argument"]}\n\nCitations: {", ".join(answer["citations"])}'
    return f'{answer["response"]}\n\nCitations: {", ".join(answer["citations"])}'


st.set_page_config(page_title="Investment Committee", layout="wide")
st.title("Debate-Driven Investment Committee")
st.caption("Watch each agent's turn as it completes. The example asset documents are fictional.")

asset = st.text_input("Asset name", "Asteria Systems")
folder = st.text_input("Document folder", str(Path(__file__).parent / "documents" / "asteria"))
threshold = st.number_input("Maximum net debt to EBITDA", min_value=0.0, value=3.0, step=0.1)
run = st.button("Run debate", type="primary")
result = None if run or not OUTPUT_FILE.exists() else json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
history = {role: [] for role in AGENT_MODELS}

if result:
    turns = {
        "research": [("Selected passages", result["research_selected_ids"])],
        "bull": [("Opening case", result["bull_case"]), ("Rebuttal", result["bull_rebuttal"])],
        "bear": [("Opening case", result["bear_case"]), ("Rebuttal", result["bear_rebuttal"])],
        "judge": [("Decision", {"recommendation": result["judge_recommendation"], "rationale": result["judge_rationale"], "deciding_citations": result["deciding_citations"]})],
        "risk": [("Veto check", result["risk"])],
    }
    for role, entries in turns.items():
        history[role] = [f"**{stage}**\n\n{display_answer(role, stage, answer)}" for stage, answer in entries]

boxes = {}
columns = [*st.columns(3), *st.columns(2)]
for (role, (provider, model)), column in zip(AGENT_MODELS.items(), columns):
    with column:
        with st.container(border=True):
            st.subheader(f"{role.title()} Agent")
            st.caption(f"{provider.upper()} · {model}")
            boxes[role] = (st.empty(), st.empty())
            boxes[role][0].markdown("✅ Last completed run" if result else "Waiting")
            boxes[role][1].markdown("\n\n---\n\n".join(history[role]))

if run:
    def update(role, stage, answer, seconds):
        status, body = boxes[role]
        if answer is None:
            status.markdown(f"⏳ Thinking: {stage}")
        else:
            status.markdown(f"✅ {stage} complete · {seconds:.1f}s")
            history[role].append(f"**{stage}**\n\n{display_answer(role, stage, answer)}")
            body.markdown("\n\n---\n\n".join(history[role]))

    try:
        result = asyncio.run(committee(asset, folder, threshold, create_services(), update))
        OUTPUT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception as error:
        st.error(str(error))

if result:
    if not run:
        st.caption("Showing the last completed run. Click Run debate for a fresh result.")
    st.success(f'Final recommendation: {result["recommendation"]}')
    st.write(f'Judge: {result["judge_recommendation"]} · Risk veto: {result["risk"]["veto"]} · Total: {result.get("total_seconds", "—")}s')
    with st.expander("Cited source passages"):
        for source in result["sources"]:
            st.markdown(f'**{source["id"]} — {source["file"]}**  \n{source["text"]}')
    st.download_button("Download full JSON", json.dumps(result, indent=2), "recommendation.json")
