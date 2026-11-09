import copy
import csv
import json
import os
import re
import sqlite3
import time
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datetime import datetime, timezone
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

POLICIES_DIR = os.path.join("data", "policies")
STATE_DIR = "state"
OUTPUTS_DIR = "outputs"

os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def chunk_text(text, chunk_size=40, overlap=10):
    words = text.split()
    if not words:
        return []
    step = chunk_size - overlap
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
        i += step
    return chunks

def build_corpus(overlap=10):
    files = ["refund_policy.txt", "delivery_policy.txt", "payment_policy.txt",
             "privacy_policy.txt", "escalation_policy.txt"]
    corpus = []
    for fn in files:
        fpath = os.path.join(POLICIES_DIR, fn)
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = chunk_text(content, chunk_size=40, overlap=overlap)
        for c in chunks:
            corpus.append({"source": fn, "text": c})
    return corpus

class PolicyRetriever:
    def __init__(self, overlap=10):
        self.corpus = build_corpus(overlap=overlap)
        self.texts = [c["text"] for c in self.corpus]
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query, k=3):
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.doc_vectors)[0]
        ranked_indices = sims.argsort()[::-1][:k]
        results = []
        for idx in ranked_indices:
            results.append({
                "source": self.corpus[idx]["source"],
                "score": float(sims[idx]),
                "text": self.corpus[idx]["text"]
            })
        return results

retriever = PolicyRetriever(overlap=10)

def answer(question, retr=None):
    if retr is None:
        retr = retriever
    chunks = retr.retrieve(question, k=3)
    sources = [c["source"] for c in chunks]

    context_lines = []
    for c in chunks:
        context_lines.append(f"[{c['source']}]\n{c['text']}")
    context_block = "\n".join(context_lines)

    user_msg = f"CONTEXT:\n{context_block}\nQUESTION:\n{question}"
    messages = [
        {"role": "system", "content": "You are a POLICY ASSISTANT. Answer the question using ONLY the provided context. If the answer cannot be found in the context, say you could not find the information."},
        {"role": "user", "content": user_msg}
    ]
    resp = mock_llm.chat(messages)
    return resp.choices[0].message.content, sources

def evaluate_rag():
    print("=== PART D: RETRIEVAL & GROUNDING EVALUATION ===")
    eval_csv = os.path.join("data", "raw", "policy_eval.csv")
    with open(eval_csv, "r", encoding="utf-8") as f:
        qa_pairs = list(csv.DictReader(f))

    print("\n--- Evaluation with Overlap = 10 ---")
    corpus_10 = build_corpus(overlap=10)
    counts = {}
    for c in corpus_10:
        counts[c["source"]] = counts.get(c["source"], 0) + 1
    print(f"Total chunks: {len(corpus_10)} (Expected: 15)")
    for fn, count in counts.items():
        print(f"  {fn}: {count}")

    doc_matches = 0
    fact_matches = 0
    fallback_matches = 0

    for qa in qa_pairs:
        q = qa["question"]
        exp_doc = qa["expected_doc"]
        exp_fact = qa["expected_fact"]

        ans, sources = answer(q, retriever)

        if exp_doc != "NONE":
            if exp_doc in sources:
                doc_matches += 1
            if exp_fact in ans:
                fact_matches += 1
        else:
            if ans.startswith("I could not find"):
                fallback_matches += 1

    print(f"Answerable Expected Doc Match: {doc_matches}/12 (Expected: 12/12)")
    print(f"Answerable Expected Fact Match: {fact_matches}/12 (Expected: 12/12)")
    print(f"Unanswerable Fallback Match: {fallback_matches}/2 (Expected: 2/2)")

    print("\n--- Evaluation with Zero Overlap (Overlap = 0) ---")
    retriever_0 = PolicyRetriever(overlap=0)
    broken_questions = []

    for qa in qa_pairs:
        q = qa["question"]
        exp_doc = qa["expected_doc"]
        exp_fact = qa["expected_fact"]

        ans, sources = answer(q, retriever_0)
        if exp_doc != "NONE":
            if exp_doc not in sources or exp_fact not in ans:
                broken_questions.append((q, exp_doc, exp_fact, ans))

    print(f"Failed questions under zero-overlap: {len(broken_questions)}")
    for bq in broken_questions:
        print(f"  Question: '{bq[0]}'")
        print(f"  Expected Fact: '{bq[2]}'")
        print(f"  Model Ans: '{bq[3]}'")

import shutil
import tempfile

def get_order(order_id):
    db_path = os.path.join("data", "instamart.db")
    try:
        temp_dir = tempfile.gettempdir()
        temp_db = os.path.join(temp_dir, f"instamart_read_{os.getpid()}.db")
        shutil.copy2(db_path, temp_db)
        conn = sqlite3.connect(temp_db)
    except Exception:
        conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return {"error": "order not found"}
    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": float(row[2]),
        "delivery_minutes": int(row[3]) if row[3] is not None else ""
    }

def search_policy(query):
    results = retriever.retrieve(query, k=3)
    return {"results": results}

def calculate_refund(order_amount, delivery_minutes):
    try:
        deliv = int(delivery_minutes)
    except (ValueError, TypeError):
        deliv = 0
    if deliv > 45:
        return {"refund_inr": float(order_amount)}
    else:
        return {"refund_inr": 0.0}

def issue_refund(order_id, refund_inr):
    ledger_path = os.path.join(OUTPUTS_DIR, "refund_ledger.jsonl")
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": "issue_refund",
        "order_id": order_id,
        "refund_inr": float(refund_inr)
    }
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"refund_inr": float(refund_inr)}

def escalate_to_human(order_id, refund_inr):
    ledger_path = os.path.join(OUTPUTS_DIR, "refund_ledger.jsonl")
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": "escalate_to_human",
        "order_id": order_id,
        "refund_inr": float(refund_inr)
    }
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"refund_inr": float(refund_inr)}

TOOLS_SPEC = [
    {"type": "function", "function": {"name": "get_order", "description": "Fetches order details from the database.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "search_policy", "description": "Searches store policy documents for relevant rules.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "calculate_refund", "description": "Calculates refund eligibility based on order amount and delivery duration.", "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "integer"}}, "required": ["order_amount", "delivery_minutes"]}}},
    {"type": "function", "function": {"name": "issue_refund", "description": "Issues automatic refund to customer.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}}},
    {"type": "function", "function": {"name": "escalate_to_human", "description": "Escalates high-value refund to a human support lead.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}}}
]

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

def save_checkpoint(run_id, scenario_id, messages, tokens_used, step, tools_executed):
    state_file = os.path.join(STATE_DIR, f"{run_id}.json")
    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step": step,
        "tools_executed": tools_executed
    }
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, run_id=None, crash_after_tool_count=None):
    if run_id is None:
        run_id = f"run_{scenario_id}_{int(time.time()*1000)}"

    start_time = time.monotonic()
    messages = [
        {"role": "system", "content": "You are a customer support AGENT. Use tools to check order details, search policies, and process refunds."},
        {"role": "user", "content": user_message}
    ]

    tokens_used = 0
    llm_calls = 0
    steps = []
    tools_executed = []

    for it in range(max_iterations + 1):
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            return {
                "scenario_id": scenario_id,
                "status": "FALLBACK_TIMEOUT",
                "final_answer": "FALLBACK: Execution timed out. Handoff to human agent.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used
            }

        if it >= max_iterations:
            return {
                "scenario_id": scenario_id,
                "status": "FALLBACK_MAX_ITERATIONS",
                "final_answer": "FALLBACK: Maximum iterations reached. Handoff to human agent.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used
            }

        resp = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        call_tokens = resp.usage.total_tokens
        tokens_used += call_tokens

        if tokens_used > token_limit:
            return {
                "scenario_id": scenario_id,
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": "FALLBACK: Token budget exceeded. Handoff to human agent.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used
            }

        choice = resp.choices[0]
        tool_calls = getattr(choice.message, "tool_calls", None)
        if not tool_calls:
            final_content = choice.message.content or ""
            if ":" in final_content:
                status = final_content.split(":")[0].strip()
            else:
                status = final_content.strip()

            return {
                "scenario_id": scenario_id,
                "status": status,
                "final_answer": final_content,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used
            }

        for tc in tool_calls:
            tc_id = tc.id
            fn_name = tc.name
            fn_args = tc.arguments
            if isinstance(fn_args, str):
                fn_args = json.loads(fn_args)

            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": tc_id, "name": fn_name, "arguments": fn_args}]
            })

            func = TOOL_MAP[fn_name]
            obs = func(**fn_args)
            obs_json = json.dumps(obs)
            tools_executed.append(fn_name)

            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "name": fn_name,
                "content": obs_json
            })

            steps.append({
                "tool": fn_name,
                "arguments": fn_args,
                "observation": obs
            })

            save_checkpoint(run_id, scenario_id, messages, tokens_used, len(steps), tools_executed)

            if crash_after_tool_count is not None and len(tools_executed) == crash_after_tool_count:
                raise RuntimeError(f"SIMULATED CRASH after {len(tools_executed)} tools executed (run_id={run_id})")

def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    state_file = os.path.join(STATE_DIR, f"{run_id}.json")
    with open(state_file, "r", encoding="utf-8") as f:
        ckpt = json.load(f)

    scenario_id = ckpt["scenario_id"]
    messages = ckpt["messages"]
    tokens_used = ckpt["tokens_used"]
    tools_executed = ckpt["tools_executed"]
    steps = [{"tool": t, "arguments": {}, "observation": {}} for t in tools_executed]

    llm_calls = len(tools_executed)
    start_time = time.monotonic()

    for it in range(len(tools_executed), max_iterations + 1):
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            return {
                "scenario_id": scenario_id,
                "status": "FALLBACK_TIMEOUT",
                "final_answer": "FALLBACK: Execution timed out after resume.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed
            }

        if it >= max_iterations:
            return {
                "scenario_id": scenario_id,
                "status": "FALLBACK_MAX_ITERATIONS",
                "final_answer": "FALLBACK: Maximum iterations reached after resume.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed
            }

        resp = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += resp.usage.total_tokens

        choice = resp.choices[0]
        tool_calls = getattr(choice.message, "tool_calls", None)
        if not tool_calls:
            final_content = choice.message.content or ""
            status = final_content.split(":")[0].strip() if ":" in final_content else final_content.strip()
            return {
                "scenario_id": scenario_id,
                "status": status,
                "final_answer": final_content,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed
            }

        for tc in tool_calls:
            tc_id = tc.id
            fn_name = tc.name
            fn_args = tc.arguments
            if isinstance(fn_args, str):
                fn_args = json.loads(fn_args)

            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": tc_id, "name": fn_name, "arguments": fn_args}]
            })

            func = TOOL_MAP[fn_name]
            obs = func(**fn_args)
            tools_executed.append(fn_name)

            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "name": fn_name,
                "content": json.dumps(obs)
            })

            steps.append({"tool": fn_name, "arguments": fn_args, "observation": obs})
            save_checkpoint(run_id, scenario_id, messages, tokens_used, len(steps), tools_executed)

def run_agent_scenarios():
    print("\n=== RUNNING 7 AGENT SCENARIOS ===")
    scenarios_csv = os.path.join("data", "raw", "agent_scenarios.csv")
    with open(scenarios_csv, "r", encoding="utf-8") as f:
        scenarios = list(csv.DictReader(f))

    agent_runs_path = os.path.join(OUTPUTS_DIR, "agent_runs.jsonl")
    if os.path.exists(agent_runs_path):
        os.remove(agent_runs_path)

    results_table = []

    for sc in scenarios:
        sid = sc["scenario_id"]
        cid = sc["customer_id"]
        msg = sc["message"]
        exp_oc = sc["expected_outcome"]
        exp_amt = float(sc["expected_refund_inr"])

        res = run_agent(msg, sid, max_iterations=6, token_limit=6000, time_limit_seconds=10)

        refund_amt = 0.0
        for st in res["steps"]:
            if st["tool"] in ["issue_refund", "escalate_to_human"]:
                refund_amt = float(st["arguments"].get("refund_inr", 0.0))
        if refund_amt == 0.0:
            m = re.search(r"INR\s+([\d.]+)", res["final_answer"])
            if m:
                refund_amt = float(m.group(1))

        results_table.append({
            "scenario_id": sid,
            "status": res["status"],
            "expected_outcome": exp_oc,
            "refund_amount": refund_amt,
            "expected_refund": exp_amt,
            "llm_calls": res["llm_calls"],
            "tokens_used": res["tokens_used"]
        })

        run_record = {
            "scenario_id": sid,
            "status": res["status"],
            "final_answer": res["final_answer"],
            "llm_calls": res["llm_calls"],
            "tokens_used": res["tokens_used"],
            "steps": res["steps"]
        }
        with open(agent_runs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(run_record) + "\n")

    print("\nSeven Scenario Results Table:")
    df_res = pd.DataFrame(results_table)
    print(df_res[["scenario_id", "status", "refund_amount", "llm_calls", "tokens_used"]])

    print("\n--- Safety Limit Demonstrations ---")
    s1_msg = scenarios[0]["message"]

    res_tok_demo = run_agent(s1_msg, "S1_TOKEN_DEMO", token_limit=200)
    print(f"Token Limit Demo Status: {res_tok_demo['status']} (Expected: FALLBACK_TOKEN_LIMIT)")
    assert res_tok_demo["status"] == "FALLBACK_TOKEN_LIMIT"

    res_time_demo = run_agent(s1_msg, "S1_TIME_DEMO", time_limit_seconds=0)
    print(f"Timeout Demo Status: {res_time_demo['status']} (Expected: FALLBACK_TIMEOUT)")
    assert res_time_demo["status"] == "FALLBACK_TIMEOUT"

    print("\n--- Resumable Run Demonstration (S1 Crash & Resume) ---")
    crash_run_id = "s1_crash_demo"
    try:
        run_agent(s1_msg, "S1", run_id=crash_run_id, crash_after_tool_count=2)
    except RuntimeError as e:
        print(f"Caught simulated crash: {e}")

    resumed_res = resume_agent(crash_run_id)
    print(f"Resumed Run Final Status: {resumed_res['status']}, Final Answer: {resumed_res['final_answer']}")
    print(f"Tools Executed Across Crash & Resume: {resumed_res['tools_executed']}")
    assert resumed_res["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]

if __name__ == "__main__":
    evaluate_rag()
    run_agent_scenarios()
