"""src/ticket_triage.py -- Part C: Support Ticket Triage with Guardrails
Implements:
1. Input guardrails (skipped_empty, flagged_long)
2. PII Redaction ([EMAIL], [PHONE], [ORDER_ID])
3. Zero-shot and Few-shot prompts
4. Resilient calling with exponential backoff and jitter
5. Answer validation (classified_json, classified_keyword, flagged_invalid_output)
6. Comparison between zero-shot and few-shot runs
7. Token budget enforcement demo and final routing based on customer segments
"""
import csv
import json
import os
import random
import re
import time
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import mock_llm

BASE_DELAY = 0.05
MAX_INPUT_TOKENS = 400

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")
ORDER_ID_PATTERN = re.compile(r"ORD-\d{6}")

ZERO_SHOT_PROMPT = """You are a customer support triage assistant. CLASSIFY the incoming ticket into exactly one of these four categories:
billing, delivery, technical, fraud.

Respond ONLY with a valid JSON object of the form:
{"category": "...", "reason": "..."}"""

FEW_SHOT_PROMPT = """You are a customer support triage assistant. CLASSIFY the incoming ticket into exactly one of these four categories:
billing, delivery, technical, fraud.

EXAMPLES:
Ticket: I was charged twice for my groceries.
{"category": "billing", "reason": "double charge"}

Ticket: The app freezes on checkout screen.
{"category": "technical", "reason": "app freeze"}

Ticket: Someone unauthorized placed an order from my account.
{"category": "fraud", "reason": "unauthorized account access"}

Ticket: Rider is very late and not picking up phone.
{"category": "delivery", "reason": "late delivery"}

Respond ONLY with a valid JSON object of the form:
{"category": "...", "reason": "..."}"""

def redact_pii(text):
    """Applies email, phone, and order id redaction in exact order."""
    num_email = len(EMAIL_PATTERN.findall(text))
    t1 = EMAIL_PATTERN.sub("[EMAIL]", text)

    num_phone = len(PHONE_PATTERN.findall(t1))
    t2 = PHONE_PATTERN.sub("[PHONE]", t1)

    num_order = len(ORDER_ID_PATTERN.findall(t2))
    redacted = ORDER_ID_PATTERN.sub("[ORDER_ID]", t2)

    return redacted, num_email, num_phone, num_order

def call_mock_with_retry(messages, ticket_id):
    """Wraps mock_llm.chat with exponential backoff and jitter up to 3 retries."""
    for n in range(4): # initial call (n=0 wait before retry 0) + up to 3 retries
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError as e:
            if n >= 3:
                raise e
            wait = BASE_DELAY * (2 ** n) * random.uniform(0.9, 1.15)
            print(f"Retry {n} for ticket {ticket_id}: waiting {wait:.4f}s ({e})")
            time.sleep(wait)

def validate_response(content):
    """Parses reply according to Task 5 validation rules."""
    allowed = ["billing", "delivery", "technical", "fraud"]
    try:
        data = json.loads(content)
        if isinstance(data, dict) and data.get("category") in allowed:
            return "classified_json", data["category"]
        else:
            return "flagged_invalid_output", None
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", (content or "").lower())
        matched = [c for c in allowed if c in words]
        if len(matched) == 1:
            return "classified_keyword", matched[0]
        return "flagged_invalid_output", None

def run_triage_pipeline(prompt_type="zero_shot", token_budget=None):
    """Runs the full triage pipeline over tickets.csv."""
    mock_llm.reset()
    random.seed(42)

    tickets_path = os.path.join("data", "raw", "tickets.csv")
    with open(tickets_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tickets = list(reader)

    system_prompt = FEW_SHOT_PROMPT if prompt_type == "few_shot" else ZERO_SHOT_PROMPT

    total_tokens_used = 0
    results = []

    for ticket in tickets:
        tid = ticket["ticket_id"]
        cid = ticket["customer_id"]
        raw_text = ticket["text"]
        true_cat = ticket["true_category"]

        # 1. Input Guardrails
        if not raw_text.strip():
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "skipped_empty",
                "category": "", "true_category": true_cat, "tokens_used": 0
            })
            continue

        est_toks = mock_llm.est_tokens(raw_text)
        if est_toks > MAX_INPUT_TOKENS:
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "flagged_long",
                "category": "", "true_category": true_cat, "tokens_used": 0
            })
            continue

        # 2. Redact PII
        redacted_text, _, _, _ = redact_pii(raw_text)

        # Assertion: no raw PII remains
        assert not EMAIL_PATTERN.findall(redacted_text), f"Email leaked in ticket {tid}!"
        assert not PHONE_PATTERN.findall(redacted_text), f"Phone leaked in ticket {tid}!"
        assert not ORDER_ID_PATTERN.findall(redacted_text), f"Order ID leaked in ticket {tid}!"

        # 3. Check token budget
        if token_budget is not None and total_tokens_used >= token_budget:
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "budget_exhausted",
                "category": "", "true_category": true_cat, "tokens_used": 0
            })
            continue

        # 4. Model Call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": redacted_text}
        ]

        resp = call_mock_with_retry(messages, tid)
        call_tokens = resp.usage.total_tokens
        total_tokens_used += call_tokens

        # 5. Validate Output
        content = resp.choices[0].message.content
        status, cat = validate_response(content)

        results.append({
            "ticket_id": tid,
            "customer_id": cid,
            "status": status,
            "category": cat or "",
            "true_category": true_cat,
            "tokens_used": call_tokens
        })

    return results, total_tokens_used

def evaluate_runs():
    print("=== PART C: SUPPORT TICKET TRIAGE ===")

    # 1. PII Redaction Audit across all 164 tickets
    tickets_path = os.path.join("data", "raw", "tickets.csv")
    with open(tickets_path, "r", encoding="utf-8") as f:
        tickets = list(csv.DictReader(f))

    tot_emails = tot_phones = tot_orders = 0
    for t in tickets:
        _, e, p, o = redact_pii(t["text"])
        tot_emails += e
        tot_phones += p
        tot_orders += o

    print(f"PII Redacted across 164 tickets:")
    print(f"  Emails: {tot_emails} (Expected: 55)")
    print(f"  Phones: {tot_phones} (Expected: 91)")
    print(f"  Order IDs: {tot_orders} (Expected: 132)")
    print(f"  Total Substitutions: {tot_emails + tot_phones + tot_orders} (Expected: 278)")
    assert tot_emails == 55 and tot_phones == 91 and tot_orders == 132

    # 2. Zero-Shot Run (No token budget)
    print("\n--- Running Zero-Shot Pipeline ---")
    res_zs, tok_zs = run_triage_pipeline(prompt_type="zero_shot", token_budget=None)
    df_zs = pd.DataFrame(res_zs)
    status_zs = df_zs["status"].value_counts().to_dict()
    print("Zero-Shot Status Breakdown:")
    for k, v in status_zs.items():
        print(f"  {k}: {v}")

    classified_zs = df_zs[df_zs["status"].isin(["classified_json", "classified_keyword"])].copy()
    acc_zs = (classified_zs["category"] == classified_zs["true_category"]).mean() * 100
    print(f"Zero-Shot Classified Tickets: {len(classified_zs)} (Expected: 147)")
    print(f"Zero-Shot Accuracy: {acc_zs:.1f}% (Expected: 85.7%)")
    print(f"Zero-Shot Total Tokens: {tok_zs}, Tokens/ticket: {tok_zs / len(classified_zs):.1f}")

    # Confusion matrix for zero-shot
    cm_zs = pd.crosstab(classified_zs["true_category"], classified_zs["category"], rownames=["True"], colnames=["Predicted"])
    print("\nZero-Shot Confusion Table:")
    print(cm_zs)

    # 3. Few-Shot Run (No token budget)
    print("\n--- Running Few-Shot Pipeline ---")
    res_fs, tok_fs = run_triage_pipeline(prompt_type="few_shot", token_budget=None)
    df_fs = pd.DataFrame(res_fs)
    classified_fs = df_fs[df_fs["status"].isin(["classified_json", "classified_keyword"])].copy()
    acc_fs = (classified_fs["category"] == classified_fs["true_category"]).mean() * 100
    print(f"Few-Shot Classified Tickets: {len(classified_fs)} (Expected: 147)")
    print(f"Few-Shot Accuracy: {acc_fs:.1f}% (Expected: 100.0%)")
    print(f"Few-Shot Total Tokens: {tok_fs}, Tokens/ticket: {tok_fs / len(classified_fs):.1f}")

    # 4. Token Budget Demo (Zero-shot with TOKEN_BUDGET = 6000)
    print("\n--- Running Budget Demo (TOKEN_BUDGET=6000) ---")
    res_demo, tok_demo = run_triage_pipeline(prompt_type="zero_shot", token_budget=6000)
    df_demo = pd.DataFrame(res_demo)
    exhausted_count = (df_demo["status"] == "budget_exhausted").sum()
    print(f"Budget Exhausted Tickets: {exhausted_count}")
    os.makedirs("outputs", exist_ok=True)
    df_demo.to_csv(os.path.join("outputs", "triage_budget_demo.csv"), index=False)

    # 5. Final Routing Output (Few-shot with TOKEN_BUDGET = 100000)
    print("\n--- Generating Final Triage Output with Routing ---")
    res_final, _ = run_triage_pipeline(prompt_type="few_shot", token_budget=100000)
    df_final = pd.DataFrame(res_final)

    # Read customer segments from Part B
    segments_path = os.path.join("outputs", "customer_segments.csv")
    if os.path.exists(segments_path):
        segments = pd.read_csv(segments_path)
        seg_map = dict(zip(segments["customer_id"], segments["segment_name"]))
        prem_map = dict(zip(segments["customer_id"], segments["is_premium"]))
    else:
        seg_map = {}
        prem_map = {}

    df_final["segment_name"] = df_final["customer_id"].map(seg_map).fillna("")
    df_final["is_premium"] = df_final["customer_id"].map(prem_map).fillna(False)

    def route_ticket(row):
        status = row["status"]
        cat = row["category"]
        is_prem = row["is_premium"]

        if status in ["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"]:
            return "NORMAL", "human_review"
        elif cat == "fraud":
            return "HIGH", "security_team"
        elif is_prem:
            return "HIGH", "premium_desk"
        else:
            return "NORMAL", "support_queue"

    routed = [route_ticket(r) for _, r in df_final.iterrows()]
    df_final["priority"] = [r[0] for r in routed]
    df_final["route"] = [r[1] for r in routed]

    # Clean unneeded category/segment_name for non-classified
    mask_blank = df_final["status"].isin(["skipped_empty", "flagged_long", "budget_exhausted"])
    df_final.loc[mask_blank, "category"] = ""
    df_final.loc[mask_blank, "segment_name"] = ""

    out_cols = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
    final_output = df_final[out_cols]
    final_csv = os.path.join("outputs", "triage_results.csv")
    final_output.to_csv(final_csv, index=False)
    print(f"Saved {final_csv} with {len(final_output)} rows.")

    print("\nRouting Summary:")
    print(final_output["route"].value_counts())
    print("\nPriority Summary:")
    print(final_output["priority"].value_counts())

if __name__ == "__main__":
    evaluate_runs()
