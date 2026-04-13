"""Helper script: viết eval.py với nội dung mới."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

EVAL_CONTENT = r'''"""
eval.py — Sprint 4: Evaluation & Scorecard
==========================================
Mục tiêu Sprint 4:
  - Chạy 10 test questions qua pipeline
  - Chấm điểm 4 metrics: Faithfulness, Relevance, Context Recall, Completeness
  - So sánh baseline vs variant (A/B)
  - RAGAS 0.4.x evaluation với gpt-4o-mini làm LLM judge

A/B Rule: Chỉ đổi MỘT biến mỗi lần.

RAGAS vs Simple eval:
  Simple eval  -> debug, tiết kiệm token, prompt tiếng Việt tuỳ chỉnh được
  RAGAS 0.4.x  -> benchmark chuẩn hoá, cite paper (Es et al. 2023), gpt-4o-mini judge

Kết luận so sánh RAGAS vs Simple:
  ┌──────────────────┬────────────────────────┬────────────────────────┐
  │ Tiêu chí         │ Simple (run_scorecard) │ RAGAS 0.4.x            │
  ├──────────────────┼────────────────────────┼────────────────────────┤
  │ Context Recall   │ Source-match (fast)    │ NLI/LLM-based          │
  │ Faithfulness     │ LLM-as-Judge tự viết  │ LLM chuẩn (paper)      │
  │ Answer Relevance │ LLM-as-Judge tự viết  │ LLM chuẩn (paper)      │
  │ Completeness     │ LLM-as-Judge tự viết  │ answer_correctness     │
  │ Chi phí LLM      │ 3 calls / câu         │ 6-8 calls / câu        │
  │ Tiếng Việt       │ Prompt tuỳ chỉnh      │ Cần config LLM         │
  │ Debug            │ Dễ trace từng bước    │ Black-box              │
  │ Chuẩn hoá        │ Custom                │ Chuẩn ngành (paper)    │
  └──────────────────┴────────────────────────┴────────────────────────┘
"""

import json
import csv
import re
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

from rag_answer import rag_answer

# =============================================================================
# CONFIG
# =============================================================================

TEST_QUESTIONS_PATH = Path(__file__).parent / "data" / "test_questions.json"
RESULTS_DIR = Path(__file__).parent / "results"
LOGS_DIR    = Path(__file__).parent / "logs"
LLM_JUDGE_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

BASELINE_CONFIG = {
    "retrieval_mode": "dense",
    "top_k_search": 10, "top_k_select": 3,
    "use_rerank": False, "label": "baseline_dense",
}
VARIANT_CONFIG = {
    "retrieval_mode": "hybrid",
    "top_k_search": 10, "top_k_select": 3,
    "use_rerank": True, "label": "variant_hybrid_rerank",
}


# =============================================================================
# LLM JUDGE HELPER — gpt-4o-mini
# =============================================================================

def _call_judge_llm(prompt: str, max_tokens: int = 300) -> str:
    """Gọi gpt-4o-mini để chấm điểm (LLM-as-Judge)."""
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GOOGLE_API_KEY")
    if openai_key:
        from openai import OpenAI
        r = OpenAI(api_key=openai_key).chat.completions.create(
            model=LLM_JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=max_tokens,
        )
        return r.choices[0].message.content
    if gemini_key:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        return genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt).text
    raise RuntimeError("Cần OPENAI_API_KEY hoặc GOOGLE_API_KEY để dùng LLM-as-Judge")


def _parse_judge(raw: str) -> Dict[str, Any]:
    raw = re.sub(r"```json\s*|```\s*", "", raw).strip()
    m = re.search(r"\{.*?\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    nums = re.findall(r'"score"\s*:\s*([1-5])', raw)
    if nums:
        return {"score": int(nums[0]), "reason": raw[:100]}
    return {"score": None, "reason": f"parse error: {raw[:80]}"}


# =============================================================================
# SCORING FUNCTIONS — LLM-as-Judge (gpt-4o-mini)
# =============================================================================

def score_faithfulness(answer: str, chunks_used: List[Dict]) -> Dict:
    """Faithfulness 1-5: answer có grounded trong retrieved context không?"""
    if not chunks_used:
        return {"score": 1, "notes": "No context retrieved"}
    ctx = "\n\n".join(
        f"[{i+1}] {c.get('text', '')[:400]}"
        for i, c in enumerate(chunks_used[:3])
    )
    prompt = f"""Rate FAITHFULNESS of the answer given retrieved context.

Context:
{ctx}

Answer: {answer}

Scale 1-5:
5 = every claim grounded in context (correct abstain = 5 too)
4 = mostly grounded, 1 minor uncertainty
3 = partial grounding, some model knowledge used
2 = several claims NOT in context
1 = mostly hallucinated

Output JSON only: {{"score":<1-5>,"reason":"<brief>"}}"""
    try:
        d = _parse_judge(_call_judge_llm(prompt))
        return {"score": d.get("score"), "notes": d.get("reason", "")}
    except Exception as e:
        return {"score": None, "notes": f"judge error: {e}"}


def score_answer_relevance(query: str, answer: str) -> Dict:
    """Answer Relevance 1-5: answer có trả lời đúng câu hỏi không?"""
    prompt = f"""Rate ANSWER RELEVANCE.

Question: {query}
Answer: {answer}

Scale 1-5:
5 = directly and fully answers
4 = mostly answers, minor details missing
3 = partially answers, off-focus
2 = partly off-topic
1 = does not answer
Note: "insufficient data" is relevant if question truly not in docs.

Output JSON only: {{"score":<1-5>,"reason":"<brief>"}}"""
    try:
        d = _parse_judge(_call_judge_llm(prompt))
        return {"score": d.get("score"), "notes": d.get("reason", "")}
    except Exception as e:
        return {"score": None, "notes": f"judge error: {e}"}


def score_context_recall(
    chunks_used: List[Dict], expected_sources: List[str]
) -> Dict:
    """Context Recall (source-match, no LLM needed): score 0-5 = recall * 5."""
    if not expected_sources:
        return {"score": None, "recall": None, "notes": "No expected sources"}
    retrieved = {c.get("metadata", {}).get("source", "") for c in chunks_used}
    found, missing = 0, []
    for exp in expected_sources:
        name = exp.split("/")[-1].replace(".pdf", "").replace(".md", "").replace(".txt", "")
        if any(name.lower() in r.lower() for r in retrieved):
            found += 1
        else:
            missing.append(exp)
    recall = found / len(expected_sources)
    return {
        "score": round(recall * 5),
        "recall": recall,
        "found": found,
        "missing": missing,
        "notes": f"Retrieved {found}/{len(expected_sources)}"
                 + (f". Missing: {missing}" if missing else ""),
    }


def score_completeness(query: str, answer: str, expected_answer: str) -> Dict:
    """Completeness 1-5: answer có bao phủ đủ thông tin so với expected không?"""
    if not expected_answer:
        return {"score": None, "notes": "No expected_answer"}
    prompt = f"""Rate COMPLETENESS of the answer vs expected answer.

Question: {query}
Answer: {answer}
Expected: {expected_answer}

Scale 1-5:
5 = all key points covered
4 = missing 1 minor detail
3 = has main idea but missing 1-2 key info
2 = missing many important info
1 = missing most core content

Output JSON only: {{"score":<1-5>,"reason":"<brief>","missing_points":[]}}"""
    try:
        d = _parse_judge(_call_judge_llm(prompt))
        notes = d.get("reason", "")
        mp = d.get("missing_points", [])
        if mp:
            notes += f" | Missing: {', '.join(str(x) for x in mp[:2])}"
        return {"score": d.get("score"), "notes": notes}
    except Exception as e:
        return {"score": None, "notes": f"judge error: {e}"}


# =============================================================================
# SCORECARD RUNNER
# =============================================================================

def run_scorecard(
    config: Dict,
    test_questions: Optional[List[Dict]] = None,
    verbose: bool = True,
) -> List[Dict]:
    """
    Chạy toàn bộ test questions qua pipeline → chấm 4 metrics.
    Mỗi row kết quả giữ chunks_used để RAGAS dùng lại (không export ra CSV/MD).
    """
    if test_questions is None:
        with open(TEST_QUESTIONS_PATH, encoding="utf-8") as f:
            test_questions = json.load(f)

    label = config.get("label", "unnamed")
    results: List[Dict] = []

    print(f"\n{'='*65}")
    print(f"Scorecard: {label}")
    print(f"mode={config['retrieval_mode']} | top_k={config['top_k_search']}>{config['top_k_select']} | rerank={config['use_rerank']}")
    print("=" * 65)

    for q in test_questions:
        qid     = q["id"]
        query   = q["question"]
        expected = q.get("expected_answer", "")
        exp_src  = q.get("expected_sources", [])

        if verbose:
            print(f"\n[{qid}] {query}")

        try:
            res = rag_answer(
                query=query,
                retrieval_mode=config["retrieval_mode"],
                top_k_search=config["top_k_search"],
                top_k_select=config["top_k_select"],
                use_rerank=config["use_rerank"],
                verbose=False,
            )
            answer      = res["answer"]
            chunks_used = res["chunks_used"]
        except NotImplementedError:
            answer, chunks_used = "PIPELINE_NOT_IMPLEMENTED", []
        except Exception as e:
            answer, chunks_used = f"ERROR: {e}", []

        faith    = score_faithfulness(answer, chunks_used)
        relevance = score_answer_relevance(query, answer)
        recall   = score_context_recall(chunks_used, exp_src)
        complete = score_completeness(query, answer, expected)

        row = {
            "id": qid,
            "category": q.get("category", ""),
            "query": query,
            "answer": answer,
            "expected_answer": expected,
            "faithfulness": faith["score"],
            "faithfulness_notes": faith["notes"],
            "relevance": relevance["score"],
            "relevance_notes": relevance["notes"],
            "context_recall": recall["score"],
            "context_recall_notes": recall["notes"],
            "completeness": complete["score"],
            "completeness_notes": complete["notes"],
            "config_label": label,
            "chunks_used": chunks_used,   # giữ cho RAGAS
        }
        results.append(row)

        if verbose:
            print(f"  Answer: {answer[:120]}")
            print(f"  F={faith['score']} R={relevance['score']} Rc={recall['score']} C={complete['score']}")

    # Averages
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]
    print(f"\n{'─'*40}")
    for m in metrics:
        s = [r[m] for r in results if r[m] is not None]
        avg = sum(s) / len(s) if s else None
        print(f"  Avg {m:<18}: {f'{avg:.2f}/5' if avg else 'N/A'}")

    return results


# =============================================================================
# A/B COMPARISON
# =============================================================================

def compare_ab(
    baseline: List[Dict],
    variant: List[Dict],
    output_csv: Optional[str] = None,
) -> None:
    """In bảng A/B comparison: baseline vs variant theo từng metric và câu."""
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]

    print(f"\n{'='*65}")
    print("A/B Comparison: Baseline vs Variant")
    print("=" * 65)
    print(f"  {'Metric':<22} {'Baseline':>9} {'Variant':>9} {'Delta':>8} {'Win':>10}")
    print("  " + "─" * 54)

    for m in metrics:
        b = [r[m] for r in baseline if r[m] is not None]
        v = [r[m] for r in variant  if r[m] is not None]
        ba = sum(b) / len(b) if b else None
        va = sum(v) / len(v) if v else None
        d  = (va - ba) if (ba and va) else None
        win = ("Variant ▲" if d and d > 0 else "Baseline ▼" if d and d < 0 else "Tie")
        print(f"  {m:<22} {f'{ba:.2f}' if ba else 'N/A':>9} "
              f"{f'{va:.2f}' if va else 'N/A':>9} "
              f"{f'{d:+.2f}' if d else 'N/A':>8} {win:>10}")

    print(f"\n  {'ID':<6} {'Base F/R/Rc/C':<18} {'Var F/R/Rc/C':<18} {'Better':<10}")
    print("  " + "─" * 54)
    b_map = {r["id"]: r for r in baseline}
    for vr in variant:
        br = b_map.get(vr["id"], {})
        bs = "/".join(str(br.get(m, "?")) for m in metrics)
        vs = "/".join(str(vr.get(m, "?")) for m in metrics)
        bt = sum(br.get(m, 0) or 0 for m in metrics)
        vt = sum(vr.get(m, 0) or 0 for m in metrics)
        better = "Variant" if vt > bt else ("Baseline" if bt > vt else "Tie")
        print(f"  {vr['id']:<6} {bs:<18} {vs:<18} {better:<10}")

    if output_csv:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        excl = {"chunks_used"}
        rows = baseline + variant
        if rows:
            fields = [k for k in rows[0] if k not in excl]
            with open(RESULTS_DIR / output_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            print(f"\n  CSV saved: results/{output_csv}")


# =============================================================================
# RAGAS 0.4.x EVALUATION — LLM judge: gpt-4o-mini
# =============================================================================

def _build_ragas_llm_emb():
    """
    Khởi tạo LangchainLLMWrapper với gpt-4o-mini cho RAGAS 0.4.x.
    Trả về (llm, embeddings) hoặc (None, None) nếu thiếu langchain_openai.
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None, None
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        llm = LangchainLLMWrapper(
            ChatOpenAI(model=LLM_JUDGE_MODEL, temperature=0, openai_api_key=key)
        )
        emb = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=key)
        )
        return llm, emb
    except ImportError:
        print("[RAGAS] langchain_openai chưa cài: pip install langchain-openai")
        return None, None


def run_ragas_eval(
    scorecard_results: List[Dict],
    config_label: str = "eval",
) -> Optional[Dict[str, float]]:
    """
    RAGAS 0.4.x evaluation.

    Metrics: Faithfulness, AnswerRelevancy, ContextRecall.
    LLM judge: gpt-4o-mini (qua LangchainLLMWrapper).

    Tái dùng answer + chunks_used từ run_scorecard() — không chạy pipeline lại.

    Returns:
        {"faithfulness": float, "answer_relevancy": float, "context_recall": float}
        hoặc None nếu lỗi/chưa cài ragas.
    """
    valid = [
        r for r in scorecard_results
        if r["answer"] not in ("PIPELINE_NOT_IMPLEMENTED",)
        and not r["answer"].startswith("ERROR:")
    ]
    if not valid:
        print("[RAGAS] Không có kết quả hợp lệ — chạy pipeline trước.")
        return None

    print(f"\n{'='*65}")
    print(f"RAGAS Evaluation: {config_label}")
    print(f"LLM judge: {LLM_JUDGE_MODEL}  |  Câu: {len(valid)}")
    print("=" * 65)

    questions     = [r["query"] for r in valid]
    answers       = [r["answer"] for r in valid]
    ground_truths = [r.get("expected_answer", "") for r in valid]
    contexts_list = [
        [c.get("text", "") for c in r.get("chunks_used", [])] or [""]
        for r in valid
    ]

    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall

        llm, emb = _build_ragas_llm_emb()

        if llm is not None:
            fm = Faithfulness(llm=llm)
            am = AnswerRelevancy(llm=llm, embeddings=emb)
            cm = ContextRecall(llm=llm)
            print(f"  Using explicit LLM: {LLM_JUDGE_MODEL}")
        else:
            fm, am, cm = Faithfulness(), AnswerRelevancy(), ContextRecall()
            print("  Using default LLM from OPENAI_API_KEY env")

        metrics = [fm, am, cm]

        # Try RAGAS 0.4.x / 0.2.x API: EvaluationDataset + SingleTurnSample
        try:
            from ragas import EvaluationDataset, SingleTurnSample
            samples = [
                SingleTurnSample(
                    user_input=q, response=a,
                    retrieved_contexts=ctxs, reference=ref,
                )
                for q, a, ctxs, ref in zip(questions, answers, contexts_list, ground_truths)
            ]
            result = ragas_evaluate(
                dataset=EvaluationDataset(samples=samples),
                metrics=metrics,
            )
        except (ImportError, TypeError, AttributeError):
            # Fallback RAGAS 0.1.x: HuggingFace Dataset
            from datasets import Dataset
            from ragas.metrics import faithfulness, answer_relevancy, context_recall
            result = ragas_evaluate(
                Dataset.from_dict({
                    "question": questions, "answer": answers,
                    "contexts": contexts_list, "ground_truth": ground_truths,
                }),
                metrics=[faithfulness, answer_relevancy, context_recall],
            )

        df = result.to_pandas()
        ragas_scores: Dict[str, float] = {}

        target_cols = [c for c in df.columns if c in ("faithfulness", "answer_relevancy", "context_recall")]
        print(f"\n  {'Metric':<28} {'Score (0-1)':>12} {'equiv /5':>10}")
        print("  " + "─" * 52)
        for col in target_cols:
            avg = float(df[col].mean())
            ragas_scores[col] = avg
            print(f"  {col:<28} {avg:>12.4f} {avg*5:>10.2f}")

        return ragas_scores

    except ImportError:
        print("\n[RAGAS] Chưa cài. Chạy: pip install ragas langchain-openai")
        return None
    except Exception as e:
        print(f"\n[RAGAS] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_simple_vs_ragas(
    scorecard_results: List[Dict],
    ragas_scores: Dict[str, float],
    label: str = "",
) -> None:
    """
    Bảng so sánh: Simple eval (run_scorecard) vs RAGAS 0.4.x.

    Mapping:
      faithfulness (simple)   <-> faithfulness (RAGAS)
      relevance (simple)      <-> answer_relevancy (RAGAS)
      context_recall (simple) <-> context_recall (RAGAS)

    Diverge > 0.2 -> hai phương pháp không nhất quán, cần trace lại.
    """
    mapping = {
        "faithfulness":   "faithfulness",
        "relevance":      "answer_relevancy",
        "context_recall": "context_recall",
    }
    title = f"Simple eval vs RAGAS{' — ' + label if label else ''}"
    print(f"\n{'='*65}\n{title}\n{'='*65}")
    print(f"  {'Metric':<22} {'Simple/5':>9} {'Simple/1':>9} {'RAGAS/1':>9} {'|Diff|':>8}")
    print("  " + "─" * 60)

    for sk, rk in mapping.items():
        s = [r[sk] for r in scorecard_results if r[sk] is not None]
        sa = sum(s) / len(s) if s else None
        sn = sa / 5.0 if sa is not None else None
        ra = ragas_scores.get(rk)
        diff = abs(sn - ra) if (sn is not None and ra is not None) else None
        flag = "  <-- diverge!" if diff and diff > 0.2 else ""
        print(f"  {sk:<22} "
              f"{f'{sa:.2f}' if sa is not None else 'N/A':>9} "
              f"{f'{sn:.3f}' if sn is not None else 'N/A':>9} "
              f"{f'{ra:.3f}' if ra is not None else 'N/A':>9} "
              f"{f'{diff:.3f}' if diff is not None else 'N/A':>8}{flag}")

    print("""
  Kết luận:
    Simple eval  -> nhanh, rẻ, debug-friendly, tuỳ chỉnh prompt tiếng Việt
    RAGAS 0.4.x  -> benchmark chuẩn ngành, cite paper, gpt-4o-mini judge
    Diverge      -> cần trace câu đó: indexing? retrieval? generation?""")


# =============================================================================
# SCORECARD MARKDOWN GENERATOR
# =============================================================================

def generate_scorecard_summary(results: List[Dict], label: str) -> str:
    """Tạo scorecard.md từ results."""
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]
    avgs: Dict[str, Optional[float]] = {}
    for m in metrics:
        s = [r[m] for r in results if r[m] is not None]
        avgs[m] = sum(s) / len(s) if s else None

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    md  = f"# Scorecard: {label}\nGenerated: {ts}\n\n## Summary\n\n"
    md += "| Metric | Average Score |\n|--------|-------------|\n"
    for m, avg in avgs.items():
        md += f"| {m.replace('_', ' ').title()} | {f'{avg:.2f}/5' if avg else 'N/A'} |\n"

    md += "\n## Per-Question Results\n\n"
    md += "| ID | Category | Faithful | Relevant | Recall | Complete | Answer |\n"
    md += "|----|----------|----------|----------|--------|----------|--------|\n"
    for r in results:
        ans = r.get("answer", "")[:60].replace("\n", " ")
        md += (f"| {r['id']} | {r['category']} | {r.get('faithfulness','N/A')} | "
               f"{r.get('relevance','N/A')} | {r.get('context_recall','N/A')} | "
               f"{r.get('completeness','N/A')} | {ans} |\n")
    return md


# =============================================================================
# GRADING RUN GENERATOR
# Chạy sau khi grading_questions.json được public lúc 17:00
# =============================================================================

def generate_grading_run_log(
    grading_questions_path: str = "data/grading_questions.json",
    output_path: str = "logs/grading_run.json",
    retrieval_mode: str = "hybrid",
    use_rerank: bool = True,
) -> None:
    """
    Chạy pipeline với grading_questions.json và lưu log.

    Format log theo SCORING.md:
      [{"id", "question", "answer", "sources", "chunks_retrieved",
        "retrieval_mode", "timestamp"}, ...]
    """
    gq_path = Path(__file__).parent / grading_questions_path
    if not gq_path.exists():
        print(f"[Grading] {gq_path.name} chưa có (public lúc 17:00).")
        return

    with open(gq_path, encoding="utf-8") as f:
        questions = json.load(f)

    print(f"\n[Grading] {len(questions)} câu | mode={retrieval_mode} rerank={use_rerank}")
    log = []

    for q in questions:
        try:
            res = rag_answer(
                q["question"],
                retrieval_mode=retrieval_mode,
                use_rerank=use_rerank,
                verbose=False,
            )
            log.append({
                "id":               q["id"],
                "question":         q["question"],
                "answer":           res["answer"],
                "sources":          res["sources"],
                "chunks_retrieved": len(res["chunks_used"]),
                "retrieval_mode":   res["config"]["retrieval_mode"],
                "timestamp":        datetime.now().isoformat(),
            })
            print(f"  [{q['id']}] {q['question'][:50]}...")
        except Exception as e:
            log.append({
                "id": q["id"], "question": q["question"],
                "answer": f"PIPELINE_ERROR: {e}",
                "sources": [], "chunks_retrieved": 0,
                "retrieval_mode": retrieval_mode,
                "timestamp": datetime.now().isoformat(),
            })
            print(f"  [{q['id']}] ERROR: {e}")

    out = Path(__file__).parent / output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"[Grading] Saved: {out}  ({len(log)} entries)")


# =============================================================================
# MAIN — Chạy evaluation đầy đủ
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print(f"Sprint 4: Evaluation & Scorecard  |  LLM judge: {LLM_JUDGE_MODEL}")
    print("=" * 65)

    try:
        with open(TEST_QUESTIONS_PATH, encoding="utf-8") as f:
            test_questions = json.load(f)
        print(f"\nTest questions: {len(test_questions)} câu")
        for q in test_questions[:3]:
            print(f"  [{q['id']}] {q['question']} ({q.get('category', '')})")
        print("  ...")
    except FileNotFoundError:
        print("test_questions.json not found!")
        test_questions = []

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    baseline_results: List[Dict] = []
    variant_results:  List[Dict] = []

    # ─── Baseline ────────────────────────────────────────────────────────
    print("\n" + "─" * 65 + "\nBASELINE\n" + "─" * 65)
    try:
        baseline_results = run_scorecard(BASELINE_CONFIG, test_questions, verbose=True)
        md = generate_scorecard_summary(baseline_results, BASELINE_CONFIG["label"])
        (RESULTS_DIR / "scorecard_baseline.md").write_text(md, encoding="utf-8")
        print("\n[Saved] results/scorecard_baseline.md")
    except NotImplementedError:
        print("[!] Pipeline not implemented. Run index.py first.")
    except Exception as e:
        print(f"[!] Baseline error: {e}")

    # ─── Variant ─────────────────────────────────────────────────────────
    print("\n" + "─" * 65 + "\nVARIANT\n" + "─" * 65)
    try:
        variant_results = run_scorecard(VARIANT_CONFIG, test_questions, verbose=True)
        md = generate_scorecard_summary(variant_results, VARIANT_CONFIG["label"])
        (RESULTS_DIR / "scorecard_variant.md").write_text(md, encoding="utf-8")
        print("\n[Saved] results/scorecard_variant.md")
    except NotImplementedError:
        print("[!] Variant not implemented. Complete Sprint 3 first.")
    except Exception as e:
        print(f"[!] Variant error: {e}")

    # ─── A/B Simple ──────────────────────────────────────────────────────
    if baseline_results and variant_results:
        compare_ab(baseline_results, variant_results, output_csv="ab_comparison.csv")

    # ─── RAGAS ───────────────────────────────────────────────────────────
    ragas_b: Optional[Dict] = None
    ragas_v: Optional[Dict] = None

    if baseline_results:
        ragas_b = run_ragas_eval(baseline_results, BASELINE_CONFIG["label"])
    if variant_results:
        ragas_v = run_ragas_eval(variant_results, VARIANT_CONFIG["label"])

    if ragas_b and baseline_results:
        compare_simple_vs_ragas(baseline_results, ragas_b, "Baseline")
    if ragas_v and variant_results:
        compare_simple_vs_ragas(variant_results, ragas_v, "Variant")

    # RAGAS A/B delta
    if ragas_b and ragas_v:
        print(f"\n{'='*65}\nRAGAS A/B: Baseline vs Variant\n{'='*65}")
        print(f"  {'Metric':<28} {'Baseline':>10} {'Variant':>10} {'Delta':>8}")
        print("  " + "─" * 58)
        for m in ("faithfulness", "answer_relevancy", "context_recall"):
            b = ragas_b.get(m)
            v = ragas_v.get(m)
            d = (v - b) if (b is not None and v is not None) else None
            print(f"  {m:<28} "
                  f"{f'{b:.4f}' if b else 'N/A':>10} "
                  f"{f'{v:.4f}' if v else 'N/A':>10} "
                  f"{f'{d:+.4f}' if d else 'N/A':>8}")

    # ─── Grading run ──────────────────────────────────────────────────────
    gq = Path(__file__).parent / "data" / "grading_questions.json"
    if gq.exists():
        print("\n" + "─" * 65 + "\nGRADING RUN\n" + "─" * 65)
        generate_grading_run_log(
            retrieval_mode=VARIANT_CONFIG["retrieval_mode"],
            use_rerank=VARIANT_CONFIG["use_rerank"],
        )
    else:
        print(f"\n[Grading] {gq.name} chưa có — public lúc 17:00")
        print("  Khi có file, chạy: from eval import generate_grading_run_log; generate_grading_run_log()")

    print("\n" + "=" * 65)
    print("Sprint 4 done!")
    print("  results/scorecard_baseline.md")
    print("  results/scorecard_variant.md")
    print("  results/ab_comparison.csv")
    print("  logs/grading_run.json  (sau 17:00)")
'''

dest = r'c:\Users\Admin\Lecture-Day-08-09-10\day08\lab\eval.py'
with open(dest, 'w', encoding='utf-8') as f:
    f.write(EVAL_CONTENT)
print(f"Written {len(EVAL_CONTENT)} chars to {dest}")
