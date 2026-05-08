# ============================================================
# rag/llm.py — LLM client, generation, and Arabic fallback response
# ============================================================

from __future__ import annotations

import numpy as np

from webmd.config import LLM_MODEL, OPENROUTER_API_KEY
from webmd.rag.retriever import Hit


def build_client(api_key: str = OPENROUTER_API_KEY):
    """Build an OpenAI-compatible client pointed at OpenRouter.

    Returns None if the openai package is unavailable or no key is provided.
    """
    if not api_key:
        return None
    try:
        from openai import OpenAI  # heavy import — kept local
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=30.0,
        )
    except ImportError:
        return None


def llm_generate(
    query: str,
    hits: list[Hit],
    client,
    model: str = LLM_MODEL,
) -> str:
    """Send retrieved reviews as context to the LLM and return a medical summary.

    Falls back to fallback_response() if the client is None, hits is empty,
    or the API call raises an exception.
    """
    if not client or not hits:
        return fallback_response(query, hits)

    context = "\n\n".join(_format_hit_for_llm(i, h) for i, h in enumerate(hits, 1))

    system_prompt = (
        "أنت مساعد طبي متخصص في تحليل تجارب المرضى مع الأدوية.\n"
        "مهمتك: تلخيص تجارب المرضى الحقيقية بناءً على المراجعات المقدمة لك.\n\n"
        "قواعد مهمة:\n"
        "- استند فقط على المراجعات المقدمة، لا تخترع معلومات\n"
        "- اذكر الآثار الجانبية الموجودة في المراجعات بوضوح\n"
        "- وضح نسبة الرضا العامة\n"
        "- استخدم لغة طبية بسيطة ومفهومة\n"
        "- نبّه دائماً أن هذه تجارب شخصية وليست نصيحة طبية"
    )
    user_prompt = (
        f"سؤال المريض: {query}\n\n"
        f"المراجعات المسترجعة ({len(hits)} مراجعة):\n"
        f"{context}\n\n"
        "المطلوب: اكتب ملخصاً طبياً دقيقاً يجيب على سؤال المريض "
        "بناءً على هذه التجارب الحقيقية."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=1024,
            temperature=0.3,
        )
        llm_text   = response.choices[0].message.content.strip()
        structured = fallback_response(query, hits)
        return f"{llm_text}\n\n{'═' * 60}\n📊 البيانات الإحصائية:\n{'═' * 60}\n{structured}"
    except Exception as exc:
        return f"⚠️ LLM غير متاح ({exc})\n\n{fallback_response(query, hits)}"


def fallback_response(query: str, hits: list[Hit]) -> str:
    """Build a structured Arabic-language summary from retrieved hits.

    Used when no LLM client is available or as the statistical appendix
    appended after the LLM answer.
    """
    if not hits:
        return "لم يتم العثور على مراجعات مطابقة لطلبك."

    drugs      = list({h.drug      for h in hits if h.drug})
    conditions = list({h.condition for h in hits if h.condition})

    sat_vals = [h.satisfaction_float for h in hits if h.satisfaction_float]
    eff_vals = [
        float(h.effectiveness)
        for h in hits
        if h.effectiveness.replace(".", "").isdigit()
    ]
    avg_sat = float(np.mean(sat_vals)) if sat_vals else 0.0
    avg_eff = float(np.mean(eff_vals)) if eff_vals else 0.0

    all_sides = [
        h.sides for h in hits
        if h.sides and h.sides.lower() not in ("not reported", "nan", "")
    ]

    pos = sum(1 for h in hits if "sentiment: positive" in h.document.lower())
    neg = sum(1 for h in hits if "sentiment: negative" in h.document.lower())
    neu = len(hits) - pos - neg

    lines = [
        f"بناءً على {len(hits)} مراجعة مشابهة لسؤالك:\n",
        f"الدواء/الأدوية: {', '.join(drugs) or 'متنوعة'}",
        f"الحالة المرضية: {', '.join(conditions) or 'متنوعة'}",
        f"متوسط الرضا: {avg_sat:.1f}/5  |  متوسط الفعالية: {avg_eff:.1f}/5",
        f"التقييمات: ✅ إيجابية: {pos}  ⚠️ محايدة: {neu}  ❌ سلبية: {neg}",
    ]

    if all_sides:
        lines.append("\nالآثار الجانبية المذكورة:\n  • " + "\n  • ".join(all_sides[:5]))

    lines.append(f"\n── جميع المراجعات المسترجعة ({len(hits)}) ──")
    for i, h in enumerate(hits, 1):
        sat_f = h.satisfaction_float
        icon  = "✅" if sat_f >= 4 else ("❌" if sat_f <= 2 else "⚠️")
        lines.append(
            f"\n[{i}] {icon}  تشابه: {h.similarity:.0%}  |  "
            f"رضا: {h.satisfaction}/5  |  فعالية: {h.effectiveness}/5  |  "
            f"عمر: {h.age}  |  جنس: {h.sex}\n"
            f"الدواء: {h.drug}  |  الحالة: {h.condition}\n"
            f'"{h.review_text}"'
        )

    return "\n".join(lines)


# ── Private helpers ──────────────────────────────────────────────────────────

def _format_hit_for_llm(index: int, h: Hit) -> str:
    """Format a single Hit as an Arabic-labelled context block for the LLM prompt."""
    sat_f     = h.satisfaction_float
    sentiment = "إيجابية" if sat_f >= 4 else ("سلبية" if sat_f <= 2 else "محايدة")
    return (
        f"[مراجعة {index}] الدواء: {h.drug} | الحالة: {h.condition} | "
        f"الرضا: {h.satisfaction}/5 | التقييم: {sentiment} | "
        f"الآثار الجانبية: {h.sides}\n"
        f"النص: {h.review_text}"
    )
