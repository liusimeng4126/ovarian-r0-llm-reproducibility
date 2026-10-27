#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extraction V6.1 FINAL ovarian-cancer MRI structured extractor.

Design goals
------------
1) PRE and POST are extracted independently by the LLM.
2) The LLM may only cite clause IDs; evidence text is always copied by Python.
3) Python applies deterministic rescue/guard rules for anatomy, sizes, change scope,
   bowel relation, node negation scope, benign exclusions, fluid, and report damage.
4) V6 validation means INTERNAL CONSISTENCY / SOURCE-RULE AUDIT ONLY. It is not a
   claim of factual accuracy against a human gold standard.
5) No patient-specific answers are hard-coded.

Typical run
-----------
  pip install openai
  set DASHSCOPE_API_KEY=...
  python extract_v6_1_FINAL.py --input "C:\\path\\测试1-20.txt" --outdir "C:\\path\\v6_output"

For a later batch:
  python extract_v6_1_FINAL.py --input "C:\\path\\all_473_pairs.txt" --outdir "C:\\path\\v6_473"

Outputs
-------
  all_patients_v6_1.jsonl
  review_v6_1.csv
  run_manifest_v6_1.json
  model_trace_v6_1.jsonl   (unless --no-trace)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from copy import deepcopy
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

VERSION = "V6.1-FINAL-STRUCTURING-2026-10-11"
DEFAULT_MODEL = "qwen3.8-flash"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
TEMPERATURE = 0

SITE_ORDER = [
    "primary_pelvic_mass",
    "pelvic_peritoneum",
    "greater_omentum",
    "lesser_omentum_lesser_sac",
    "small_bowel_mesenteric",
    "colonic_mesenteric",
    "paracolic_wall_peritoneum",
    "diaphragmatic_peritoneum",
    "upper_abdominal_peritoneum",
    "porta_hepatis",
    "bowel_stomach_involvement",
    "pelvic_retroperitoneal_nodes",
    "upper_abdominal_cardiophrenic_nodes",
    "distant_metastasis",
]

SITE_CN = {
    "primary_pelvic_mass": "盆腔/附件原发肿块",
    "pelvic_peritoneum": "盆底/盆腔腹膜",
    "greater_omentum": "大网膜",
    "lesser_omentum_lesser_sac": "小网膜/网膜囊/肝胃韧带",
    "small_bowel_mesenteric": "小肠系膜/肠系膜根部",
    "colonic_mesenteric": "结肠/直肠系膜",
    "paracolic_wall_peritoneum": "结肠旁沟/壁层腹膜",
    "diaphragmatic_peritoneum": "膈下/膈腹膜",
    "upper_abdominal_peritoneum": "肝周/脾周/肝被膜等上腹腹膜",
    "porta_hepatis": "肝门/肝十二指肠韧带",
    "bowel_stomach_involvement": "肠管/胃受累",
    "pelvic_retroperitoneal_nodes": "盆腔/腹膜后淋巴结",
    "upper_abdominal_cardiophrenic_nodes": "上腹/心膈角淋巴结",
    "distant_metastasis": "远处转移",
}

PERITONEAL_SITES = {
    "pelvic_peritoneum",
    "greater_omentum",
    "lesser_omentum_lesser_sac",
    "small_bowel_mesenteric",
    "colonic_mesenteric",
    "paracolic_wall_peritoneum",
    "diaphragmatic_peritoneum",
    "upper_abdominal_peritoneum",
    "porta_hepatis",
}

ALLOWED_STATUS = {"present", "absent", "uncertain", "not_mentioned", "not_evaluable"}
ALLOWED_CHANGE = {"decreased", "increased", "stable", "mixed", "resolved", "new", "not_evaluable", "not_stated"}

BENIGN_TERMS = (
    "囊肿", "卵泡", "子宫肌瘤", "肌瘤", "内膜异位", "腺肌症", "纳囊", "那囊",
    "腺瘤", "副脾", "脂肪瘤", "血管平滑肌脂肪瘤", "结石", "息肉",
)
MALIGNANT_TERMS = (
    "卵巢癌", "恶性", "转移", "癌", "治疗后", "肿瘤", "占位",
)
LESION_TERMS = ("增厚", "结节", "肿块", "占位", "异常信号", "信号影", "病灶", "饼状", "斑片", "条索")

# Damage markers must be UI/report-capture language, not clinical words such as "输尿管截断".
INCOMPLETE_RE = re.compile(
    r"【[^】]*(?:截断|遮挡|无法(?:完整)?辨认|未显示|下缘|窗口)[^】]*】"
    r"|(?:图片|诊断框|对话框|窗口)[^。；\n]{0,30}(?:截断|遮挡|未显示|无法(?:完整)?辨认)"
)

SIZE_MULTI_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*[×xX*]\s*(\d+(?:\.\d+)?)(?:\s*[×xX*]\s*(\d+(?:\.\d+)?))?\s*mm", re.I)
SIZE_SINGLE_RE = re.compile(r"(?:长径|短径|直径|径线|大小|范围|截面)(?:约|为|示|大小|范围|较大截面|最大)?\s*(\d+(?:\.\d+)?)\s*mm", re.I)
SHORT_AXIS_RE = re.compile(r"短径(?:约|为)?\s*(\d+(?:\.\d+)?)\s*mm", re.I)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def as_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        v = float(x)
        if not math.isfinite(v) or v < 0:
            return None
        return round(v, 3)
    except Exception:
        return None


def unique_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def parse_patient_selector(spec: Optional[str]) -> Optional[set[str]]:
    if not spec:
        return None
    out: set[str] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            for n in range(int(a), int(b) + 1):
                out.add(str(n))
        else:
            out.add(str(int(token)))
    return out


def parse_patient_pairs(text: str) -> Dict[str, Dict[str, str]]:
    """Parse blocks headed by lines like 1a / 1b / 20a / 20b."""
    # Accept both bare headers (148a) and bracketed headers (【148a】/[148a]).
    matches = list(re.finditer(r"(?m)^\s*[【\[]?\s*(\d+)\s*([abAB])\s*[】\]]?\s*$", text))
    if not matches:
        raise ValueError("No patient headers like '1a'/'1b' or '【1a】'/'【1b】' were found in the input text.")
    pairs: Dict[str, Dict[str, str]] = defaultdict(dict)
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        pid = str(int(m.group(1)))
        arm = m.group(2).lower()
        pairs[pid][arm] = text[start:end].strip()
    return dict(pairs)


def clauseize(report_text: str) -> List[Dict[str, str]]:
    """Turn a report into stable source clauses while repairing visual line wraps."""
    raw_lines = report_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    clauses_raw: List[Tuple[str, str]] = []
    section = "unknown"
    buf = ""

    def flush() -> None:
        nonlocal buf
        txt = re.sub(r"\s+", " ", buf).strip()
        if txt:
            clauses_raw.append((section, txt))
        buf = ""

    for raw in raw_lines:
        line = raw.strip()
        if not line:
            if buf and re.search(r"[。；;]$", buf):
                flush()
            continue
        if line in {"影像学表现", "影像学所见"}:
            flush(); section = "findings"; continue
        if line in {"影像学诊断", "诊断", "影像诊断"}:
            flush(); section = "impression"; continue
        if re.match(r"^\d+\s*[\.．、]\s*", line):
            flush(); buf = line; continue
        if not buf:
            buf = line
        elif re.search(r"[。；;]$", buf):
            flush(); buf = line
        else:
            buf += " " + line
    flush()

    out: List[Dict[str, str]] = []
    for idx, (sec, txt) in enumerate(clauses_raw, 1):
        out.append({"id": f"C{idx:03d}", "section": sec, "text": txt})
    return out


def clause_map(clauses: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {c["id"]: c for c in clauses}


def format_clauses_for_prompt(clauses: Sequence[Dict[str, str]]) -> str:
    return "\n".join(f'{c["id"]} [{c["section"].upper()}] {c["text"]}' for c in clauses)


SYSTEM_PROMPT = """You are a radiology data-extraction engine for ovarian-cancer MRI reports written in Chinese.
Your job is structured extraction, not clinical advice. Read ONLY the single report supplied in this call.
PRE and POST are intentionally processed in separate calls. Never infer facts from the paired report.
Return strict JSON only. Never invent evidence text. Evidence must be clause IDs from the supplied list.
When wording is uncertain, preserve uncertainty rather than upgrading to definite disease."""

PROMPT_TEMPLATE = r"""
Extract the current report into the JSON schema below.

SITE DEFINITIONS
- primary_pelvic_mass: ovarian/adnexal/pelvic primary tumor mass, including a malignant-appearing mass posterior to uterus when impression says ovarian origin.
- pelvic_peritoneum: 盆底腹膜, 盆腔腹膜, 子宫直肠陷窝/Douglas pouch peritoneal disease.
- greater_omentum: 大网膜.
- lesser_omentum_lesser_sac: 小网膜, 网膜囊, 肝胃韧带.
- small_bowel_mesenteric: 小肠系膜 or 肠系膜根部 PERITONEAL disease; do not confuse mesenteric lymph nodes with peritoneal disease.
- colonic_mesenteric: 结肠系膜, 直肠系膜, 直肠系膜筋膜.
- paracolic_wall_peritoneum: 结肠旁沟/结肠旁腹膜 plus 壁层腹膜/壁腹膜/腹壁腹膜.
- diaphragmatic_peritoneum: 膈下腹膜, 膈肌下缘腹膜, 膈顶/膈腹膜.
- upper_abdominal_peritoneum: 肝周, 肝被膜, 肝肾隐窝, 脾周/脾旁, 脾周韧带, 肝周韧带, 脾胃/脾结肠韧带 when describing peritoneal disease.
- porta_hepatis: 肝门/肝十二指肠韧带 disease.
- bowel_stomach_involvement: relationship/invasion of bowel or stomach.
- pelvic_retroperitoneal_nodes: pelvic/iliac/obturator/presacral/para-aortic/retroperitoneal nodes.
- upper_abdominal_cardiophrenic_nodes: upper-abdominal, mesenteric, cardiophrenic/costophrenic nodes not covered above.
- distant_metastasis: definite/suspicious distant metastatic disease; exclude benign cysts/adenomas/accessory spleen/lipoma/angiomyolipoma/stones.

STATUS
present = explicitly present disease/lesion for that site.
absent = explicitly stated absent for that site. Do NOT turn silence into absent.
uncertain = suspicious/indeterminate relationship or lesion.
not_evaluable = explicitly poorly visualized / cannot be assessed.
not_mentioned = no usable statement.
For nodes, a visible node is 'present' even if small; a negative para-aortic statement must not erase a separately described iliac node.

PERITONEAL BURDEN GRADE
0 = explicitly absent.
1 = thickening/linear or subtle disease without definite nodules or mass.
2 = definite nodule(s), mass(es), or measurable implants.
3 = omental cake / confluent bulky sheet-like disease explicitly described as 饼状/omental cake or similarly extensive.
Do not automatically give every site grade 3 just because another listed site is omental cake.

BOWEL/STOMACH RELATION
relation_grade 0 = explicitly no involvement.
1 = contact/adjacent/close/compression (密接/邻近/受压) without invasion.
2 = boundary/interface unclear (分界不清/界限欠清) or invasion cannot be excluded.
3 = explicit invasion/involvement of bowel/stomach wall or rectum (侵及/侵犯/累及肠壁, 似达黏膜层).
If grade 3 -> status present. Grades 1-2 -> status uncertain unless stronger explicit wording exists.
"累及子宫直肠陷窝" is pelvic-peritoneal involvement, NOT rectal invasion by itself.

SIZE
Give max_size_mm only when the measurement unambiguously belongs to that site.
For a multi-site peritoneal sentence such as 'A、B、C...大者22mm', treat 22 mm as overall group maximum and DO NOT copy it to A/B/C unless the text says the largest lesion is located at a specific site.
If wording says '较大者位于左侧结肠旁沟...34mm', both overall group max and that site can receive 34 mm.
For multidimensional measurements use the largest dimension. For lymph nodes use short-axis when stated.

CHANGE_HINT
Use only an explicit comparison phrase in THIS report: decreased / increased / stable / mixed / resolved / new / not_evaluable / not_stated.
A shared predicate at the end of an enumerated clause applies to the listed sites when unqualified, e.g. '盆底腹膜、大网膜...较前减小'.
If wording is '部分减小' across multiple different sites and it does not identify which sites changed, do not assign a definite site-specific decrease; use not_stated for those individual sites.
If one compartment itself contains mixed response ('部分减小、部分无著变' or '囊性增大、实性稳定'), use mixed.
Do not infer new/resolved from PRE/POST presence alone.

BENIGN EXCLUSION
Do not use benign entities (囊肿, 卵泡, 肌瘤, 内膜异位症, 腺肌症, 纳囊/那囊, 腺瘤, 副脾, 脂肪瘤, 血管平滑肌脂肪瘤, 结石, 息肉) as malignant evidence unless the same clause explicitly says malignant/metastatic.

OUTPUT JSON SCHEMA
{
  "sites": {
    "<each of the 14 site keys>": {
      "status": "present|absent|uncertain|not_mentioned|not_evaluable",
      "burden_grade": 0|1|2|3|null,
      "max_size_mm": number|null,
      "relation_grade": 0|1|2|3|null,
      "evidence_ids": ["C001", ...],
      "change_hint": "decreased|increased|stable|mixed|resolved|new|not_evaluable|not_stated"
    }
  },
  "overall_peritoneal": {
    "group_max_size_mm": number|null,
    "evidence_ids": ["C001", ...]
  },
  "abdominopelvic_fluid": {
    "status": "present|absent|not_mentioned|not_evaluable",
    "grade": 0|1|2|3|null,
    "evidence_ids": ["C001", ...]
  },
  "pleural_effusion": {
    "status": "present|absent|not_mentioned|not_evaluable",
    "grade": 0|1|2|3|null,
    "evidence_ids": ["C001", ...]
  }
}

Every site key must be present in the JSON. Evidence IDs must come only from the supplied clauses.

REPORT CLAUSES:
{clauses}
"""

PROMPT_SHA256 = sha256_text(SYSTEM_PROMPT + "\n" + PROMPT_TEMPLATE)


def empty_site() -> Dict[str, Any]:
    return {
        "status": "not_mentioned",
        "burden_grade": None,
        "max_size_mm": None,
        "relation_grade": None,
        "evidence_ids": [],
        "change_hint": "not_stated",
    }


def empty_model_result() -> Dict[str, Any]:
    return {
        "sites": {s: empty_site() for s in SITE_ORDER},
        "overall_peritoneal": {"group_max_size_mm": None, "evidence_ids": []},
        "abdominopelvic_fluid": {"status": "not_mentioned", "grade": None, "evidence_ids": []},
        "pleural_effusion": {"status": "not_mentioned", "grade": None, "evidence_ids": []},
    }


def parse_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("JSON root is not an object")
        return obj
    except Exception:
        a = text.find("{")
        b = text.rfind("}")
        if a >= 0 and b > a:
            obj = json.loads(text[a:b + 1])
            if isinstance(obj, dict):
                return obj
        raise


def normalize_model_result(raw: Dict[str, Any], valid_ids: set[str]) -> Dict[str, Any]:
    out = empty_model_result()
    raw_sites = raw.get("sites") if isinstance(raw.get("sites"), dict) else {}
    for site in SITE_ORDER:
        src = raw_sites.get(site) if isinstance(raw_sites.get(site), dict) else {}
        status = src.get("status")
        out["sites"][site]["status"] = status if status in ALLOWED_STATUS else "not_mentioned"
        burden = src.get("burden_grade")
        out["sites"][site]["burden_grade"] = int(burden) if burden in {0, 1, 2, 3} else None
        rel = src.get("relation_grade")
        out["sites"][site]["relation_grade"] = int(rel) if rel in {0, 1, 2, 3} else None
        out["sites"][site]["max_size_mm"] = as_float(src.get("max_size_mm"))
        ev = src.get("evidence_ids") if isinstance(src.get("evidence_ids"), list) else []
        out["sites"][site]["evidence_ids"] = unique_keep_order(str(x) for x in ev if str(x) in valid_ids)
        ch = src.get("change_hint")
        out["sites"][site]["change_hint"] = ch if ch in ALLOWED_CHANGE else "not_stated"

    op = raw.get("overall_peritoneal") if isinstance(raw.get("overall_peritoneal"), dict) else {}
    out["overall_peritoneal"]["group_max_size_mm"] = as_float(op.get("group_max_size_mm"))
    ev = op.get("evidence_ids") if isinstance(op.get("evidence_ids"), list) else []
    out["overall_peritoneal"]["evidence_ids"] = unique_keep_order(str(x) for x in ev if str(x) in valid_ids)

    for key in ("abdominopelvic_fluid", "pleural_effusion"):
        src = raw.get(key) if isinstance(raw.get(key), dict) else {}
        st = src.get("status")
        out[key]["status"] = st if st in {"present", "absent", "not_mentioned", "not_evaluable"} else "not_mentioned"
        gr = src.get("grade")
        out[key]["grade"] = int(gr) if gr in {0, 1, 2, 3} else None
        ev = src.get("evidence_ids") if isinstance(src.get("evidence_ids"), list) else []
        out[key]["evidence_ids"] = unique_keep_order(str(x) for x in ev if str(x) in valid_ids)
    return out


def call_llm(client: Any, model: str, clauses: Sequence[Dict[str, str]], max_retries: int = 3) -> Tuple[Dict[str, Any], str]:
    prompt = PROMPT_TEMPLATE.replace("{clauses}", format_clauses_for_prompt(clauses))
    last_error: Optional[Exception] = None
    raw_text = ""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            raw_text = resp.choices[0].message.content or ""
            parsed = parse_json_object(raw_text)
            return normalize_model_result(parsed, {c["id"] for c in clauses}), raw_text
        except Exception as e:
            last_error = e
            if attempt + 1 < max_retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"LLM call/JSON parse failed after {max_retries} attempts: {last_error}")


def has_any(text: str, terms: Sequence[str]) -> bool:
    t = compact(text)
    return any(term in t for term in terms)


def extract_all_sizes_mm(text: str) -> List[float]:
    t = compact(text)
    vals: List[float] = []
    for m in SIZE_MULTI_RE.finditer(t):
        nums = [float(x) for x in m.groups() if x is not None]
        if nums:
            vals.append(max(nums))
    for m in SIZE_SINGLE_RE.finditer(t):
        vals.append(float(m.group(1)))
    return vals


def extract_short_axes_mm(text: str) -> List[float]:
    return [float(x) for x in SHORT_AXIS_RE.findall(compact(text))]


def direct_location_to_site(text: str) -> Optional[str]:
    t = compact(text)
    if any(x in t for x in ("盆底腹膜", "盆腔腹膜", "子宫直肠陷窝", "直肠子宫陷窝", "Douglas")):
        return "pelvic_peritoneum"
    if "大网膜" in t:
        return "greater_omentum"
    if any(x in t for x in ("小网膜", "网膜囊", "肝胃韧带")):
        return "lesser_omentum_lesser_sac"
    if any(x in t for x in ("小肠系膜", "肠系膜根部")):
        return "small_bowel_mesenteric"
    if any(x in t for x in ("结肠系膜", "直肠系膜", "直肠系膜筋膜", "结肠系膜侧")):
        return "colonic_mesenteric"
    if any(x in t for x in ("结肠旁", "结肠旁沟", "结肠旁腹膜", "壁层腹膜", "壁腹膜", "腹壁腹膜", "壁层腹膜")) and "结肠旁系膜" not in t:
        return "paracolic_wall_peritoneum"
    if any(x in t for x in ("膈下腹膜", "膈肌下缘腹膜", "膈顶腹膜", "膈腹膜", "膈面腹膜")):
        return "diaphragmatic_peritoneum"
    upper_hit = any(x in t for x in ("肝周", "肝被膜", "肝肾隐窝", "脾周", "脾旁", "脾周韧带", "肝周韧带", "脾胃韧带", "脾结肠韧带"))
    subcapsular_only = any(x in t for x in ("肝被膜下", "肝顶被膜下")) and "腹膜" not in t and "转移" not in t
    if upper_hit and not subcapsular_only:
        return "upper_abdominal_peritoneum"
    if any(x in t for x in ("肝门", "肝十二指肠韧带")):
        return "porta_hepatis"
    return None


def mapped_peritoneal_sites(text: str) -> List[str]:
    t = compact(text)
    out: List[str] = []
    if any(x in t for x in ("盆底腹膜", "盆腔腹膜", "子宫直肠陷窝", "直肠子宫陷窝", "Douglas")):
        out.append("pelvic_peritoneum")
    if "大网膜" in t:
        out.append("greater_omentum")
    if any(x in t for x in ("小网膜", "网膜囊", "肝胃韧带")):
        out.append("lesser_omentum_lesser_sac")
    if "小肠系膜" in t or ("肠系膜根部" in t and "淋巴结" not in t):
        out.append("small_bowel_mesenteric")
    if any(x in t for x in ("结肠系膜", "直肠系膜", "直肠系膜筋膜", "结肠系膜侧")):
        out.append("colonic_mesenteric")
    if any(x in t for x in ("结肠旁", "结肠旁沟", "结肠旁腹膜", "壁层腹膜", "壁腹膜", "腹壁腹膜", "右上腹壁腹膜", "双侧壁腹膜")) and "结肠旁系膜" not in t:
        out.append("paracolic_wall_peritoneum")
    if any(x in t for x in ("膈下腹膜", "膈肌下缘腹膜", "膈顶腹膜", "膈腹膜", "膈面腹膜")):
        out.append("diaphragmatic_peritoneum")
    upper_hit = any(x in t for x in ("肝周", "肝被膜", "肝肾隐窝", "脾周", "脾旁", "脾周韧带", "肝周韧带", "脾胃韧带", "脾结肠韧带"))
    subcapsular_only = any(x in t for x in ("肝被膜下", "肝顶被膜下")) and "腹膜" not in t and "转移" not in t
    if upper_hit and not subcapsular_only:
        out.append("upper_abdominal_peritoneum")
    if any(x in t for x in ("肝门", "肝十二指肠韧带")):
        out.append("porta_hepatis")
    return unique_keep_order(out)


def is_negative_peritoneal_clause(text: str) -> bool:
    t = compact(text)
    return bool(re.search(r"未见(?:明显)?(?:腹膜|大网膜|小网膜|网膜囊|结肠旁沟|壁层腹膜|肠系膜)?[^。；]{0,10}(?:增厚|结节|肿块|占位)", t))


def peritoneal_site_positive(text: str, site: str) -> bool:
    t = compact(text)
    mapped = mapped_peritoneal_sites(text)
    if site not in mapped or is_negative_peritoneal_clause(text) or is_not_evaluable_wording(text):
        return False
    # Explicit involvement of Douglas pouch is pelvic-peritoneal disease.
    if site == "pelvic_peritoneum" and re.search(r"(?:累及|侵犯|侵及)(?:子宫直肠陷窝|直肠子宫陷窝)", t):
        return True
    # A multi-site enumerated peritoneal clause followed by a shared disease predicate is valid.
    if len(mapped) >= 2 and any(x in t for x in LESION_TERMS):
        # But do not treat a pure primary-tumor relation sentence as peritoneal disease.
        if re.search(r"与[^。；]{0,25}(?:盆底腹膜|盆腔腹膜|壁层腹膜|结肠旁沟)[^。；]{0,10}(?:密接|分界不清|界限欠清)", t) and not re.search(r"(?:腹膜|大网膜|小网膜|网膜囊|结肠旁沟|肠系膜)[^。；]{0,35}(?:增厚|结节|肿块|占位|病灶|信号影)", t):
            return False
        return True
    # Single-site clause: require the disease descriptor to be locally attached to the site.
    terms = {
        "pelvic_peritoneum": r"(?:盆底腹膜|盆腔腹膜|子宫直肠陷窝|直肠子宫陷窝)",
        "greater_omentum": r"大网膜",
        "lesser_omentum_lesser_sac": r"(?:小网膜|网膜囊|肝胃韧带)",
        "small_bowel_mesenteric": r"(?:小肠系膜|肠系膜根部)",
        "colonic_mesenteric": r"(?:结肠系膜|直肠系膜|直肠系膜筋膜|结肠系膜侧)",
        "paracolic_wall_peritoneum": r"(?:结肠旁(?:沟|腹膜)?|壁层腹膜|壁腹膜|腹壁腹膜|右上腹壁腹膜|双侧壁腹膜)",
        "diaphragmatic_peritoneum": r"(?:膈下腹膜|膈肌下缘腹膜|膈顶腹膜|膈腹膜|膈面腹膜)",
        "upper_abdominal_peritoneum": r"(?:肝周|肝被膜|肝肾隐窝|脾周|脾旁|脾周韧带|肝周韧带|脾胃韧带|脾结肠韧带)",
        "porta_hepatis": r"(?:肝门|肝十二指肠韧带)",
    }[site]
    disease = r"(?:增厚|略厚|结节|肿块|占位|病灶|异常信号|信号影|饼状)"
    if re.search(terms + r"[^。；]{0,35}" + disease, t):
        return True
    if re.search(disease + r"[^。；]{0,20}" + terms, t):
        return True
    return False


def positive_peritoneal_sites(text: str) -> List[str]:
    return [s for s in mapped_peritoneal_sites(text) if peritoneal_site_positive(text, s)]


def is_positive_peritoneal_clause(text: str) -> bool:
    return bool(positive_peritoneal_sites(text))


def peritoneal_burden_from_clause(text: str, site: str) -> Optional[int]:
    if not peritoneal_site_positive(text, site):
        return None
    t = compact(text)
    # Grade 3 only when cake/bulky wording reasonably applies to this site.
    if "饼状" in t or "网膜饼" in t:
        if site == "greater_omentum":
            return 3
        # Shared predicate after an enumerated list can apply to all listed sites.
        if re.search(r"(?:增厚|占位)[^。；]{0,12}(?:呈|为)?[^。；]{0,8}饼状", t):
            return 3
    if any(x in t for x in ("结节", "肿块", "占位")):
        return 2
    if any(x in t for x in ("增厚", "略厚", "斑索状", "条状")):
        return 1
    return None


def looks_like_primary_clause(text: str) -> bool:
    t = compact(text)
    patterns = [
        r"(?<!与)(?:(?:双侧|左侧|右侧)?附件区|双侧附件|左侧附件|右侧附件)[^。；]{0,40}(?:肿块|占位|异常信号|信号影|结节|囊实性|囊实混合)",
        r"(?:盆腔|下腹及盆腔|子宫后方)[^。；]{0,35}(?:肿块|占位|异常信号|信号影|结节|囊实性|囊实混合)",
        r"(?:盆腔多发占位|盆腔占位|附件区占位|附件区结节)",
    ]
    if not any(re.search(p, t) for p in patterns):
        return False
    if any(x in t for x in BENIGN_TERMS) and not any(x in t for x in MALIGNANT_TERMS):
        if any(x in t for x in ("考虑卵巢囊肿", "考虑卵泡", "内膜异位症", "子宫肌瘤", "纳囊", "那囊")):
            return False
    return True

def malignant_primary_context(clauses: Sequence[Dict[str, str]]) -> bool:
    t = compact(" ".join(c["text"] for c in clauses))
    return any(x in t for x in ("卵巢癌", "来源于卵巢", "考虑源于卵巢", "卵巢恶性", "盆腔恶性肿瘤", "考虑恶性", "盆腔多发占位治疗后", "盆腔占位治疗后", "附件区占位治疗后"))


def adnexal_sides(text: str) -> set[str]:
    t = compact(text)
    out: set[str] = set()
    if "双侧附件" in t:
        out.update({"L", "R"})
    if "左侧附件" in t:
        out.add("L")
    if "右侧附件" in t:
        out.add("R")
    return out


def benign_adnexal_sides(clauses: Sequence[Dict[str, str]]) -> set[str]:
    out: set[str] = set()
    benign_re = re.compile(r"卵巢囊肿|卵泡|内膜异位症|内膜异位")
    side_terms = [("L", "左侧附件"), ("R", "右侧附件"), ("B", "双侧附件")]
    for c in clauses:
        if c.get("section") != "impression":
            continue
        t = compact(c["text"])
        for bm in benign_re.finditer(t):
            prefix = t[max(0, bm.start() - 45):bm.start()]
            nearest = None
            nearest_pos = -1
            for label, term in side_terms:
                pos = prefix.rfind(term)
                if pos > nearest_pos:
                    nearest = label
                    nearest_pos = pos
            if nearest == "L":
                out.add("L")
            elif nearest == "R":
                out.add("R")
            elif nearest == "B":
                out.update({"L", "R"})
    return out

def clause_is_benign_adnexal_side(text: str, benign_sides: set[str]) -> bool:
    sides = adnexal_sides(text)
    return bool(sides) and sides.issubset(benign_sides) and not any(x in compact(text) for x in ("卵巢癌", "恶性", "转移"))


def primary_clause_ids(clauses: Sequence[Dict[str, str]]) -> List[str]:
    ids: List[str] = []
    malignant_context = malignant_primary_context(clauses)
    benign_sides = benign_adnexal_sides(clauses)
    for c in clauses:
        t = compact(c["text"])
        if clause_is_benign_adnexal_side(c["text"], benign_sides):
            continue
        if looks_like_primary_clause(t):
            if malignant_context or any(x in t for x in MALIGNANT_TERMS):
                ids.append(c["id"])
        elif c["section"] == "impression" and any(x in t for x in ("卵巢癌", "盆腔恶性肿瘤", "盆腔多发占位治疗后", "盆腔占位治疗后", "附件区占位治疗后")):
            ids.append(c["id"])
    return unique_keep_order(ids)

def node_kind(text: str) -> List[str]:
    t = compact(text)
    if "淋巴结" not in t and not any(x in t for x in ("闭孔区", "髂外血管旁", "髂内血管旁")):
        return []
    out: List[str] = []
    if any(x in t for x in ("盆腔", "髂外", "髂内", "髂血管", "闭孔", "骶前", "腹主动脉旁", "腹膜后")):
        out.append("pelvic_retroperitoneal_nodes")
    if any(x in t for x in ("心膈角", "肋膈角", "膈脚", "肠系膜区", "上腹", "肝门淋巴结")):
        out.append("upper_abdominal_cardiophrenic_nodes")
    return unique_keep_order(out)


def _node_region_terms(site: str) -> Tuple[str, ...]:
    if site == "pelvic_retroperitoneal_nodes":
        return ("盆腔", "髂外", "髂内", "髂血管", "闭孔", "骶前", "腹主动脉旁", "腹膜后")
    return ("心膈角", "肋膈角", "膈脚", "肠系膜区", "上腹", "肝门")


def _node_segments(text: str) -> List[str]:
    return [x for x in re.split(r"[，,；;。]", compact(text)) if x]


def positive_node_clause(text: str, site: str) -> bool:
    terms = _node_region_terms(site)
    pos_patterns = [
        r"(?<!未)见[^，。；]{0,10}淋巴结",
        r"可见[^，。；]{0,10}淋巴结",
        r"多发淋巴结",
        r"淋巴结[^，。；]{0,12}短径",
        r"淋巴结[^，。；]{0,10}(?:明显)?(?:缩小|减小)",
        r"淋巴结肿大",
    ]
    for seg in _node_segments(text):
        if not any(term in seg for term in terms):
            continue
        if any(re.search(p, seg) for p in pos_patterns):
            return True
    return False


def negative_node_clause(text: str, site: str) -> bool:
    terms = _node_region_terms(site)
    negpat = r"(?:未见|无)[^，。；]{0,15}(?:明显)?(?:增大|肿大)?淋巴结|淋巴结未见(?:明显)?肿大"
    for seg in _node_segments(text):
        if any(term in seg for term in terms) and re.search(negpat, seg):
            return True
    return False

def bowel_relation_from_text(text: str) -> Optional[Tuple[str, int]]:
    t = compact(text)
    bowel_target = r"(?:直肠(?!陷窝)|结肠|小肠|肠管|肠壁|胃壁|胃)"
    # Explicit invasion. '子宫直肠陷窝' is deliberately excluded by target pattern.
    invasion = bool(re.search(rf"(?:侵及|侵犯|累及)[^。；]{{0,20}}{bowel_target}|{bowel_target}[^。；]{{0,8}}(?:受侵|侵犯|累及)|似达(?:黏膜|粘膜)层", t))
    if invasion:
        # '侵及直肠可能/待除外' remains uncertain; direct wall invasion or '似达黏膜层' is definite.
        if any(x in t for x in ("可能", "待除外", "不能除外", "考虑侵及可能")) and not re.search(r"似达(?:黏膜|粘膜)层", t):
            return ("uncertain", 2)
        return ("present", 3)
    if re.search(rf"(?:与|同){bowel_target}[^。；]{{0,12}}(?:分界不清|界限欠清|分界欠清)|{bowel_target}[^。；]{{0,12}}(?:分界不清|界限欠清|分界欠清)", t):
        return ("uncertain", 2)
    if re.search(rf"{bowel_target}[^。；]{{0,12}}(?:密接|邻近|受压)|(?:密接|邻近|受压)[^。；]{{0,12}}{bowel_target}", t):
        return ("uncertain", 1)
    return None


def strongest_bowel_relation(clauses: Sequence[Dict[str, str]]) -> Tuple[str, Optional[int], List[str]]:
    best_grade = -1
    best_status = "not_mentioned"
    ids: List[str] = []
    for c in clauses:
        rel = bowel_relation_from_text(c["text"])
        if rel is None:
            continue
        status, grade = rel
        if grade > best_grade:
            best_grade = grade
            best_status = status
            ids = [c["id"]]
        elif grade == best_grade:
            ids.append(c["id"])
    return best_status, (best_grade if best_grade >= 0 else None), unique_keep_order(ids)


def explicit_largest_location_site(text: str) -> Optional[str]:
    t = compact(text)
    m = re.search(r"(?:较大者|大者|最大(?:者|病灶)?)(?:位于|在)([^，。；;]{1,24})", t)
    if not m:
        return None
    return direct_location_to_site(m.group(1))


def peritoneal_measurement_context(text: str) -> bool:
    """True when a size in this clause can reasonably belong to peritoneal disease, not a primary mass merely touching/involving peritoneum."""
    if not looks_like_primary_clause(text):
        return True
    t = compact(text)
    perit_term = r"(?:盆底腹膜|盆腔腹膜|子宫直肠陷窝|直肠子宫陷窝|大网膜|小网膜|网膜囊|肝胃韧带|小肠系膜|肠系膜根部|结肠系膜|直肠系膜|结肠旁|壁层腹膜|壁腹膜|膈下腹膜|膈肌下缘腹膜|膈顶腹膜|膈腹膜|肝周|肝被膜|肝肾隐窝|脾周|脾旁|肝门)"
    disease = r"(?:增厚|结节|肿块|占位|病灶|饼状)"
    return bool(re.search(perit_term + r"[^。；]{0,35}" + disease + r"[^。；]{0,35}(?:mm|cm)", t))


def safe_size_candidates(site: str, clauses: Sequence[Dict[str, str]]) -> List[Tuple[float, str]]:
    out: List[Tuple[float, str]] = []
    benign_sides = benign_adnexal_sides(clauses) if site == "primary_pelvic_mass" else set()
    for c in clauses:
        text = c["text"]
        sizes = extract_all_sizes_mm(text)
        if not sizes:
            continue
        if site == "primary_pelvic_mass":
            if clause_is_benign_adnexal_side(text, benign_sides):
                continue
            if looks_like_primary_clause(text):
                out.append((max(sizes), c["id"]))
            continue
        if site in {"pelvic_retroperitoneal_nodes", "upper_abdominal_cardiophrenic_nodes"}:
            if positive_node_clause(text, site):
                short = extract_short_axes_mm(text)
                if short:
                    out.append((max(short), c["id"]))
            continue
        if site in PERITONEAL_SITES:
            mapped = mapped_peritoneal_sites(text)
            if site not in mapped or not peritoneal_site_positive(text, site):
                continue
            if not peritoneal_measurement_context(text):
                continue
            if len(mapped) == 1:
                out.append((max(sizes), c["id"]))
            else:
                loc_site = explicit_largest_location_site(text)
                if loc_site == site:
                    out.append((max(sizes), c["id"]))
            continue
        if site == "distant_metastasis":
            t = compact(text)
            if any(x in t for x in ("转移", "转移瘤", "骨转移", "肝转移", "肺转移")) and not any(x in t for x in BENIGN_TERMS):
                out.append((max(sizes), c["id"]))
    return out


def compute_overall_peritoneal_max(clauses: Sequence[Dict[str, str]]) -> Tuple[Optional[float], List[str]]:
    best: Optional[float] = None
    ids: List[str] = []
    for c in clauses:
        if not is_positive_peritoneal_clause(c["text"]):
            continue
        if not peritoneal_measurement_context(c["text"]):
            continue
        sizes = extract_all_sizes_mm(c["text"])
        if not sizes:
            continue
        v = max(sizes)
        if best is None or v > best:
            best = v
            ids = [c["id"]]
        elif v == best:
            ids.append(c["id"])
    return best, unique_keep_order(ids)


def deterministic_fluid(clauses: Sequence[Dict[str, str]], pleural: bool = False) -> Dict[str, Any]:
    positives: List[Tuple[int, str]] = []
    negatives: List[str] = []
    abdo_loc = r"(?:腹盆腔|盆腹腔|腹、盆腔|盆、腹腔|腹腔及盆腔|盆腔及腹腔|腹腔、盆腔|盆腔、腹腔|腹腔|盆腔)"
    for c in clauses:
        t = compact(c["text"])
        if pleural:
            pos = bool(re.search(r"胸腔[^。；]{0,24}(?:可见|见)?[^。；]{0,16}积液|胸水", t))
            neg = bool(re.search(r"胸腔[^。；]{0,12}(?:未见|无)[^。；]{0,12}积液|(?:未见|无)[^。；]{0,12}(?:胸腔积液|胸水)", t))
            # A negative phrase should not be turned positive merely because it contains the word 积液.
            if neg and not re.search(r"胸腔[^。；]{0,12}(?<!未)(?:可见|见)[^。；]{0,12}积液", t):
                pos = False
        else:
            has_loc = bool(re.search(abdo_loc, t))
            pos = False
            neg = False
            if has_loc and "积液" in t:
                neg = bool(re.search(abdo_loc + r"[^。；]{0,10}(?:未见|无)[^。；]{0,12}(?:明显)?积液", t))
                neg = neg or bool(re.search(r"(?:未见|无)[^。；]{0,10}" + abdo_loc + r"[^。；]{0,12}(?:明显)?积液", t))
                # Positive forms: 盆腔见少量积液 / 腹盆腔内可见大量积液 / 盆腔少量积液.
                pos = bool(re.search(abdo_loc + r"[^。；]{0,20}(?<!未)(?:可见|见)[^。；]{0,20}积液", t))
                pos = pos or bool(re.search(abdo_loc + r"[^。；]{0,10}(?:少量|少许|大量)?积液(?:信号|影)?", t) and not neg)
            if re.search(r"(?:未见|无)腹水", t):
                neg = True
            if "腹水" in t and not re.search(r"(?:未见|无)腹水", t):
                pos = True
        if pos:
            if "大量" in t:
                grade = 3
            elif any(x in t for x in ("少量", "少许")):
                grade = 1
            else:
                grade = 2
            positives.append((grade, c["id"]))
        elif neg:
            negatives.append(c["id"])
    if positives:
        grade = max(g for g, _ in positives)
        ids = [cid for g, cid in positives if g == grade]
        return {"status": "present", "grade": grade, "evidence_ids": unique_keep_order(ids)}
    if negatives:
        return {"status": "absent", "grade": 0, "evidence_ids": unique_keep_order(negatives)}
    return {"status": "not_mentioned", "grade": None, "evidence_ids": []}

def change_from_text(text: str) -> str:
    t = compact(text)
    if any(x in t for x in ("显示欠清", "显示不清", "未见明确显示", "无法评价", "不能评价")):
        return "not_evaluable"
    if ("部分" in t and any(x in t for x in ("减小", "缩小", "减少")) and any(x in t for x in ("无著变", "稳定", "相仿"))) or ("囊性" in t and "增大" in t and "实性" in t and any(x in t for x in ("稳定", "无著变"))):
        return "mixed"
    if any(x in t for x in ("新发", "新出现", "新增")):
        return "new"
    if any(x in t for x in ("消失", "完全消退")) and any(x in t for x in ("较前", "前片", "治疗后")):
        return "resolved"
    if any(x in t for x in ("较前明显减小", "较前略减小", "较前减小", "较前缩小", "较前略缩小", "范围较前减小", "较前减少", "减小、减少", "明显缩小", "好转")) or re.search(r"[（(](?:略|明显)?(?:减小|缩小|减少)[）)]", t):
        return "decreased"
    if (any(x in t for x in ("较前增大", "较前增加", "增大", "进展")) or re.search(r"[（(](?:略|明显)?(?:增大|增加|进展)[）)]", t)) and "囊性部分增大" not in t and not any(x in t for x in ("未见增大", "无增大")):
        return "increased"
    if any(x in t for x in ("较前无著变", "未见著变", "无著变", "较前相仿", "稳定")):
        return "stable"
    return "not_stated"


def clause_relevant_to_change(site: str, text: str) -> bool:
    t = compact(text)
    if site == "primary_pelvic_mass":
        return looks_like_primary_clause(text) or any(x in t for x in ("卵巢癌治疗后", "盆腔多发占位", "盆腔占位"))
    if site in PERITONEAL_SITES:
        return site in positive_peritoneal_sites(text)
    if site in {"pelvic_retroperitoneal_nodes", "upper_abdominal_cardiophrenic_nodes"}:
        return positive_node_clause(text, site) or ("淋巴结" in t and any(x in t for x in ("治疗后", "转移")) and not negative_node_clause(text, site))
    if site == "bowel_stomach_involvement":
        return bowel_relation_from_text(text) is not None or bool(re.search(r"(?:侵及|侵犯|累及)(?:直肠|结肠|小肠|肠壁|胃)", t))
    if site == "distant_metastasis":
        return any(x in t for x in ("转移", "骨转移", "肝转移", "肺转移"))
    return False


def bowel_change_from_clause(clause: Dict[str, str]) -> str:
    t = compact(clause["text"])
    target = r"(?:直肠(?!陷窝)|结肠|小肠|肠壁|肠管|胃壁|胃)"
    invasion = rf"(?:侵及|侵犯|累及)[^。；]{{0,20}}{target}|{target}[^。；]{{0,8}}(?:受侵|侵犯|累及)"
    # Require the change phrase to be syntactically tied to bowel involvement itself.
    tied = re.search(r"(?:" + invasion + r")" + r"[^。；]{0,20}(?:（|\()(?:较前)?(?:减小|缩小|减少|稳定|无著变|好转|增大)(?:）|\))", t)
    tied2 = re.search(r"(?:" + invasion + r")" + r"[^。；]{0,16}(?:较前(?:减小|缩小|减少|增大|无著变)|明显缩小|好转)", t)
    if not (tied or tied2):
        return "not_stated"
    return change_from_text(t)


def determine_site_change(site: str, post_clauses: Sequence[Dict[str, str]], post_status: str) -> Tuple[str, List[str]]:
    if post_status == "not_evaluable":
        return "not_evaluable", []
    changes: List[Tuple[str, str]] = []
    benign_primary_sides = benign_adnexal_sides(post_clauses) if site == "primary_pelvic_mass" else set()
    for c in post_clauses:
        if site == "primary_pelvic_mass" and clause_is_benign_adnexal_side(c["text"], benign_primary_sides):
            continue
        if site == "bowel_stomach_involvement":
            ch = bowel_change_from_clause(c)
            if ch == "not_stated":
                continue
        else:
            if not clause_relevant_to_change(site, c["text"]):
                continue
            ch = change_from_text(c["text"])
            if ch == "not_stated":
                continue
        if site in {"pelvic_retroperitoneal_nodes", "upper_abdominal_cardiophrenic_nodes"}:
            if re.search(r"部分[^。；]{0,10}(?:较前)?(?:减小|缩小|减少)", compact(c["text"])) and ch == "decreased":
                ch = "mixed"
        if site in PERITONEAL_SITES:
            mapped = positive_peritoneal_sites(c["text"])
            t = compact(c["text"])
            # '部分减小' across multiple distinct anatomic sites does not identify which site changed.
            if re.search(r"部分[^。；]{0,10}(?:较前)?(?:减小|缩小|减少)", t) and len(mapped) > 1 and ch in {"decreased", "mixed"}:
                continue
        changes.append((ch, c["id"]))
    if not changes:
        return "not_stated", []
    vals = [x[0] for x in changes]
    ids = unique_keep_order(x[1] for x in changes)
    if "mixed" in vals:
        return "mixed", ids
    unique_vals = set(vals)
    if len(unique_vals) == 1:
        return vals[0], ids
    # not_evaluable plus a definite response -> do not silently choose one.
    return "mixed", ids


def is_not_evaluable_wording(text: str) -> bool:
    t = compact(text)
    return any(x in t for x in ("显示欠清", "显示不清", "未见明确显示", "无法评价", "不能评价"))


def benign_only_clause(text: str) -> bool:
    t = compact(text)
    return any(x in t for x in BENIGN_TERMS) and not any(x in t for x in ("转移", "恶性", "癌"))


def add_evidence(site_obj: Dict[str, Any], ids: Iterable[str], limit: int = 5) -> None:
    site_obj["evidence_ids"] = unique_keep_order(list(site_obj.get("evidence_ids") or []) + list(ids))[:limit]


def postprocess_report(model_result: Dict[str, Any], clauses: Sequence[Dict[str, str]]) -> Tuple[Dict[str, Any], List[str]]:
    """Apply source-backed deterministic corrections without using the paired report."""
    out = json.loads(json.dumps(model_result, ensure_ascii=False))
    cmap = clause_map(clauses)
    warnings: List[str] = []

    # Filter evidence IDs that are clearly benign-only for primary/distant disease.
    for site in ("primary_pelvic_mass", "distant_metastasis"):
        kept = []
        for cid in out["sites"][site]["evidence_ids"]:
            txt = cmap[cid]["text"]
            if benign_only_clause(txt):
                warnings.append(f"{site}: removed benign-only evidence {cid}")
            else:
                kept.append(cid)
        out["sites"][site]["evidence_ids"] = kept

    # Peritoneal anatomy rescue / negative / not-evaluable handling.
    for site in PERITONEAL_SITES:
        positive_ids: List[str] = []
        negative_ids: List[str] = []
        ne_ids: List[str] = []
        grades: List[int] = []
        for c in clauses:
            mapped = mapped_peritoneal_sites(c["text"])
            if site not in mapped:
                continue
            if is_not_evaluable_wording(c["text"]):
                ne_ids.append(c["id"])
                continue
            if is_negative_peritoneal_clause(c["text"]):
                negative_ids.append(c["id"])
                continue
            if peritoneal_site_positive(c["text"], site):
                positive_ids.append(c["id"])
                g = peritoneal_burden_from_clause(c["text"], site)
                if g is not None:
                    grades.append(g)
        s = out["sites"][site]
        if positive_ids:
            if s["status"] in {"not_mentioned", "absent", "not_evaluable"}:
                warnings.append(f"{site}: Python anatomy rescue -> present")
            s["status"] = "present"
            add_evidence(s, positive_ids)
            if grades:
                # Prefer the strongest source-supported grade over a weaker/missing model grade.
                src_grade = max(grades)
                if s["burden_grade"] != src_grade:
                    if s["burden_grade"] is not None:
                        warnings.append(f"{site}: burden grade corrected {s['burden_grade']} -> {src_grade} from source wording")
                    s["burden_grade"] = src_grade
        elif ne_ids:
            s["status"] = "not_evaluable"
            s["burden_grade"] = None
            s["max_size_mm"] = None
            add_evidence(s, ne_ids)
        elif negative_ids:
            s["status"] = "absent"
            s["burden_grade"] = 0
            s["max_size_mm"] = None
            add_evidence(s, negative_ids)
        elif s["status"] == "absent" and not s["evidence_ids"]:
            # Never accept unsupported absent from the model.
            s["status"] = "not_mentioned"
            s["burden_grade"] = None
            warnings.append(f"{site}: unsupported absent downgraded to not_mentioned")

    # Primary-mass rescue. This fixes omissions while remaining generic.
    pids = primary_clause_ids(clauses)
    primary = out["sites"]["primary_pelvic_mass"]
    benign_sides = benign_adnexal_sides(clauses)
    primary_like_ids = [c["id"] for c in clauses if looks_like_primary_clause(c["text"]) and not clause_is_benign_adnexal_side(c["text"], benign_sides)]
    if pids and malignant_primary_context(clauses):
        if primary["status"] != "present":
            warnings.append("primary_pelvic_mass: source-backed malignant primary rescue -> present")
        primary["status"] = "present"
        add_evidence(primary, pids)
    elif primary_like_ids and primary["status"] in {"not_mentioned", "absent"}:
        primary["status"] = "uncertain"
        add_evidence(primary, primary_like_ids)
        warnings.append("primary_pelvic_mass: source shows an adnexal/pelvic lesion but malignancy is not explicit in this report -> uncertain")
    elif primary["status"] == "present" and primary["evidence_ids"] and all(benign_only_clause(cmap[cid]["text"]) for cid in primary["evidence_ids"]):
        primary["status"] = "not_mentioned"
        primary["evidence_ids"] = []
        primary["max_size_mm"] = None
        warnings.append("primary_pelvic_mass: benign-only model positive removed")

    # Node negation scope: any explicit positive node in the composite region wins over
    # a separate negative statement about another subregion.
    for site in ("pelvic_retroperitoneal_nodes", "upper_abdominal_cardiophrenic_nodes"):
        pos = [c["id"] for c in clauses if positive_node_clause(c["text"], site)]
        neg = [c["id"] for c in clauses if negative_node_clause(c["text"], site)]
        s = out["sites"][site]
        if pos:
            if s["status"] != "present":
                warnings.append(f"{site}: positive-node clause overrides regional negation/model status")
            s["status"] = "present"
            add_evidence(s, pos)
        elif neg:
            s["status"] = "absent"
            s["max_size_mm"] = None
            add_evidence(s, neg)
        elif s["status"] == "absent" and not s["evidence_ids"]:
            s["status"] = "not_mentioned"
            warnings.append(f"{site}: unsupported absent downgraded to not_mentioned")

    # Bowel/stomach relationship is fully guarded in Python.
    bstatus, bgrade, bids = strongest_bowel_relation(clauses)
    bowel = out["sites"]["bowel_stomach_involvement"]
    if bgrade is not None:
        if bowel["status"] != bstatus or bowel["relation_grade"] != bgrade:
            warnings.append(f"bowel_stomach_involvement: Python relation guard -> {bstatus}/grade{bgrade}")
        bowel["status"] = bstatus
        bowel["relation_grade"] = bgrade
        bowel["max_size_mm"] = None
        add_evidence(bowel, bids)
    else:
        # A model must not upgrade bowel invasion without source wording.
        if bowel["status"] == "present" or bowel["relation_grade"] == 3:
            warnings.append("bowel_stomach_involvement: unsupported invasion downgraded")
            bowel["status"] = "not_mentioned"
            bowel["relation_grade"] = None
            bowel["evidence_ids"] = []
        elif bowel["status"] == "absent" and not bowel["evidence_ids"]:
            bowel["status"] = "not_mentioned"
            bowel["relation_grade"] = None

    # Distant metastasis benign guard.
    distant = out["sites"]["distant_metastasis"]
    if distant["evidence_ids"] and all(benign_only_clause(cmap[cid]["text"]) for cid in distant["evidence_ids"]):
        distant["status"] = "not_mentioned"
        distant["max_size_mm"] = None
        distant["evidence_ids"] = []
        warnings.append("distant_metastasis: benign-only evidence removed")

    # Site-specific size: validate model value against safe source candidates, then rescue.
    for site in SITE_ORDER:
        s = out["sites"][site]
        if site == "bowel_stomach_involvement":
            s["max_size_mm"] = None
            continue
        candidates = safe_size_candidates(site, clauses)
        candidate_values = [v for v, _ in candidates]
        model_v = as_float(s.get("max_size_mm"))
        if model_v is not None:
            if not any(abs(model_v - v) <= 0.11 for v in candidate_values):
                warnings.append(f"{site}: blocked unsupported/group/other-entity size {model_v} mm")
                s["max_size_mm"] = None
        if s.get("max_size_mm") is None and candidates and s["status"] in {"present", "uncertain"}:
            v, cid = max(candidates, key=lambda x: x[0])
            s["max_size_mm"] = v
            add_evidence(s, [cid])
        if s["status"] in {"absent", "not_mentioned", "not_evaluable"}:
            s["max_size_mm"] = None

    # Overall peritoneal maximum is computed directly from source; never copied down to sites.
    group_v, group_ids = compute_overall_peritoneal_max(clauses)
    out["overall_peritoneal"] = {"group_max_size_mm": group_v, "evidence_ids": group_ids}

    # Fluid is deterministic because it is high-value and linguistically simple.
    out["abdominopelvic_fluid"] = deterministic_fluid(clauses, pleural=False)
    out["pleural_effusion"] = deterministic_fluid(clauses, pleural=True)

    # Clean burden/relation fields by site type/status.
    for site in SITE_ORDER:
        s = out["sites"][site]
        if site not in PERITONEAL_SITES:
            s["burden_grade"] = None
        elif s["status"] == "absent":
            s["burden_grade"] = 0
        elif s["status"] in {"not_mentioned", "not_evaluable"}:
            s["burden_grade"] = None
        if site != "bowel_stomach_involvement":
            s["relation_grade"] = None
        if s["status"] == "not_mentioned" and not s["evidence_ids"]:
            s["change_hint"] = "not_stated"

    return out, unique_keep_order(warnings)



# -----------------------------------------------------------------------------
# V6.1 FINAL deterministic hotfix layer.
# The LLM SYSTEM_PROMPT and PROMPT_TEMPLATE above remain unchanged from V6.
# This layer only tightens source/object scope after the V6 source-backed record
# has already passed the V6 internal consistency audit.
# -----------------------------------------------------------------------------
V61_BOWEL_TARGET_RE = re.compile(
    r"(?:直肠|肠管|肠壁|结肠(?!旁|系膜)|小肠(?!系膜)|乙状结肠|盲肠|回肠|空肠|胃(?!韧带)|胃壁)"
)
V61_MESENTERIC_PERITONEAL_RE = re.compile(
    r"(?:小肠系膜|结肠系膜|肠系膜)(?:根部)?(?:腹膜)?[^。；;]{0,20}(?:增厚|结节|肿块|占位|种植|转移)"
)
V61_DISTANT_TARGET_RE = re.compile(
    r"(?:肺|胸膜|骨|骨骼|椎体|椎管|脑|肾上腺|腹壁肌|肌肉内|皮下|"
    r"肝(?:S\d+|实质|内|脏|顶(?!被膜))|"
    r"脾(?!周|旁|胃韧带|结肠韧带)|"
    r"肾(?!隐窝)|远处)"
)
V61_BENIGN_RE = re.compile(
    r"(?:囊肿|腺瘤|增生|脂肪瘤|血管平滑肌脂肪瘤|副脾|结石|纳囊|那囊|肌瘤|息肉)"
)
V61_LOCAL_ONLY_RE = re.compile(
    r"(?:腹膜|大网膜|小网膜|网膜|系膜|盆底|结肠旁|壁腹膜|壁层腹膜|"
    r"膈腹膜|膈下腹膜|肝被膜|肝肾隐窝|脾周|韧带|淋巴结)"
)


def v61_site_map(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {s.get("site"): s for s in record.get("sites", []) if s.get("site")}


def v61_phase_evidence(site: Dict[str, Any], phase: str) -> str:
    return str(site.get(f"{phase}_evidence") or "")


def v61_clear_phase(site: Dict[str, Any], phase: str) -> None:
    site[f"{phase}_status"] = "not_mentioned"
    site[f"{phase}_burden_grade"] = None
    site[f"{phase}_max_size_mm"] = None
    site[f"{phase}_relation_grade"] = None
    site[f"{phase}_evidence_ids"] = []
    site[f"{phase}_evidence"] = ""


def v61_is_node_centered_mesenteric_only(text: str) -> bool:
    text = text or ""
    if "淋巴结" not in text:
        return False
    if "腹膜" in text:
        return False
    masked = re.sub(r"[^。；;]{0,18}淋巴结[^。；;]{0,24}", " ", text)
    if V61_MESENTERIC_PERITONEAL_RE.search(masked):
        return False
    return True


def v61_distant_change_supported(site: Dict[str, Any]) -> bool:
    text = " ".join([v61_phase_evidence(site, "pre"), v61_phase_evidence(site, "post")]).strip()
    if not text:
        return False
    if V61_LOCAL_ONLY_RE.search(text) and not V61_DISTANT_TARGET_RE.search(text):
        return False
    if not V61_DISTANT_TARGET_RE.search(text):
        return False
    segments = [seg.strip() for seg in re.split(r"[。；;]", text) if seg.strip()]
    distant_segments = [seg for seg in segments if V61_DISTANT_TARGET_RE.search(seg)]
    if distant_segments and all(V61_BENIGN_RE.search(seg) for seg in distant_segments):
        return False
    return True


def apply_v61_hotfix(record: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    r = deepcopy(record)
    pid = str(r.get("patient_id", ""))
    sm = v61_site_map(r)
    audit: List[Dict[str, str]] = []

    def log(rule: str, site: str, field: str, old: Any, new: Any, note: str) -> None:
        audit.append({
            "patient_id": pid,
            "rule": rule,
            "site": site,
            "field": field,
            "old": json.dumps(old, ensure_ascii=False),
            "new": json.dumps(new, ensure_ascii=False),
            "note": note,
        })

    # B. Bowel/stomach relation must bind to an explicit bowel/stomach object.
    bowel = sm.get("bowel_stomach_involvement")
    if bowel:
        for phase in ("pre", "post"):
            status = bowel.get(f"{phase}_status")
            if status in {"present", "uncertain"}:
                txt = v61_phase_evidence(bowel, phase)
                if not V61_BOWEL_TARGET_RE.search(txt):
                    old = {"status": status, "relation_grade": bowel.get(f"{phase}_relation_grade"), "evidence": txt}
                    v61_clear_phase(bowel, phase)
                    log("B_bowel_target_guard", "bowel_stomach_involvement", phase, old,
                        {"status": "not_mentioned"},
                        "Relation wording lacked an explicit bowel/stomach target.")

    # C. A lymph-node-centered clause cannot open mesenteric-peritoneal disease.
    for site_name in ("small_bowel_mesenteric", "colonic_mesenteric"):
        s = sm.get(site_name)
        if not s:
            continue
        for phase in ("pre", "post"):
            status = s.get(f"{phase}_status")
            if status in {"present", "uncertain"}:
                txt = v61_phase_evidence(s, phase)
                if v61_is_node_centered_mesenteric_only(txt):
                    old = {"status": status, "burden_grade": s.get(f"{phase}_burden_grade"),
                           "max_size_mm": s.get(f"{phase}_max_size_mm"), "evidence": txt}
                    v61_clear_phase(s, phase)
                    log("C_mesenteric_node_guard", site_name, phase, old,
                        {"status": "not_mentioned"},
                        "Evidence was lymph-node-centered without independent mesenteric-peritoneal disease wording.")

    # A. Distant-metastasis change must have distant-organ-specific source support.
    distant = sm.get("distant_metastasis")
    if distant and distant.get("radiologist_reported_change") not in {None, "not_stated"}:
        if not v61_distant_change_supported(distant):
            old = distant.get("radiologist_reported_change")
            distant["radiologist_reported_change"] = "not_stated"
            log("A_distant_change_scope", "distant_metastasis", "radiologist_reported_change",
                old, "not_stated",
                "Change lacked distant-organ-specific source support and/or came from local/peritoneal/nodal wording.")

    # D. If both phases are not_mentioned, a site-level change is impossible.
    for s in r.get("sites", []):
        if (s.get("pre_status") == "not_mentioned" and s.get("post_status") == "not_mentioned"
                and s.get("radiologist_reported_change") not in {None, "not_stated"}):
            old = s.get("radiologist_reported_change")
            s["radiologist_reported_change"] = "not_stated"
            log("D_both_not_mentioned_change_guard", str(s.get("site")), "radiologist_reported_change",
                old, "not_stated", "PRE and POST were both not_mentioned.")

    r["_v6_1_hotfix"] = {
        "version": "V6.1-FINAL-HOTFIX",
        "n_actions": len(audit),
        "rules": ["A_distant_change_scope", "B_bowel_target_guard", "C_mesenteric_node_guard", "D_both_not_mentioned_change_guard"],
    }
    return r, audit


def validate_v61_final(patient: Dict[str, Any], base_validation: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    if not base_validation.get("passed", False):
        errors.extend([f"V6_BASE: {x}" for x in base_validation.get("errors", [])])
    warnings.extend([f"V6_BASE: {x}" for x in base_validation.get("warnings", [])])
    sm = v61_site_map(patient)

    bowel = sm.get("bowel_stomach_involvement")
    if bowel:
        for phase in ("pre", "post"):
            if bowel.get(f"{phase}_status") in {"present", "uncertain"}:
                if not V61_BOWEL_TARGET_RE.search(v61_phase_evidence(bowel, phase)):
                    errors.append(f"bowel_stomach_involvement/{phase}: lacks explicit bowel/stomach target")

    for site_name in ("small_bowel_mesenteric", "colonic_mesenteric"):
        s = sm.get(site_name)
        if s:
            for phase in ("pre", "post"):
                if s.get(f"{phase}_status") in {"present", "uncertain"} and v61_is_node_centered_mesenteric_only(v61_phase_evidence(s, phase)):
                    errors.append(f"{site_name}/{phase}: node-centered evidence opened mesenteric disease")

    distant = sm.get("distant_metastasis")
    if distant and distant.get("radiologist_reported_change") not in {None, "not_stated"}:
        if not v61_distant_change_supported(distant):
            errors.append("distant_metastasis: unsupported change scope")

    for s in patient.get("sites", []):
        if (s.get("pre_status") == "not_mentioned" and s.get("post_status") == "not_mentioned"
                and s.get("radiologist_reported_change") not in {None, "not_stated"}):
            errors.append(f"{s.get('site')}: both phases not_mentioned but change={s.get('radiologist_reported_change')}")

    return {
        "passed": len(errors) == 0,
        "scope": "V6 source-rule audit plus V6.1 deterministic scope guards; not human gold-standard accuracy",
        "errors": unique_keep_order(errors),
        "warnings": unique_keep_order(warnings),
    }


def append_v61_hotfix_audit(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    if not rows:
        return
    fields = ["patient_id", "rule", "site", "field", "old", "new", "note"]
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        for row in rows:
            w.writerow(row)


def evidence_text(ids: Sequence[str], clauses: Sequence[Dict[str, str]]) -> str:
    cmap = clause_map(clauses)
    return " ".join(cmap[cid]["text"] for cid in ids if cid in cmap)


def report_incomplete(report_text: str) -> bool:
    return bool(INCOMPLETE_RE.search(report_text or ""))


def build_final_patient(
    patient_id: str,
    pre_pp: Dict[str, Any],
    post_pp: Dict[str, Any],
    pre_clauses: Sequence[Dict[str, str]],
    post_clauses: Sequence[Dict[str, str]],
    pre_text: str,
    post_text: str,
    pre_warnings: Sequence[str],
    post_warnings: Sequence[str],
) -> Dict[str, Any]:
    pre_map = clause_map(pre_clauses)
    post_map = clause_map(post_clauses)
    pre_inc = report_incomplete(pre_text)
    post_inc = report_incomplete(post_text)
    notes: List[str] = []
    if pre_inc:
        notes.append("PRE: source capture may be truncated/obscured")
    if post_inc:
        notes.append("POST: source capture may be truncated/obscured")

    result: Dict[str, Any] = {
        "patient_id": str(patient_id),
        "data_quality": {
            "pre_report_incomplete": pre_inc,
            "post_report_incomplete": post_inc,
            "notes": "; ".join(notes),
        },
        "overall_peritoneal": {
            "pre_group_max_size_mm": pre_pp["overall_peritoneal"]["group_max_size_mm"],
            "post_group_max_size_mm": post_pp["overall_peritoneal"]["group_max_size_mm"],
            "pre_evidence_ids": pre_pp["overall_peritoneal"]["evidence_ids"],
            "post_evidence_ids": post_pp["overall_peritoneal"]["evidence_ids"],
            "pre_evidence": evidence_text(pre_pp["overall_peritoneal"]["evidence_ids"], pre_clauses),
            "post_evidence": evidence_text(post_pp["overall_peritoneal"]["evidence_ids"], post_clauses),
        },
        "sites": [],
    }

    for site in SITE_ORDER:
        a = pre_pp["sites"][site]
        b = post_pp["sites"][site]
        change, change_ids = determine_site_change(site, post_clauses, b["status"])
        # If deterministic scan found no change but model had an explicit hint, do NOT promote it.
        # Instead record a warning for review; this keeps evidence source-grounded.
        if change == "not_stated" and b.get("change_hint") not in {None, "not_stated"}:
            post_warnings = list(post_warnings) + [f"{site}: model change_hint={b.get('change_hint')} not promoted without deterministic scope match"]
        # Add source clause(s) supporting the final change to POST evidence.
        post_ids = unique_keep_order(list(b["evidence_ids"]) + change_ids)
        result["sites"].append({
            "site": site,
            "pre_status": a["status"],
            "post_status": b["status"],
            "pre_burden_grade": a["burden_grade"],
            "post_burden_grade": b["burden_grade"],
            "pre_max_size_mm": a["max_size_mm"],
            "post_max_size_mm": b["max_size_mm"],
            "pre_relation_grade": a["relation_grade"],
            "post_relation_grade": b["relation_grade"],
            "radiologist_reported_change": change,
            "pre_evidence_ids": a["evidence_ids"],
            "post_evidence_ids": post_ids,
            "pre_evidence": evidence_text(a["evidence_ids"], pre_clauses),
            "post_evidence": evidence_text(post_ids, post_clauses),
        })

    for key in ("abdominopelvic_fluid", "pleural_effusion"):
        dst = {}
        for label, src, cls in (("pre", pre_pp[key], pre_clauses), ("post", post_pp[key], post_clauses)):
            dst[label] = {
                "status": src["status"],
                "grade": src["grade"],
                "evidence_ids": src["evidence_ids"],
                "evidence": evidence_text(src["evidence_ids"], cls),
            }
        result[key] = dst

    result["_python_warnings"] = {
        "pre": unique_keep_order(pre_warnings),
        "post": unique_keep_order(post_warnings),
    }
    return result


def get_site_record(patient: Dict[str, Any], site: str) -> Dict[str, Any]:
    for s in patient["sites"]:
        if s["site"] == site:
            return s
    raise KeyError(site)


def validate_patient(
    patient: Dict[str, Any],
    pre_clauses: Sequence[Dict[str, str]],
    post_clauses: Sequence[Dict[str, str]],
) -> Dict[str, Any]:
    """Internal consistency + deterministic source-rule audit. NOT human gold-standard accuracy."""
    errors: List[str] = []
    warnings: List[str] = []

    for site in SITE_ORDER:
        s = get_site_record(patient, site)
        for tp in ("pre", "post"):
            status = s[f"{tp}_status"]
            if status not in ALLOWED_STATUS:
                errors.append(f"{site}/{tp}: invalid status {status}")
            if status in {"absent", "not_mentioned", "not_evaluable"} and s[f"{tp}_max_size_mm"] is not None:
                errors.append(f"{site}/{tp}: size retained with status={status}")
            if site in PERITONEAL_SITES:
                g = s[f"{tp}_burden_grade"]
                if status == "absent" and g != 0:
                    errors.append(f"{site}/{tp}: absent peritoneal site must have burden 0")
                if status in {"not_mentioned", "not_evaluable"} and g is not None:
                    errors.append(f"{site}/{tp}: burden present with status={status}")
        if s["radiologist_reported_change"] not in ALLOWED_CHANGE:
            errors.append(f"{site}: invalid change enum")

    # Source-rule checks that specifically protect against V5 failure modes.
    for tp, clauses in (("pre", pre_clauses), ("post", post_clauses)):
        # Positive peritoneal clauses should be represented by their mapped sites.
        for c in clauses:
            if not is_positive_peritoneal_clause(c["text"]) or is_not_evaluable_wording(c["text"]):
                continue
            for site in positive_peritoneal_sites(c["text"]):
                s = get_site_record(patient, site)
                if s[f"{tp}_status"] not in {"present", "uncertain"}:
                    errors.append(f"{site}/{tp}: source-positive peritoneal clause {c['id']} not represented")

        # Positive node in a composite compartment must never end as absent.
        for node_site in ("pelvic_retroperitoneal_nodes", "upper_abdominal_cardiophrenic_nodes"):
            if any(positive_node_clause(c["text"], node_site) for c in clauses):
                if get_site_record(patient, node_site)[f"{tp}_status"] == "absent":
                    errors.append(f"{node_site}/{tp}: positive node overwritten by negation")

        # Explicit bowel invasion must be present/grade3; ambiguous interface must not be grade3.
        bs, bg, _ = strongest_bowel_relation(clauses)
        br = get_site_record(patient, "bowel_stomach_involvement")
        if bg == 3 and (br[f"{tp}_status"] != "present" or br[f"{tp}_relation_grade"] != 3):
            errors.append(f"bowel_stomach_involvement/{tp}: explicit invasion not retained")
        if bg in {1, 2} and br[f"{tp}_relation_grade"] == 3:
            errors.append(f"bowel_stomach_involvement/{tp}: ambiguous relation upgraded to invasion")

        # Primary malignant mass with an explicit source measurement should not lose size.
        pcands = safe_size_candidates("primary_pelvic_mass", clauses)
        if pcands and malignant_primary_context(clauses):
            pr = get_site_record(patient, "primary_pelvic_mass")
            if pr[f"{tp}_status"] != "present":
                errors.append(f"primary_pelvic_mass/{tp}: malignant primary source omitted")
            if pr[f"{tp}_max_size_mm"] is None:
                errors.append(f"primary_pelvic_mass/{tp}: source measurement omitted")

    # Explicit POST change with clear site scope should be captured.
    for site in SITE_ORDER:
        expected, _ = determine_site_change(site, post_clauses, get_site_record(patient, site)["post_status"])
        actual = get_site_record(patient, site)["radiologist_reported_change"]
        if expected != actual:
            errors.append(f"{site}: deterministic change mismatch expected={expected}, actual={actual}")

    # Report damage is a data-quality warning, not a validation failure.
    if patient["data_quality"]["pre_report_incomplete"]:
        warnings.append("PRE report contains an explicit truncation/obscuration marker")
    if patient["data_quality"]["post_report_incomplete"]:
        warnings.append("POST report contains an explicit truncation/obscuration marker")

    warnings.extend(patient.get("_python_warnings", {}).get("pre", []))
    warnings.extend(patient.get("_python_warnings", {}).get("post", []))
    return {
        "passed": len(errors) == 0,
        "scope": "internal_consistency_and_source_rule_audit_only_not_gold_standard_accuracy",
        "errors": unique_keep_order(errors),
        "warnings": unique_keep_order(warnings),
    }


REVIEW_FIELDS = [
    "patient_id", "site", "site_cn",
    "pre_status", "post_status",
    "pre_burden_grade", "post_burden_grade",
    "pre_max_size_mm", "post_max_size_mm",
    "pre_relation_grade", "post_relation_grade",
    "radiologist_reported_change",
    "pre_evidence_ids", "post_evidence_ids", "pre_evidence", "post_evidence",
    "pre_group_max_size_mm", "post_group_max_size_mm",
    "pre_fluid_status", "pre_fluid_grade", "pre_fluid_evidence",
    "post_fluid_status", "post_fluid_grade", "post_fluid_evidence",
    "pre_pleural_status", "pre_pleural_grade", "pre_pleural_evidence",
    "post_pleural_status", "post_pleural_grade", "post_pleural_evidence",
    "pre_report_incomplete", "post_report_incomplete",
    "validation_passed", "validation_scope", "n_errors", "n_warnings", "validation_messages",
]


def write_review_csv(patients: Sequence[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        w.writeheader()
        for p in patients:
            v = p.get("_validation_v6", {})
            messages = list(v.get("errors", [])) + list(v.get("warnings", []))
            fluid = p["abdominopelvic_fluid"]
            pleural = p["pleural_effusion"]
            for s in p["sites"]:
                row = {
                    "patient_id": p["patient_id"],
                    "site": s["site"],
                    "site_cn": SITE_CN[s["site"]],
                    "pre_status": s["pre_status"],
                    "post_status": s["post_status"],
                    "pre_burden_grade": s["pre_burden_grade"],
                    "post_burden_grade": s["post_burden_grade"],
                    "pre_max_size_mm": s["pre_max_size_mm"],
                    "post_max_size_mm": s["post_max_size_mm"],
                    "pre_relation_grade": s["pre_relation_grade"],
                    "post_relation_grade": s["post_relation_grade"],
                    "radiologist_reported_change": s["radiologist_reported_change"],
                    "pre_evidence_ids": ";".join(s["pre_evidence_ids"]),
                    "post_evidence_ids": ";".join(s["post_evidence_ids"]),
                    "pre_evidence": s["pre_evidence"],
                    "post_evidence": s["post_evidence"],
                    "pre_group_max_size_mm": p["overall_peritoneal"]["pre_group_max_size_mm"],
                    "post_group_max_size_mm": p["overall_peritoneal"]["post_group_max_size_mm"],
                    "pre_fluid_status": fluid["pre"]["status"],
                    "pre_fluid_grade": fluid["pre"]["grade"],
                    "pre_fluid_evidence": fluid["pre"]["evidence"],
                    "post_fluid_status": fluid["post"]["status"],
                    "post_fluid_grade": fluid["post"]["grade"],
                    "post_fluid_evidence": fluid["post"]["evidence"],
                    "pre_pleural_status": pleural["pre"]["status"],
                    "pre_pleural_grade": pleural["pre"]["grade"],
                    "pre_pleural_evidence": pleural["pre"]["evidence"],
                    "post_pleural_status": pleural["post"]["status"],
                    "post_pleural_grade": pleural["post"]["grade"],
                    "post_pleural_evidence": pleural["post"]["evidence"],
                    "pre_report_incomplete": p["data_quality"]["pre_report_incomplete"],
                    "post_report_incomplete": p["data_quality"]["post_report_incomplete"],
                    "validation_passed": v.get("passed", False),
                    "validation_scope": v.get("scope", ""),
                    "n_errors": len(v.get("errors", [])),
                    "n_warnings": len(v.get("warnings", [])),
                    "validation_messages": " | ".join(messages),
                }
                w.writerow(row)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def detect_duplicate_pairs(pairs: Dict[str, Dict[str, str]], selected: Sequence[str]) -> List[List[str]]:
    by_hash: Dict[str, List[str]] = defaultdict(list)
    for pid in selected:
        pair = pairs.get(pid, {})
        if "a" not in pair or "b" not in pair:
            continue
        h = sha256_text(pair["a"] + "\n---PAIR---\n" + pair["b"])
        by_hash[h].append(pid)
    return [ids for ids in by_hash.values() if len(ids) > 1]


def sort_patient_ids(ids: Iterable[str]) -> List[str]:
    return sorted(ids, key=lambda x: int(x) if str(x).isdigit() else str(x))


def build_manifest(
    *, input_path: Path, model: str, base_url: str, selected: Sequence[str],
    completed: Sequence[Dict[str, Any]], run_errors: Dict[str, str],
    started: str, finished: Optional[str], duplicates: Sequence[Sequence[str]], status: str,
) -> Dict[str, Any]:
    n_failed = sum(1 for p in completed if not p.get("_validation_v6", {}).get("passed", False))
    return {
        "version": VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "status": status,
        "run_started_utc": started,
        "run_finished_utc": finished,
        "input_file": str(input_path),
        "input_sha256": sha256_bytes(input_path.read_bytes()),
        "model": model,
        "base_url": base_url,
        "temperature": TEMPERATURE,
        "requested_patient_ids": list(selected),
        "n_patients_completed": len(completed),
        "n_passed_internal_validation": len(completed) - n_failed,
        "n_failed_internal_validation": n_failed,
        "validation_scope": "internal_consistency_and_source_rule_audit_only_not_gold_standard_accuracy",
        "run_errors": run_errors,
        "duplicate_report_pairs": [list(x) for x in duplicates],
        "site_order": SITE_ORDER,
        "notes": [
            "PRE/POST independently extracted",
            "LLM returns clause IDs only for evidence; evidence text is copied from source by Python",
            "No patient-specific answers are hard-coded",
            "V6 anatomy rescue: Douglas/pelvic, paracolic/wall, diaphragmatic, upper-abdominal synonyms",
            "V6 group/site size guard: group maximum is never copied to a site unless largest location is explicit",
            "V6 primary-mass rescue also rescues a source-supported size",
            "V6 bowel guard: contact/compression < unclear interface < explicit invasion; Douglas pouch is not rectal invasion",
            "V6 node negation is subregion-scoped: a positive iliac/pelvic node is not erased by para-aortic negative wording",
            "V6 change uses deterministic source scope; enumerated shared predicates can propagate, ambiguous partial multi-site change cannot",
            "V6 report-incomplete detection only targets UI/capture truncation/obscuration language",
            "V6 fluid/pleural extraction is deterministic",
            "Built-in generic unit tests run before any API call",
            "Validation passed is NOT factual accuracy against a human gold standard",
            "V6.1 FINAL: Qwen extraction prompt is unchanged from V6; only deterministic scope guards were added",
            "V6.1 FINAL: supports both bare and bracketed patient headers",
            "V6.1 FINAL: contains no surgical-result labels, no endpoint classifier, and no endpoint scoring logic",
        ],
    }


def run_internal_unit_tests() -> None:
    # 1) Clinical '截断' is not report truncation; explicit UI marker is.
    assert not report_incomplete("右侧输尿管于病灶处截断。")
    assert report_incomplete("【本行下部被窗口遮挡】")

    # 2) Bowel relation hierarchy.
    assert bowel_relation_from_text("病变侵及直肠似达黏膜层。") == ("present", 3)
    assert bowel_relation_from_text("病灶与直肠邻近、界限欠清晰。") == ("uncertain", 2)
    assert bowel_relation_from_text("病灶邻近肠管受压。") == ("uncertain", 1)

    # 3) Node negation is region-scoped even inside one sentence.
    ntxt = "肠系膜区及右侧肋膈角可见小淋巴结，较大者短径约6mm，腹主动脉旁及盆腔未见明显肿大淋巴结。"
    assert positive_node_clause(ntxt, "upper_abdominal_cardiophrenic_nodes")
    assert not positive_node_clause(ntxt, "pelvic_retroperitoneal_nodes")
    assert negative_node_clause(ntxt, "pelvic_retroperitoneal_nodes")

    # 4) Multi-site group max is retained globally but not copied to a site without explicit location.
    c1 = [{"id": "C001", "section": "findings", "text": "双侧结肠旁、肝周、脾旁、大网膜见多发结节影，大者长径约22mm。"}]
    assert compute_overall_peritoneal_max(c1)[0] == 22.0
    assert safe_size_candidates("greater_omentum", c1) == []
    assert "paracolic_wall_peritoneum" in positive_peritoneal_sites(c1[0]["text"])

    # 5) Explicit largest location allows site-specific size.
    c2 = [{"id": "C001", "section": "findings", "text": "右侧结肠旁沟、大网膜及盆底腹膜增厚见结节，较大者位于左下腹大网膜，长径约16mm。"}]
    assert safe_size_candidates("greater_omentum", c2)[0][0] == 16.0
    assert safe_size_candidates("paracolic_wall_peritoneum", c2) == []

    # 6) Primary rescue size uses largest dimension.
    c3 = [{"id": "C001", "section": "findings", "text": "子宫后方见类圆形囊实性肿块影，大小约71×94mm。"},
          {"id": "C002", "section": "impression", "text": "子宫后方占位，考虑来源于卵巢，恶性可能。"}]
    assert safe_size_candidates("primary_pelvic_mass", c3)[0][0] == 94.0

    # 7) Shared unqualified peritoneal change propagates; ambiguous partial multi-site change does not.
    c4 = [{"id": "C001", "section": "findings", "text": "盆底腹膜、大网膜增厚见结节，较前减小。"}]
    assert determine_site_change("pelvic_peritoneum", c4, "present")[0] == "decreased"
    c5 = [{"id": "C001", "section": "findings", "text": "盆底腹膜、大网膜增厚见结节，部分较前减小。"}]
    assert determine_site_change("pelvic_peritoneum", c5, "present")[0] == "not_stated"

    # 8) Fluid positive/negative language.
    f1 = [{"id": "C001", "section": "findings", "text": "盆腔见少量积液。"}]
    f2 = [{"id": "C001", "section": "findings", "text": "腹腔及盆腔未见积液。"}]
    assert deterministic_fluid(f1)["grade"] == 1
    assert deterministic_fluid(f2)["status"] == "absent"



def run_v61_unit_tests() -> None:
    # Header parser accepts both input styles.
    pp = parse_patient_pairs("1a\nPRE\n1b\nPOST\n【2a】\nPRE2\n【2b】\nPOST2\n")
    assert set(pp) == {"1", "2"} and set(pp["2"]) == {"a", "b"}

    # B: non-bowel object must not survive as bowel involvement.
    rec = {"patient_id":"x", "sites":[{
        "site":"bowel_stomach_involvement", "pre_status":"uncertain", "post_status":"not_mentioned",
        "pre_burden_grade":None,"post_burden_grade":None,"pre_max_size_mm":None,"post_max_size_mm":None,
        "pre_relation_grade":1,"post_relation_grade":None,"radiologist_reported_change":"not_stated",
        "pre_evidence_ids":["C1"],"post_evidence_ids":[],"pre_evidence":"病灶与子宫及膀胱密接。","post_evidence":""
    }]}
    fixed,audit = apply_v61_hotfix(rec)
    assert fixed["sites"][0]["pre_status"] == "not_mentioned" and audit

    # C: mesenteric node wording is not mesenteric-peritoneal disease.
    rec2 = {"patient_id":"x", "sites":[{
        "site":"small_bowel_mesenteric", "pre_status":"uncertain", "post_status":"not_mentioned",
        "pre_burden_grade":None,"post_burden_grade":None,"pre_max_size_mm":None,"post_max_size_mm":None,
        "pre_relation_grade":None,"post_relation_grade":None,"radiologist_reported_change":"not_stated",
        "pre_evidence_ids":["C1"],"post_evidence_ids":[],"pre_evidence":"小肠系膜根部见淋巴结，转移待除外。","post_evidence":""
    }]}
    fixed2,_ = apply_v61_hotfix(rec2)
    assert fixed2["sites"][0]["pre_status"] == "not_mentioned"


def run(args: argparse.Namespace) -> int:
    run_internal_unit_tests()
    run_v61_unit_tests()
    if OpenAI is None:
        raise RuntimeError("The 'openai' package is required. Install with: pip install openai")

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    text = input_path.read_text(encoding="utf-8-sig")
    pairs = parse_patient_pairs(text)
    selector = parse_patient_selector(args.patients)
    selected = [pid for pid in sort_patient_ids(pairs.keys()) if selector is None or pid in selector]
    if not selected:
        raise ValueError("No patient IDs selected.")

    missing = [pid for pid in selected if "a" not in pairs[pid] or "b" not in pairs[pid]]
    if missing:
        raise ValueError(f"Selected patient(s) missing a or b report: {missing}")

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else input_path.parent / "v6_1_output"
    outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = outdir / "all_patients_v6_1.jsonl"
    review_path = outdir / "review_v6_1.csv"
    manifest_path = outdir / "run_manifest_v6_1.json"
    trace_path = outdir / "model_trace_v6_1.jsonl"
    hotfix_audit_path = outdir / "hotfix_audit_v6_1.csv"

    if args.overwrite:
        for p in (jsonl_path, review_path, manifest_path, trace_path, hotfix_audit_path):
            if p.exists():
                p.unlink()
    elif jsonl_path.exists() and not args.resume:
        raise RuntimeError(f"{jsonl_path} already exists. Use --resume or --overwrite.")

    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("No API key found. Set DASHSCOPE_API_KEY or pass --api-key.")
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    existing = load_jsonl(jsonl_path) if args.resume else []
    done_ids = {str(p.get("patient_id")) for p in existing}
    run_errors: Dict[str, str] = {}
    duplicates = detect_duplicate_pairs(pairs, selected)
    started = now_utc()

    preliminary = build_manifest(
        input_path=input_path, model=args.model, base_url=args.base_url, selected=selected,
        completed=existing, run_errors=run_errors, started=started, finished=None,
        duplicates=duplicates, status="running",
    )
    manifest_path.write_text(json.dumps(preliminary, ensure_ascii=False, indent=2), encoding="utf-8")

    for idx, pid in enumerate(selected, 1):
        if pid in done_ids:
            print(f"[{idx}/{len(selected)}] patient {pid}: already completed, skipped (--resume)", flush=True)
            continue
        print(f"[{idx}/{len(selected)}] patient {pid}: PRE", flush=True)
        pre_text = pairs[pid]["a"]
        post_text = pairs[pid]["b"]
        pre_clauses = clauseize(pre_text)
        post_clauses = clauseize(post_text)
        try:
            pre_model, pre_raw = call_llm(client, args.model, pre_clauses, max_retries=args.max_retries)
            pre_pp, pre_warn = postprocess_report(pre_model, pre_clauses)
            print(f"[{idx}/{len(selected)}] patient {pid}: POST", flush=True)
            post_model, post_raw = call_llm(client, args.model, post_clauses, max_retries=args.max_retries)
            post_pp, post_warn = postprocess_report(post_model, post_clauses)
            patient = build_final_patient(
                pid, pre_pp, post_pp, pre_clauses, post_clauses,
                pre_text, post_text, pre_warn, post_warn,
            )
            base_validation = validate_patient(patient, pre_clauses, post_clauses)
            patient["_validation_v6_base"] = base_validation
            patient, hotfix_rows = apply_v61_hotfix(patient)
            patient["_validation_v6"] = validate_v61_final(patient, base_validation)
            append_jsonl(jsonl_path, patient)
            append_v61_hotfix_audit(hotfix_audit_path, hotfix_rows)
            existing.append(patient)
            done_ids.add(pid)

            if not args.no_trace:
                trace = {
                    "patient_id": pid,
                    "version": VERSION,
                    "prompt_sha256": PROMPT_SHA256,
                    "pre_clauses": pre_clauses,
                    "post_clauses": post_clauses,
                    "pre_model_normalized": pre_model,
                    "post_model_normalized": post_model,
                    "pre_raw_response": pre_raw,
                    "post_raw_response": post_raw,
                }
                append_jsonl(trace_path, trace)

            print(
                f"[{idx}/{len(selected)}] patient {pid}: "
                f"validation={'PASS' if patient['_validation_v6']['passed'] else 'FAIL'} "
                f"errors={len(patient['_validation_v6']['errors'])} "
                f"warnings={len(patient['_validation_v6']['warnings'])}",
                flush=True,
            )
        except Exception as e:
            run_errors[pid] = f"{type(e).__name__}: {e}"
            print(f"[{idx}/{len(selected)}] patient {pid}: ERROR {run_errors[pid]}", file=sys.stderr, flush=True)

        # Persist a running manifest after each patient so a long batch is auditable/resumable.
        current = build_manifest(
            input_path=input_path, model=args.model, base_url=args.base_url, selected=selected,
            completed=existing, run_errors=run_errors, started=started, finished=None,
            duplicates=duplicates, status="running",
        )
        manifest_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")

    completed = load_jsonl(jsonl_path)
    write_review_csv(completed, review_path)
    finished = now_utc()
    final_manifest = build_manifest(
        input_path=input_path, model=args.model, base_url=args.base_url, selected=selected,
        completed=completed, run_errors=run_errors, started=started, finished=finished,
        duplicates=duplicates, status="finished" if not run_errors else "finished_with_errors",
    )
    manifest_path.write_text(json.dumps(final_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"JSONL:    {jsonl_path}")
    print(f"Review:   {review_path}")
    print(f"Manifest: {manifest_path}")
    if not args.no_trace:
        print(f"Trace:    {trace_path}")
    print(f"Hotfix:   {hotfix_audit_path}")
    if duplicates:
        print(f"Duplicate report pairs detected: {duplicates}")
    if run_errors:
        print(f"Patients with run errors: {sorted(run_errors)}", file=sys.stderr)
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extraction V6.1 FINAL source-backed ovarian MRI structured extractor")
    p.add_argument("--input", required=True, help="UTF-8 text file containing patient blocks 1a/1b, 2a/2b, ...")
    p.add_argument("--outdir", default=None, help="Output directory; default: <input_dir>/v6_1_output")
    p.add_argument("--patients", default=None, help="Optional subset, e.g. 1-20 or 4,6,7,13,19")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", default=None, help="Prefer env DASHSCOPE_API_KEY; command-line key is supported but less safe")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--resume", action="store_true", help="Skip patient IDs already present in all_patients_v6_1.jsonl")
    p.add_argument("--overwrite", action="store_true", help="Delete prior V6 outputs in outdir before starting")
    p.add_argument("--no-trace", action="store_true", help="Do not save model_trace_v6_1.jsonl")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())