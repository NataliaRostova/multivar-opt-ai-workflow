"""End-to-end orchestrator for automated financial data cleaning, plotting, and LLM extraction."""

from __future__ import annotations

import json
from pathlib import Path
from python_workflow.data_cleaner import create_sample_messy_panel, clean_financial_panel, generate_workflow_chart
from python_workflow.llm_pipeline import run_structured_extraction

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


SAMPLE_FILINGS = [
    {
        "doc_id": "FILING-2026Q3-01",
        "entity_hint": "华泰证券（示例研报摘要）",
        "period_hint": "2026Q3",
        "catalyst_hint": "机构衍生品与财富管理手续费收入双增",
        "risk_hint": "权益市场波动对自营公允价值变动存在影响",
        "text": "2026年前三季度，公司实现营业收入约 298.6 亿元，同比增长 12.4%，归母净利润约 118.2 亿元。财富管理与场外衍生品做市业务保持高景气，但需关注二级市场波动带来的自营敞口波动风险。",
    },
    {
        "doc_id": "FILING-2026Q3-02",
        "entity_hint": "南京银行（示例财报快讯）",
        "period_hint": "2026Q3",
        "catalyst_hint": "长三角对公科创信贷扩张与低不良率韧性",
        "risk_hint": "LPR 调降周期下净息差（NIM）持续承压",
        "text": "2026年三季报显示，本行实现营业收入约 392.4 亿元，同比增长 8.9%，归母净利润约 169.5 亿元。长三角区域科创与制造业贷款投放稳健，拨备覆盖率保持高位；行业净息差仍面临温和收窄压力。",
    },
    {
        "doc_id": "FILING-2026Q3-03",
        "entity_hint": "某有色再生铝龙头（示例公告）",
        "period_hint": "2026Q3",
        "catalyst_hint": "新能源车一体化压铸铝合金订单放量",
        "risk_hint": "废铝原料采购价波动及期货套保基差风险",
        "text": "报告期内公司实现营业收入约 146.8 亿元，同比增长 5.2%，归母净利润约 9.4 亿元。铸造铝合金产线产能利用率提升至 88%，公司已通过上期所 AD 合约开展部分库存套期保值以平滑原料价格波动。",
    },
]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "sample_unstructured_filings.json").write_text(
        json.dumps(SAMPLE_FILINGS, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("1/3 Cleaning messy multi-sector financial panel...")
    raw_csv = create_sample_messy_panel()
    clean_df, audit = clean_financial_panel(raw_csv)

    print("2/3 Rendering automated diagnostic & optimization surface charts...")
    chart_path = generate_workflow_chart(clean_df)

    print("3/3 Running DeepSeek/Gemini structured information extraction workflow...")
    extracts = run_structured_extraction(SAMPLE_FILINGS)
    extract_path = OUTPUT_DIR / "llm_structured_extracts.json"
    extract_path.write_text(json.dumps(extracts, indent=2, ensure_ascii=False), encoding="utf-8")

    rows_md = "\n".join(
        f"| {r['entity']} | {r['period']} | {r['revenue_bn_rmb']} | {r['revenue_yoy_pct']}% | "
        f"{r['net_profit_bn_rmb']} | {r['sentiment']} | {r['key_catalyst']} | {r['risk_flag']} |"
        for r in extracts
    )
    brief_md = f"""# 自动化数据清洗与 LLM 结构化研报摘要

## 1. 数据清洗质检审计（Data Quality Audit）
- 原始记录行数：`{audit['rows_raw']}`
- 自动去重行数：`{audit['duplicates_removed']}`
- 行业中位数插补缺失值数：`{audit['missing_cells_imputed']}`
- `[2%, 98%]` 缩尾处理异常值数：`{audit['outliers_winsorized']}`
- 清洗后有效样本：`{audit['rows_clean']}`

## 2. LLM 非结构化公告抽取结果（JSON Schema → 表格）

| 主体 | 报告期 | 营收（亿元） | 营收同比 | 归母净利润（亿元） | 情绪判定 | 核心催化 | 风险提示 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
{rows_md}
"""
    brief_path = OUTPUT_DIR / "automated_briefing.md"
    brief_path.write_text(brief_md, encoding="utf-8")

    print("[DONE]", json.dumps({"audit": audit, "chart": str(chart_path), "extracts_count": len(extracts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
