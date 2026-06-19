"""
ReviewScan 自检脚本
使用方式: python -m self_check
覆盖：
  1. 单产品 analyze：情感分布 4 类相加 = 总数，待人工确认不重复统计
  2. watch 模式增量合并：追加新评论后，报告包含全量数据
  3. JSON 与 HTML 口径一致：同一批数据两个输出文件总数/情感数/异常点一致
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from reviewscan.analyzer import (
    AnalysisResult,
    ReviewAnalyzer,
    SentimentResult,
)
from reviewscan.reporter import HtmlReporter, export_json, result_to_dict


class StubSentiment:
    """Stub情感分类器：根据评分或关键词快速打标，不依赖真实模型"""

    def __init__(self):
        self._call_count = 0

    def batch_analyze(self, texts: list[str]) -> list[SentimentResult]:
        results = []
        self._call_count += len(texts)
        positive_words = ["好评", "赞", "满意", "不错", "很棒", "推荐", "给力", "正品", "喜欢", "五星"]
        negative_words = ["差评", "垃圾", "差", "坏", "退", "破损", "刮花", "假货", "失望", "二手", "粗糙", "慢"]
        ambiguous_words = ["一般", "还行", "中规中矩", "一般般", "中评", "说不好"]

        for t in texts:
            if not t or len(t.strip()) == 0:
                results.append(SentimentResult("中性", 0.0, True))
                continue

            pos_hits = sum(1 for w in positive_words if w in t)
            neg_hits = sum(1 for w in negative_words if w in t)
            amb_hits = sum(1 for w in ambiguous_words if w in t)

            if amb_hits > 0 and pos_hits == 0 and neg_hits == 0:
                results.append(SentimentResult("中性", 0.45, True))
            elif pos_hits > neg_hits:
                conf = 0.55 if (pos_hits == 1 and len(t) < 10) else min(0.7 + pos_hits * 0.05, 0.98)
                needs = conf < 0.6
                results.append(SentimentResult("正面", conf, needs))
            elif neg_hits > pos_hits:
                conf = 0.55 if (neg_hits == 1 and len(t) < 10) else min(0.7 + neg_hits * 0.05, 0.98)
                needs = conf < 0.6
                results.append(SentimentResult("负面", conf, needs))
            else:
                results.append(SentimentResult("中性", 0.4, True))

        return results


def _inject_stub(analyzer: ReviewAnalyzer) -> StubSentiment:
    stub = StubSentiment()
    analyzer.sentiment = stub
    return stub


SAMPLE_1 = """text,rating,date
物流很快，包装也很好，商品质量不错，好评！,5,2026-06-01
发货太慢了，等了一个星期才收到，差评,1,2026-06-01
质量很差，用了一天就坏了，要求退货退款,1,2026-06-02
还可以，跟描述的差不多，中规中矩,3,2026-06-02
包装有点破损，幸好里面东西没坏，一般般吧,3,2026-06-01
非常满意，推荐购买,5,2026-06-03
这个吧，说好不好说坏不坏，自己体会,3,2026-06-04
刚拆开就有划痕，明显是二手货！,1,2026-06-05
超出预期，价格便宜质量好，五星好评,5,2026-06-06
电池一天就没电了，续航真的很差劲,1,2026-06-06
做工粗糙，细节处理不到位，不值这个价,1,2026-06-07
暂时没发现什么问题，好评先,5,2026-06-07
"""

SAMPLE_APPEND = """和图片不符，色差很大，失望透顶,1,2026-06-08
太垃圾了，完全是假货，再也不买了,1,2026-06-08
外观精美，手感不错，用了几天都挺好,5,2026-06-08
用起来还行，就是电池续航不太给力,3,2026-06-09
买错型号了，退了，客服处理很快,5,2026-06-09
"""


def check_1_sentiment_distribution() -> tuple[bool, str]:
    """情感分布 4 类相加 = 总数，待人工确认不重复统计"""
    analyzer = ReviewAnalyzer()
    _inject_stub(analyzer)

    from io import StringIO
    df = pd.read_csv(StringIO(SAMPLE_1))
    result = analyzer.analyze(df, product_name="自检-情感分布")

    total = result.total_reviews
    dist = result.sentiment_distribution
    sum_four = dist["正面"] + dist["负面"] + dist["中性"] + dist["待人工确认"]

    msgs = []
    ok = True
    if total != sum_four:
        ok = False
        msgs.append(f"总数={total}，但四类相加={sum_four} (正面{dist['正面']}+负面{dist['负面']}+中性{dist['中性']}+待确认{dist['待人工确认']})")
    if result.positive_count != dist["正面"]:
        ok = False
        msgs.append(f"AnalysisResult.positive_count({result.positive_count}) != dist正面({dist['正面']})")
    if result.negative_count != dist["负面"]:
        ok = False
        msgs.append(f"AnalysisResult.negative_count({result.negative_count}) != dist负面({dist['负面']})")
    if result.review_needed_count != dist["待人工确认"]:
        ok = False
        msgs.append(f"AnalysisResult.review_needed_count({result.review_needed_count}) != dist待确认({dist['待人工确认']})")

    # 确认raw_df中带needs_review的行数也一致
    if result.raw_df is not None:
        nr_count = int(result.raw_df["needs_review"].sum())
        if nr_count != dist["待人工确认"]:
            ok = False
            msgs.append(f"raw_df needs_review数={nr_count} != dist待确认={dist['待人工确认']}")

    msg = f"总数={total} 分布={dist} → 四类相加={sum_four} {'✅一致' if ok else '❌不一致: ' + '; '.join(msgs)}"
    return ok, msg


def check_2_watch_merge() -> tuple[bool, str]:
    """watch 增量合并：追加新评论后，报告包含历史+新增全量"""
    from io import StringIO

    analyzer = ReviewAnalyzer()
    stub = _inject_stub(analyzer)

    df1 = pd.read_csv(StringIO(SAMPLE_1))
    df2 = pd.read_csv(StringIO(SAMPLE_APPEND), header=None)
    df2.columns = ["text", "rating", "date"]

    expected_first_count = len(df1)
    expected_total = expected_first_count + len(df2)

    result_first = analyzer.analyze_incremental(
        previous_labeled_df=None,
        new_df=df1,
        product_name="自检-阶段1",
    )

    stub_calls_after_first = stub._call_count

    result_merged = analyzer.analyze_incremental(
        previous_labeled_df=result_first.raw_df,
        new_df=df2,
        product_name="自检-阶段2合并",
    )

    stub_calls_after_second = stub._call_count
    incremental_model_calls = stub_calls_after_second - stub_calls_after_first

    msgs = []
    ok = True
    if result_first.total_reviews != expected_first_count:
        ok = False
        msgs.append(f"阶段1总数={result_first.total_reviews}，期望={expected_first_count}")
    if result_merged.total_reviews != expected_total:
        ok = False
        msgs.append(f"合并后总数={result_merged.total_reviews}，期望={expected_total}")
    if incremental_model_calls != len(df2):
        ok = False
        msgs.append(f"增量阶段模型调用={incremental_model_calls}条，期望只调用新增{len(df2)}条")
    if len(result_merged.clusters) < len(result_first.clusters):
        ok = False
        msgs.append(f"合并后差评簇反而更少：阶段1={len(result_first.clusters)}，合并后={len(result_merged.clusters)}")

    msg = (f"阶段1={result_first.total_reviews}条 + 新增{len(df2)}条 = 合并后={result_merged.total_reviews}条 | "
           f"模型调用：阶段1={stub_calls_after_first}次，增量阶段额外{incremental_model_calls}次 "
           f"{'✅合并正确' if ok else '❌错误: ' + '; '.join(msgs)}")
    return ok, msg


def check_3_json_html_consistency() -> tuple[bool, str]:
    """JSON 与 HTML 报告口径一致"""
    from io import StringIO

    analyzer = ReviewAnalyzer()
    _inject_stub(analyzer)

    df = pd.read_csv(StringIO(SAMPLE_1))
    result = analyzer.analyze(df, product_name="自检-JSONvsHTML")

    with tempfile.TemporaryDirectory() as td:
        json_path = os.path.join(td, "out.json")
        html_path = os.path.join(td, "out.html")

        export_json(result, json_path)
        HtmlReporter().render(result, html_path)

        with open(json_path, "r", encoding="utf-8") as f:
            jdata = json.load(f)

        with open(html_path, "r", encoding="utf-8") as f:
            html_text = f.read()

        # HTML 模板的数据源与 JSON 是同一个 result_to_dict，我们通过抽取嵌入的JS变量值来对比
        def _extract_json_var(html: str, varname: str) -> dict | list:
            m = re.search(rf"const\s+{varname}\s*=\s*(.+?);", html, re.DOTALL)
            if not m:
                raise ValueError(f"HTML 中找不到变量 {varname}")
            return json.loads(m.group(1))

        msgs = []
        ok = True

        # 1. 总数 (模板中: num在前, label在后)
        html_total_match = re.search(r"<div class=\"num\">(\d+)</div>\s*<div class=\"label\">总评论数", html_text)
        html_total = int(html_total_match.group(1)) if html_total_match else None
        if html_total != jdata["total_reviews"]:
            ok = False
            msgs.append(f"总数 HTML={html_total} vs JSON={jdata['total_reviews']}")

        # 2. 情感分布 (通过 sentimentPie 的 data JSON)
        html_sentiment = _extract_json_var(html_text, "sentimentData")
        if html_sentiment != jdata["sentiment_distribution"]:
            ok = False
            msgs.append(f"情感分布 HTML={html_sentiment} vs JSON={jdata['sentiment_distribution']}")

        # 3. 差评簇数
        html_clusters = _extract_json_var(html_text, "clusters")
        if len(html_clusters) != len(jdata["clusters"]):
            ok = False
            msgs.append(f"差评簇数量 HTML={len(html_clusters)} vs JSON={len(jdata['clusters'])}")

        # 4. 异常点数量
        html_anomalies = _extract_json_var(html_text, "anomalies")
        if len(html_anomalies) != len(jdata["anomaly_points"]):
            ok = False
            msgs.append(f"异常点数量 HTML={len(html_anomalies)} vs JSON={len(jdata['anomaly_points'])}")

        msg = (f"总数={jdata['total_reviews']} 情感分布={jdata['sentiment_distribution']} "
               f"簇数={len(jdata['clusters'])} 异常点={len(jdata['anomaly_points'])} "
               f"{'✅口径一致' if ok else '❌不一致: ' + '; '.join(msgs)}")
        return ok, msg


CHECKS = [
    ("1.情感分布统计(四类相加=总数)", check_1_sentiment_distribution),
    ("2.Watch增量合并(历史+新增全量)", check_2_watch_merge),
    ("3.JSON与HTML口径一致性", check_3_json_html_consistency),
]


def main():
    print("=" * 70)
    print("  ReviewScan 自检套件")
    print("=" * 70)

    passed = 0
    failed = 0
    for name, fn in CHECKS:
        try:
            ok, msg = fn()
            if ok:
                passed += 1
                print(f"\n[PASS] {name}\n       {msg}")
            else:
                failed += 1
                print(f"\n[FAIL] {name}\n       {msg}")
        except Exception as e:
            failed += 1
            print(f"\n[ERROR] {name}\n       异常: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"  结果: {passed} 项通过, {failed} 项失败")
    print("=" * 70)

    if failed:
        sys.exit(1)
    else:
        print("  🎉 所有自检通过！")
        sys.exit(0)


if __name__ == "__main__":
    main()
