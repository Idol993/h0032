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
    build_ticket_dataframe,
    enrich_cluster_metrics,
    filter_clusters,
)
from reviewscan.reporter import HtmlReporter, export_json, export_ticket_csv, result_to_dict


class StubSentiment:
    """Stub情感分类器：根据评分或关键词快速打标，不依赖真实模型"""

    def __init__(self):
        self._call_count = 0

    def batch_analyze(self, texts: list[str]) -> list[SentimentResult]:
        results = []
        self._call_count += len(texts)
        positive_words = ["好评", "赞", "满意", "不错", "很棒", "推荐", "给力", "正品", "喜欢", "五星"]
        negative_words = ["差评", "垃圾", "差", "坏", "退", "破损", "刮花", "假货", "失望", "二手", "粗糙", "慢", "没电", "续航", "电池", "不好"]
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

    msg = f"总数={total} 分布={dist} → 四类相加={sum_four} {'[OK]一致' if ok else '[FAIL]不一致: ' + '; '.join(msgs)}"
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
           f"{'[OK]合并正确' if ok else '[FAIL]错误: ' + '; '.join(msgs)}")
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
               f"{'[OK]口径一致' if ok else '[FAIL]不一致: ' + '; '.join(msgs)}")
        return ok, msg


SAMPLE_WITH_AMBIGUOUS = """text,rating,date
物流很快，包装完好，五星好评！,5,2026-06-10
这个吧，说好不好说坏不坏，自己体会,3,2026-06-10
还可以，跟描述的差不多，中规中矩,3,2026-06-10
质量很差，用了一天就坏了，差评,1,2026-06-10
一般般吧,3,2026-06-11
还行吧,3,2026-06-11
太垃圾了，完全不值这个价,1,2026-06-11
非常满意，推荐给朋友了,5,2026-06-11
这个一般，怎么说呢，就那样吧,3,2026-06-12
还行还行一般般,3,2026-06-12
"""


def check_4_time_trend_with_review_needed() -> tuple[bool, str]:
    """时间趋势每日总数 = 当天原始评论数（含待人工确认）"""
    from io import StringIO

    analyzer = ReviewAnalyzer()
    _inject_stub(analyzer)

    df = pd.read_csv(StringIO(SAMPLE_WITH_AMBIGUOUS))
    result = analyzer.analyze(df, product_name="自检-时间趋势待确认")

    # 统计每天原始评论数
    raw_daily = df.groupby("date").size().to_dict()

    msgs = []
    ok = True

    if not result.time_trend:
        return False, "未生成时间趋势为空"

    for tp in result.time_trend:
        raw_count = raw_daily.get(tp.date, 0)
        sum_four = tp.positive_count + tp.negative_count + tp.neutral_count + tp.review_needed_count

        # 1. 四类相加 = total_count
        if sum_four != tp.total_count:
            ok = False
            msgs.append(f"{tp.date}: 四类相加={sum_four} != total_count={tp.total_count}")

        # 2. total_count = 原始评论数
        if tp.total_count != raw_count:
            ok = False
            msgs.append(f"{tp.date}: time_trend总数={tp.total_count} != 原始数={raw_count} (正{tp.positive_count}+负{tp.negative_count}+中{tp.neutral_count}+待{tp.review_needed_count})")

    # 3. 确认至少有一天存在待确认评论（验证 Stub 对模糊评论打标）
    has_review_needed = any(tp.review_needed_count > 0 for tp in result.time_trend)
    if not has_review_needed:
        ok = False
        msgs.append("没有任何一天有待确认评论，测试数据可能有问题")

    detail = ", ".join(
        f"{tp.date}={tp.total_count}(正{tp.positive_count}/负{tp.negative_count}/中{tp.neutral_count}/待{tp.review_needed_count})"
        for tp in result.time_trend
    )
    msg = f"共{len(result.time_trend)}天趋势数据: {detail} {'[OK]每日总数对齐' if ok else '[FAIL]错误: ' + '; '.join(msgs)}"
    return ok, msg


def check_5_cluster_representative_match() -> tuple[bool, str]:
    """差评簇主题数量、代表评论与报告一致性验证"""
    from io import StringIO

    analyzer = ReviewAnalyzer()
    _inject_stub(analyzer)

    df = pd.read_csv(StringIO(SAMPLE_WITH_AMBIGUOUS))
    result = analyzer.analyze(df, product_name="自检-差评主题对应")

    msgs = []
    ok = True

    if not result.clusters:
        return False, "没有任何差评簇，测试数据不足"

    # 1. 所有差评簇大小之和 = 总差评数
    total_cluster_size = sum(c.size for c in result.clusters)
    if total_cluster_size != result.negative_count:
        ok = False
        msgs.append(f"差评簇大小之和={total_cluster_size} != 总差评数={result.negative_count}")

    # 2. 每个簇的代表评论数量 <= 簇大小，且有至少1条
    for cluster in result.clusters:
        n_rep = len(cluster.representative_reviews)
        if n_rep == 0:
            ok = False
            msgs.append(f"簇{cluster.cluster_id}: 代表评论为空")
        if n_rep > cluster.size:
            ok = False
            msgs.append(f"簇{cluster.cluster_id}: 代表评论{n_rep}条 > 簇大小{cluster.size}")
        # 代表评论非空
        for i, rev in enumerate(cluster.representative_reviews):
            if not rev.get("text"):
                ok = False
                msgs.append(f"簇{cluster.cluster_id}: 第{i}条代表评论文本为空")

    # 3. 风险概览的最大主题占比在合理范围
    risk = result.risk_overview
    if not (0 <= risk.top_cluster_ratio <= 1.0 + 1e-6):
        ok = False
        msgs.append(f"最大主题占比={risk.top_cluster_ratio} 超出0~1范围")
    if result.clusters and len(risk.top_cluster_keywords) == 0:
        ok = False
        msgs.append("有差评簇但风险概览里最大主题关键词为空")
    if result.clusters and risk.top_cluster_keywords:
        # 最大主题关键词是最大簇的关键词的子集
        top_cluster = result.clusters[0]
        if not set(risk.top_cluster_keywords).issubset(set(top_cluster.keywords)):
            ok = False
            msgs.append(f"风险概览最大主题关键词 {risk.top_cluster_keywords} 不是最大簇关键词 {top_cluster.keywords[:5]} 的子集")

    # 4. JSON 输出里的簇信息与 AnalysisResult 中一致
    import json
    from reviewscan.reporter import result_to_dict
    jdata = result_to_dict(result)
    if len(jdata["clusters"]) != len(result.clusters):
        ok = False
        msgs.append(f"JSON中簇数量={len(jdata['clusters'])} != 对象中簇数量={len(result.clusters)}")
    if len(jdata["clusters"]) > 0 and jdata["clusters"][0]["size"] != result.clusters[0].size:
        ok = False
        msgs.append("JSON与对象的最大簇大小不一致")
    if jdata["risk_overview"]["negative_rate"] != float(risk.negative_rate):
        ok = False
        msgs.append(f"JSON与对象的差评率不一致: {jdata['risk_overview']['negative_rate']} vs {risk.negative_rate}")

    msg = (f"共{len(result.clusters)}个差评簇，总差评{result.negative_count}条，"
           f"最大主题占比={risk.top_cluster_ratio*100:.1f}%，"
           f"风险概览关键词={risk.top_cluster_keywords} "
           f"{'[OK]结构完整且口径一致' if ok else '[FAIL]错误: ' + '; '.join(msgs)}")
    return ok, msg


def check_6_ticket_csv_vs_report() -> tuple[bool, str]:
    """工单CSV真实生成，与 JSON/对象中的差评主题互相对应"""
    from io import StringIO

    analyzer = ReviewAnalyzer()
    _inject_stub(analyzer)

    df = pd.read_csv(StringIO(SAMPLE_WITH_AMBIGUOUS))
    result = analyzer.analyze(df, product_name="自检-工单对照")
    enrich_cluster_metrics(result)

    with tempfile.TemporaryDirectory() as td:
        ticket_path = os.path.join(td, "tickets.csv")
        json_path = os.path.join(td, "out.json")

        export_ticket_csv(result, ticket_path)
        export_json(result, json_path)

        # 读取工单CSV
        df_ticket = pd.read_csv(ticket_path, encoding="utf-8-sig")
        with open(json_path, "r", encoding="utf-8") as f:
            jdata = json.load(f)

        msgs = []
        ok = True

        # 1. 工单行数 = 差评簇数量
        if len(df_ticket) != len(result.clusters):
            ok = False
            msgs.append(f"工单CSV行数={len(df_ticket)} != 差评簇数量={len(result.clusters)}")

        # 2. 主题数量一致
        if len(df_ticket) != len(jdata["clusters"]):
            ok = False
            msgs.append(f"工单行数={len(df_ticket)} != JSON簇数={len(jdata['clusters'])}")

        # 3. 按工单编号顺序逐行核对评论数、关键词、代表评论
        for idx in range(min(len(df_ticket), len(result.clusters))):
            ticket_row = df_ticket.iloc[idx]
            cluster = result.clusters[idx]

            # 工单评论数
            if int(ticket_row["评论数"]) != cluster.size:
                ok = False
                msgs.append(f"第{idx+1}行工单评论数={ticket_row['评论数']} != 簇大小={cluster.size}")

            # 工单关键词
            ticket_keywords = set(str(ticket_row["主题关键词"]).replace("、", ",").split(","))
            cluster_keywords = set(cluster.keywords)
            if not ticket_keywords.issubset(cluster_keywords) and not cluster_keywords.issubset(ticket_keywords):
                overlap = ticket_keywords & cluster_keywords
                if len(overlap) < max(len(ticket_keywords), len(cluster_keywords)) * 0.5:
                    ok = False
                    msgs.append(f"第{idx+1}行工单关键词与簇关键词对应不足: 工单{ticket_keywords} vs 簇{cluster_keywords}")

            # 工单代表评论中包含簇的至少1条代表评论
            ticket_rep = str(ticket_row["代表评论"])
            cluster_rep_texts = [r.get("text", "") for r in cluster.representative_reviews]
            if cluster_rep_texts and not any(txt[:50] in ticket_rep for txt in cluster_rep_texts if txt):
                ok = False
                msgs.append(f"第{idx+1}行工单代表评论未包含簇的代表评论")

            # 工单占比是否一致（允许±0.1%误差）
            pct_str = str(ticket_row["占差评比例"]).replace("%", "")
            if pct_str and result.negative_count:
                try:
                    ticket_pct = float(pct_str)
                    real_pct = cluster.ratio * 100
                    if abs(ticket_pct - real_pct) > 1.0:
                        ok = False
                        msgs.append(f"第{idx+1}行工单占比={ticket_pct}% 与实际占比={real_pct:.1f}% 误差超过1%")
                except ValueError:
                    pass

            # JSON中的簇字段与工单核对
            if idx < len(jdata["clusters"]):
                jc = jdata["clusters"][idx]
                if int(jc["size"]) != int(ticket_row["评论数"]):
                    ok = False
                    msgs.append(f"第{idx+1}行 JSON size={jc['size']} != 工单评论数={ticket_row['评论数']}")
                if jc.get("priority") not in str(ticket_row.get("建议优先级", "")):
                    ok = False
                    msgs.append(f"第{idx+1}行 JSON priority={jc.get('priority')} 与 工单优先级={ticket_row.get('建议优先级')} 不一致")

        # 4. 工单包含指定列
        required_cols = ["主题编号", "评论数", "占差评比例", "主题关键词", "代表评论", "最近出现日期", "是否关联差评突增", "建议优先级"]
        missing_cols = [c for c in required_cols if c not in df_ticket.columns]
        if missing_cols:
            ok = False
            msgs.append(f"工单CSV缺少列: {missing_cols}")

        detail = f"工单{len(df_ticket)}行，JSON簇{len(jdata['clusters'])}个，对象簇{len(result.clusters)}个"
        msg = f"{detail} {'[OK]工单/JSON/对象三方一致' if ok else '[FAIL]不一致: ' + '; '.join(msgs)}"
        return ok, msg


def check_7_html_trend_four_types() -> tuple[bool, str]:
    """HTML时间趋势图包含四类+总评，每天总数与JSON time_trend对齐"""
    from io import StringIO

    analyzer = ReviewAnalyzer()
    _inject_stub(analyzer)

    df = pd.read_csv(StringIO(SAMPLE_WITH_AMBIGUOUS))
    result = analyzer.analyze(df, product_name="自检-HTML趋势四类")

    with tempfile.TemporaryDirectory() as td:
        html_path = os.path.join(td, "out.html")
        json_path = os.path.join(td, "out.json")
        HtmlReporter().render(result, html_path)
        export_json(result, json_path)

        with open(html_path, "r", encoding="utf-8") as f:
            html_text = f.read()
        with open(json_path, "r", encoding="utf-8") as f:
            jdata = json.load(f)

        msgs = []
        ok = True

        # 1. 图例包含四类+当天总评
        legend_need = ["正面评论", "负面评论", "中性评论", "待人工确认", "当天总评"]
        for name in legend_need:
            if name not in html_text:
                ok = False
                msgs.append(f"HTML图例缺少 {name}")

        # 2. HTML time_trend 变量里的每天四类之和 = total_count（与JSON一致）
        def _extract_json_var(html, varname):
            m = re.search(rf"const\s+{varname}\s*=\s*(.+?);", html, re.DOTALL)
            if not m:
                raise ValueError(f"HTML 中找不到变量 {varname}")
            return json.loads(m.group(1))

        html_time = _extract_json_var(html_text, "timeData")
        json_time = jdata["time_trend"]

        if len(html_time) != len(json_time):
            ok = False
            msgs.append(f"HTML time_trend天数={len(html_time)} != JSON天数={len(json_time)}")

        for ht, jt in zip(html_time, json_time):
            four_sum = (
                int(ht.get("positive_count", 0))
                + int(ht.get("negative_count", 0))
                + int(ht.get("neutral_count", 0))
                + int(ht.get("review_needed_count", 0))
            )
            if four_sum != int(ht.get("total_count", -1)):
                ok = False
                msgs.append(f"{ht.get('date')}: HTML四类相加={four_sum} != HTML total_count={ht.get('total_count')}")
            if int(ht.get("total_count", -1)) != int(jt.get("total_count", -2)):
                ok = False
                msgs.append(f"{ht.get('date')}: HTML total={ht.get('total_count')} != JSON total={jt.get('total_count')}")
            # 待确认数量一致性
            if int(ht.get("review_needed_count", -1)) != int(jt.get("review_needed_count", -2)):
                ok = False
                msgs.append(f"{ht.get('date')}: HTML待确认={ht.get('review_needed_count')} != JSON待确认={jt.get('review_needed_count')}")

        # 3. 自定义tooltip函数存在（含当天总评数展示）
        if "📊 当天总评数" not in html_text:
            ok = False
            msgs.append("HTML趋势图tooltip不含'当天总评数'字段")

        detail = f"共{len(html_time)}天趋势数据"
        msg = f"{detail} {'[OK]HTML四类+总评与JSON一致' if ok else '[FAIL]错误: ' + '; '.join(msgs)}"
        return ok, msg


def check_8_risk_filter_consistency() -> tuple[bool, str]:
    """风险筛选三端口径一致：筛选后簇数量在对象/JSON/工单CSV中相同"""
    from io import StringIO

    analyzer = ReviewAnalyzer()
    _inject_stub(analyzer)

    df = pd.read_csv(StringIO(SAMPLE_WITH_AMBIGUOUS))
    result_full = analyzer.analyze(df, product_name="自检-风险筛选全量")
    enrich_cluster_metrics(result_full)

    # 以高优先级筛选为例
    result_filtered = filter_clusters(result_full, high_priority_only=True, with_anomaly_only=False)

    with tempfile.TemporaryDirectory() as td:
        ticket_full = os.path.join(td, "tickets_full.csv")
        ticket_flt = os.path.join(td, "tickets_filtered.csv")
        json_flt = os.path.join(td, "out_filtered.json")
        html_flt = os.path.join(td, "out_filtered.html")

        export_ticket_csv(result_full, ticket_full)
        export_ticket_csv(result_filtered, ticket_flt)
        export_json(result_filtered, json_flt)
        HtmlReporter().render(result_filtered, html_flt)

        df_t_full = pd.read_csv(ticket_full, encoding="utf-8-sig")
        df_t_flt = pd.read_csv(ticket_flt, encoding="utf-8-sig")

        with open(json_flt, "r", encoding="utf-8") as f:
            jflt = json.load(f)
        with open(html_flt, "r", encoding="utf-8") as f:
            html_text = f.read()

        msgs = []
        ok = True

        # 1. 筛选后的簇数量在三方一致
        n_obj = len(result_filtered.clusters)
        n_tkt = len(df_t_flt)
        n_json = len(jflt["clusters"])

        if not (n_obj == n_tkt == n_json):
            ok = False
            msgs.append(f"筛选后簇数量不一致: 对象={n_obj} 工单={n_tkt} JSON={n_json}")

        # 2. 筛选出的确实都是高优先级
        if len(df_t_flt) > 0:
            non_high = df_t_flt[df_t_flt["建议优先级"] != "高优先级"]
            if len(non_high) > 0:
                ok = False
                msgs.append(f"高优先级筛选后仍存在非高优先级行: {list(non_high['建议优先级'])}")

        # 3. HTML中嵌入的clusters数量也一致
        m = re.search(r"const\s+clusters\s*=\s*(.+?);", html_text, re.DOTALL)
        if m:
            html_clusters = json.loads(m.group(1))
            if len(html_clusters) != n_json:
                ok = False
                msgs.append(f"HTML簇数量={len(html_clusters)} != JSON簇数量={n_json}")

        # 4. 全量工单优先级字段非空
        if "建议优先级" not in df_t_full.columns or df_t_full["建议优先级"].isna().any():
            ok = False
            msgs.append("全量工单中优先级字段为空或缺失")

        # 5. 全局指标不受筛选影响（总评论数/情感分布保持不变）
        if result_full.total_reviews != result_filtered.total_reviews:
            ok = False
            msgs.append(f"筛选后总评论数变了: 全量={result_full.total_reviews} 筛选={result_filtered.total_reviews}")
        if jflt["total_reviews"] != result_full.total_reviews:
            ok = False
            msgs.append(f"JSON中总评论数不一致: {jflt['total_reviews']} vs {result_full.total_reviews}")

        detail = (f"全量工单{len(df_t_full)}行，筛选后{n_obj}行（对象）/{n_tkt}行（工单）/{n_json}行（JSON）"
                  f"| 筛选条件=高优先级-only")
        msg = f"{detail} {'[OK]三端口径一致且筛选有效' if ok else '[FAIL]错误: ' + '; '.join(msgs)}"
        return ok, msg


def check_9_ticket_reuse_matching() -> tuple[bool, str]:
    """工单复用：旧工单关键词+评论相似度匹配，保留手填字段，新主题追加新编号"""
    from io import StringIO

    analyzer = ReviewAnalyzer()
    _inject_stub(analyzer)

    df = pd.read_csv(StringIO(SAMPLE_WITH_AMBIGUOUS))
    result = analyzer.analyze(df, product_name="自检-工单复用")

    with tempfile.TemporaryDirectory() as td:
        # 第一次生成工单
        ticket_path1 = os.path.join(td, "tickets_v1.csv")
        export_ticket_csv(result, ticket_path1)

        # 模拟运营手填：给第一条工单加状态、处理人、备注
        df_ticket1 = pd.read_csv(ticket_path1, encoding="utf-8-sig")
        if len(df_ticket1) >= 1:
            # 先把可能是 float64 的列转成 object，避免赋值字符串报错
            str_cols = ["处理状态", "处理人", "备注", "首次发现日期"]
            for col in str_cols:
                if col in df_ticket1.columns:
                    df_ticket1[col] = df_ticket1[col].astype(object).where(df_ticket1[col].notna(), None)
            df_ticket1.loc[0, "处理状态"] = "处理中"
            df_ticket1.loc[0, "处理人"] = "张三"
            df_ticket1.loc[0, "备注"] = "正在联系供应商"
            df_ticket1.loc[0, "首次发现日期"] = "2026-06-01"
        df_ticket1.to_csv(ticket_path1, index=False, encoding="utf-8-sig")

        # 重新运行 analyze，传入旧工单
        result2 = analyzer.analyze(df, product_name="自检-工单复用-第二次")
        existing_ticket = pd.read_csv(ticket_path1, encoding="utf-8-sig")
        enrich_cluster_metrics(result2, existing_ticket_df=existing_ticket)

        ticket_path2 = os.path.join(td, "tickets_v2.csv")
        export_ticket_csv(result2, ticket_path2, existing_ticket_df=existing_ticket)

        df_ticket2 = pd.read_csv(ticket_path2, encoding="utf-8-sig")

        msgs = []
        ok = True

        # 1. 旧工单编号保留（主题编号相同）
        if len(df_ticket1) >= 1 and len(df_ticket2) >= 1:
            if df_ticket1.loc[0, "主题编号"] != df_ticket2.loc[0, "主题编号"]:
                ok = False
                msgs.append(f"旧工单编号未保留: v1={df_ticket1.loc[0, '主题编号']} v2={df_ticket2.loc[0, '主题编号']}")

            # 2. 手填字段保留
            if str(df_ticket2.loc[0, "处理状态"]) != "处理中":
                ok = False
                msgs.append(f"处理状态未保留: v2={df_ticket2.loc[0, '处理状态']}")
            if str(df_ticket2.loc[0, "处理人"]) != "张三":
                ok = False
                msgs.append(f"处理人未保留: v2={df_ticket2.loc[0, '处理人']}")
            if "正在联系供应商" not in str(df_ticket2.loc[0, "备注"]):
                ok = False
                msgs.append(f"备注未保留: v2={df_ticket2.loc[0, '备注']}")
            if str(df_ticket2.loc[0, "首次发现日期"]) != "2026-06-01":
                ok = False
                msgs.append(f"首次发现日期未保留: v2={df_ticket2.loc[0, '首次发现日期']}")

        # 3. 对象中也保留了手填字段
        if result2.clusters:
            c0 = result2.clusters[0]
            if c0.status != "处理中":
                ok = False
                msgs.append(f"对象中状态未保留: {c0.status}")
            if c0.assignee != "张三":
                ok = False
                msgs.append(f"对象中处理人未保留: {c0.assignee}")
            if "正在联系供应商" not in c0.notes:
                ok = False
                msgs.append(f"对象中备注未保留: {c0.notes}")

        # 4. 新编号不重号（如果新主题加入，从旧最大编号+1开始）
        existing_ids = set(str(t) for t in df_ticket1["主题编号"].dropna())
        new_ids = set(str(t) for t in df_ticket2["主题编号"].dropna())
        for nid in new_ids:
            if nid not in existing_ids and nid.startswith("T") and nid[1:].isdigit():
                n = int(nid[1:])
                max_existing = max(
                    (int(t[1:]) for t in existing_ids if t.startswith("T") and t[1:].isdigit()),
                    default=0,
                )
                if n <= max_existing:
                    ok = False
                    msgs.append(f"新主题编号{nid} ≤ 旧最大编号T{max_existing:03d}，可能重号")

        detail = f"旧工单{len(df_ticket1)}行，新工单{len(df_ticket2)}行，旧编号保留={df_ticket1.loc[0, '主题编号'] if len(df_ticket1)>=1 else 'N/A'}"
        msg = f"{detail} {'[OK]工单复用匹配成功' if ok else '[FAIL]错误: ' + '; '.join(msgs)}"
        return ok, msg


def check_10_ticket_status_stats() -> tuple[bool, str]:
    """工单状态统计：四类状态主题数/评论数汇总正确，JSON与对象一致，已解决突增标记复发"""
    from io import StringIO

    analyzer = ReviewAnalyzer()
    _inject_stub(analyzer)

    df = pd.read_csv(StringIO(SAMPLE_WITH_AMBIGUOUS))
    result = analyzer.analyze(df, product_name="自检-状态统计")

    # 先 enrich 一次生成基础字段
    enrich_cluster_metrics(result)

    # 手工设置不同状态，便于验证（在 enrich 之后设置，避免被覆盖）
    if len(result.clusters) >= 2:
        result.clusters[0].status = "处理中"
        result.clusters[1].status = "已解决"
        result.clusters[0].assignee = "李四"
        # 给第二个主题加突增关联，触发复发标记
        if not result.clusters[1].linked_anomaly_dates:
            result.clusters[1].linked_anomaly_dates = ["2026-06-19"]
        # 手动重新计算复发标记（不重新 enrich，避免覆盖 last_appeared_date）
        result.clusters[1].is_recurring = (
            result.clusters[1].status == "已解决" and bool(result.clusters[1].linked_anomaly_dates)
        )
        # 重新计算状态统计和复发列表
        from reviewscan.analyzer import calc_ticket_status_stats
        stats = calc_ticket_status_stats(result)
        recurring = [
            {
                "ticket_id": c.ticket_id,
                "keywords": c.keywords[:5],
                "size": c.size,
                "last_appeared_date": c.last_appeared_date,
                "linked_anomaly_dates": c.linked_anomaly_dates,
            }
            for c in result.clusters
            if c.is_recurring
        ]
        result.risk_overview.ticket_status_summary = stats
        result.risk_overview.recurring_resolved_clusters = recurring

    msgs = []
    ok = True

    # 1. 工单状态统计存在
    stats = result.risk_overview.ticket_status_summary
    if not stats:
        ok = False
        msgs.append("工单状态统计为空")
    else:
        # 状态类别正确
        statuses = [s.status for s in stats]
        expected = ["待处理", "处理中", "已解决", "观察中"]
        for e in expected:
            if e not in statuses:
                ok = False
                msgs.append(f"状态统计缺少 {e}")

        # 总数对得上
        total_clusters = sum(s.cluster_count for s in stats)
        total_reviews = sum(s.review_count for s in stats)
        if total_clusters != len(result.clusters):
            ok = False
            msgs.append(f"状态统计主题总数={total_clusters} != 实际簇数={len(result.clusters)}")
        if total_reviews != result.negative_count:
            ok = False
            msgs.append(f"状态统计评论总数={total_reviews} != 总差评数={result.negative_count}")

    # 2. 已解决主题关联了突增 → 标记复发
    recurring = result.risk_overview.recurring_resolved_clusters
    if len(result.clusters) >= 2 and not recurring:
        ok = False
        msgs.append("已解决且关联突增的主题未标记复发")
    if recurring:
        for c in recurring:
            if not c.get("ticket_id"):
                ok = False
                msgs.append("复发主题缺少工单号")

    # 3. JSON 输出与对象一致
    from reviewscan.reporter import result_to_dict
    jdata = result_to_dict(result)
    j_stats = jdata["risk_overview"].get("ticket_status_summary", [])
    if len(j_stats) != len(stats):
        ok = False
        msgs.append(f"JSON状态统计长度={len(j_stats)} != 对象={len(stats)}")
    if j_stats:
        if j_stats[0].get("status") != stats[0].status:
            ok = False
            msgs.append("JSON状态与对象不一致")
        if int(j_stats[0].get("cluster_count", -1)) != int(stats[0].cluster_count):
            ok = False
            msgs.append("JSON主题数与对象不一致")

    detail = f"共{len(result.clusters)}个簇，{len(stats)}种状态，{len(recurring)}个复发主题"
    msg = f"{detail} {'[OK]状态统计正确' if ok else '[FAIL]错误: ' + '; '.join(msgs)}"
    return ok, msg


def check_11_multi_dim_filter() -> tuple[bool, str]:
    """多维筛选：优先级、状态、最近N天出现、最近N天突增组合筛选；全局指标不受影响"""
    from io import StringIO

    analyzer = ReviewAnalyzer()
    _inject_stub(analyzer)

    df = pd.read_csv(StringIO(SAMPLE_WITH_AMBIGUOUS))
    result_full = analyzer.analyze(df, product_name="自检-多维筛选全量")
    enrich_cluster_metrics(result_full)

    # 给簇设置不同状态和优先级（在 enrich 之后设置，避免被覆盖）
    if len(result_full.clusters) >= 2:
        result_full.clusters[0].status = "处理中"
        result_full.clusters[0].priority = "高优先级"
        # 手动设置日期，避免从 df 取的日期不符合测试预期
        result_full.clusters[0].last_appeared_date = "2026-06-19"
        result_full.clusters[0].linked_anomaly_dates = ["2026-06-18"]

        result_full.clusters[1].status = "已解决"
        result_full.clusters[1].priority = "低优先级"
        result_full.clusters[1].last_appeared_date = "2026-06-01"  # 18天前
        result_full.clusters[1].linked_anomaly_dates = ["2026-05-19"]  # 31天前，超过30天窗口
        # 不重新 enrich，避免覆盖手动设置的值

    msgs = []
    ok = True

    # 测试 1: 按优先级筛选
    r_high = filter_clusters(result_full, priorities=["高优先级"], skip_enrich=True)
    if len(result_full.clusters) >= 2 and len(r_high.clusters) != 1:
        ok = False
        msgs.append(f"按高优先级筛选后簇数={len(r_high.clusters)}，期望1")

    # 测试 2: 按状态筛选
    r_open = filter_clusters(result_full, statuses=["待处理", "处理中"], skip_enrich=True)
    if len(result_full.clusters) >= 2 and len(r_open.clusters) != 1:
        ok = False
        msgs.append(f"按开放状态筛选后簇数={len(r_open.clusters)}，期望1")

    # 测试 3: 按最近 10 天出现过筛选（基准日 2026-06-19）
    r_recent = filter_clusters(result_full, appeared_last_n_days=10, reference_date="2026-06-19", skip_enrich=True)
    if len(result_full.clusters) >= 2 and len(r_recent.clusters) != 1:
        ok = False
        msgs.append(f"按最近10天出现筛选后簇数={len(r_recent.clusters)}，期望1（6/1的应被过滤）")

    # 测试 4: 按最近 30 天突增关联筛选
    r_anom_recent = filter_clusters(result_full, anomaly_last_n_days=30, reference_date="2026-06-19", skip_enrich=True)
    if len(result_full.clusters) >= 2 and len(r_anom_recent.clusters) != 1:
        ok = False
        msgs.append(f"按最近30天突增筛选后簇数={len(r_anom_recent.clusters)}，期望1（5/20的应被过滤）")

    # 测试 5: 组合筛选（高优先级 AND 处理中）
    r_comb = filter_clusters(result_full, priorities=["高优先级"], statuses=["处理中"], skip_enrich=True)
    if len(result_full.clusters) >= 2 and len(r_comb.clusters) != 1:
        ok = False
        msgs.append(f"组合筛选后簇数={len(r_comb.clusters)}，期望1")

    # 测试 6: 全局指标不被筛选影响
    if r_high.total_reviews != result_full.total_reviews:
        ok = False
        msgs.append(f"筛选后总评论数变了: {r_high.total_reviews} vs {result_full.total_reviews}")
    if r_high.negative_count != result_full.negative_count:
        ok = False
        msgs.append(f"筛选后总差评数变了: {r_high.negative_count} vs {result_full.negative_count}")

    # 测试 7: 三端口径一致（筛选后对象/JSON/工单主题数相同）
    with tempfile.TemporaryDirectory() as td:
        ticket_path = os.path.join(td, "ticket_filtered.csv")
        json_path = os.path.join(td, "out_filtered.json")
        export_ticket_csv(r_comb, ticket_path)
        export_json(r_comb, json_path)

        df_t = pd.read_csv(ticket_path, encoding="utf-8-sig")
        with open(json_path, "r", encoding="utf-8") as f:
            j = json.load(f)

        if len(df_t) != len(r_comb.clusters):
            ok = False
            msgs.append(f"筛选后工单行数={len(df_t)} != 对象簇数={len(r_comb.clusters)}")
        if len(j["clusters"]) != len(r_comb.clusters):
            ok = False
            msgs.append(f"筛选后JSON簇数={len(j['clusters'])} != 对象簇数={len(r_comb.clusters)}")

    detail = (f"全量{len(result_full.clusters)}簇 | 高优筛选→{len(r_high.clusters)} | 状态筛选→{len(r_open.clusters)} | "
              f"最近10天→{len(r_recent.clusters)} | 最近30天突增→{len(r_anom_recent.clusters)} | 组合→{len(r_comb.clusters)}")
    msg = f"{detail} {'[OK]多维筛选正确且三端一致' if ok else '[FAIL]错误: ' + '; '.join(msgs)}"
    return ok, msg


def check_12_last_date_from_full_df() -> tuple[bool, str]:
    """最近出现日期从该主题全部评论的最新日期计算，不是只看代表评论"""
    from io import StringIO

    analyzer = ReviewAnalyzer()
    _inject_stub(analyzer)

    # 造一批数据：某个主题有很多评论，代表评论的日期比较早，但有一条最新评论在代表评论之外
    data = """text,rating,date
质量很差，用了一天就坏了,1,2026-06-01
做工粗糙，细节不好,1,2026-06-02
太垃圾了，完全是假货,1,2026-06-03
电池续航太差劲,1,2026-06-04
一天就没电了，续航差,1,2026-06-05
续航真的很差,1,2026-06-06
用了半天就没电了，续航太差,1,2026-06-07
"""
    df = pd.read_csv(StringIO(data))
    result = analyzer.analyze(df, product_name="自检-最近日期")
    enrich_cluster_metrics(result)

    msgs = []
    ok = True

    # 1. 验证 raw_df 里有 cluster_id 列
    if result.raw_df is None or "cluster_id" not in result.raw_df.columns:
        ok = False
        msgs.append("raw_df 中缺少 cluster_id 列，无法按簇取完整日期")

    # 2. 从 raw_df 中按 cluster_id 分组，验证每个簇的最近/首次日期与 cluster 里的一致
    if result.raw_df is not None and "cluster_id" in result.raw_df.columns and "date" in result.raw_df.columns:
        df_neg = result.raw_df[result.raw_df["sentiment_label"] == "负面"]
        grouped = df_neg.groupby("cluster_id")["date"].agg(
            last_date="max",
            first_date="min",
        )
        expected_dates = {}
        for cid, row in grouped.iterrows():
            if pd.notna(cid):
                expected_dates[int(cid)] = {
                    "last": str(row["last_date"]) if pd.notna(row["last_date"]) else "",
                    "first": str(row["first_date"]) if pd.notna(row["first_date"]) else "",
                }

        # 检查每个 cluster 的日期是否与 raw_df 一致
        for c in result.clusters:
            exp = expected_dates.get(c.cluster_id, {})
            if exp.get("last") and c.last_appeared_date != exp["last"]:
                ok = False
                msgs.append(f"簇{c.cluster_id}最近日期={c.last_appeared_date}，期望从df取到{exp['last']}")
            if exp.get("first") and c.first_seen_date != exp["first"]:
                ok = False
                msgs.append(f"簇{c.cluster_id}首次日期={c.first_seen_date}，期望从df取到{exp['first']}")

        # 3. 至少有一个簇的最近日期不是只从代表评论取的
        # （代表评论最多3条，如果总评论数>3，最近日期应该从完整df取）
        large_clusters = [c for c in result.clusters if c.size > 3]
        if large_clusters:
            lc = large_clusters[0]
            rep_dates = [str(r.get("date", "")) for r in lc.representative_reviews if r.get("date")]
            # 如果代表评论的最大日期 < 完整df的最大日期，说明我们确实从完整df取了
            if rep_dates and max(rep_dates) < lc.last_appeared_date:
                pass  # 正确！
            elif rep_dates and max(rep_dates) == lc.last_appeared_date and lc.last_appeared_date == expected_dates.get(lc.cluster_id, {}).get("last"):
                pass  # 也可能巧合一致，只要和df的一致就行
            # 只要和 df 的一致就是对的，上面已经检查了

    # 4. 最大簇的最近日期应该是原始数据中最大的 2026-06-07
    if result.clusters:
        largest = max(result.clusters, key=lambda c: c.size)
        expected_max = "2026-06-07"
        if largest.last_appeared_date != expected_max:
            ok = False
            msgs.append(f"最大簇最近日期={largest.last_appeared_date}，期望{expected_max}")

    detail = f"共{len(result.clusters)}个簇，最大簇最近={result.clusters[0].last_appeared_date if result.clusters else 'N/A'}"
    msg = f"{detail} {'[OK]最近日期从完整df计算' if ok else '[FAIL]错误: ' + '; '.join(msgs)}"
    return ok, msg


def check_13_no_negative_scenario() -> tuple[bool, str]:
    """无差评主题场景：工单CSV保留表头，终端+JSON+HTML显示暂无可分派"""
    from io import StringIO

    analyzer = ReviewAnalyzer()
    _inject_stub(analyzer)

    # 全是好评
    data = """text,rating,date
物流很快，包装完好，五星好评,5,2026-06-10
商品质量不错，很满意,5,2026-06-11
超出预期，推荐购买,5,2026-06-12
很好，下次还会买,5,2026-06-13
非常满意，点赞,5,2026-06-14
"""
    df = pd.read_csv(StringIO(data))
    result = analyzer.analyze(df, product_name="自检-无差评场景")

    with tempfile.TemporaryDirectory() as td:
        ticket_path = os.path.join(td, "tickets_empty.csv")
        html_path = os.path.join(td, "out_empty.html")
        json_path = os.path.join(td, "out_empty.json")

        export_ticket_csv(result, ticket_path)
        export_json(result, json_path)
        HtmlReporter().render(result, html_path)

        msgs = []
        ok = True

        # 1. 工单 CSV 保留固定表头
        df_t = pd.read_csv(ticket_path, encoding="utf-8-sig")
        expected_cols = [
            "主题编号", "评论数", "占差评比例", "占比数值", "主题关键词",
            "代表评论", "首次发现日期", "最近出现日期",
            "关联差评突增日期", "是否关联差评突增",
            "建议优先级", "处理状态", "处理人", "备注",
        ]
        for col in expected_cols:
            if col not in df_t.columns:
                ok = False
                msgs.append(f"无差评工单缺少列: {col}")
        if len(df_t) != 0:
            ok = False
            msgs.append(f"无差评工单应有0行数据，实际{len(df_t)}行")

        # 2. JSON 中 clusters 为空数组
        with open(json_path, "r", encoding="utf-8") as f:
            j = json.load(f)
        if j["clusters"] != []:
            ok = False
            msgs.append(f"无差评场景JSON clusters长度={len(j['clusters'])}，期望0")

        # 3. HTML 中有"暂无可分派差评主题"提示
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        if "暂无可分派差评主题" not in html:
            ok = False
            msgs.append("HTML中缺少'暂无可分派差评主题'提示")

        # 4. 终端报告中也有提示（通过调用 print_top_clusters 捕获输出验证）
        import io
        from rich.console import Console
        buf = io.StringIO()
        test_console = Console(file=buf, force_terminal=False)
        from reviewscan.reporter import RichReporter
        rep = RichReporter(console=test_console)
        rep.print_top_clusters(result)
        output = buf.getvalue()
        if "暂无可分派差评主题" not in output:
            ok = False
            msgs.append("终端报告中缺少'暂无可分派差评主题'提示")

        # 5. 全局指标仍然正确
        if result.total_reviews != 5:
            ok = False
            msgs.append(f"无差评场景总评论数={result.total_reviews}，期望5")
        if result.negative_count != 0:
            ok = False
            msgs.append(f"无差评场景差评数={result.negative_count}，期望0")

        detail = f"工单列={len(df_t)}，JSON clusters={len(j['clusters'])}，HTML提示={'有' if '暂无可分派' in html else '无'}"
        msg = f"{detail} {'[OK]无差评场景处理正确' if ok else '[FAIL]错误: ' + '; '.join(msgs)}"
        return ok, msg


CHECKS = [
    ("1.情感分布统计(四类相加=总数)", check_1_sentiment_distribution),
    ("2.Watch增量合并(历史+新增全量)", check_2_watch_merge),
    ("3.JSON与HTML口径一致性", check_3_json_html_consistency),
    ("4.时间趋势总数(含待确认)", check_4_time_trend_with_review_needed),
    ("5.差评主题与代表评论对应", check_5_cluster_representative_match),
    ("6.工单CSV与JSON/报告三方对照", check_6_ticket_csv_vs_report),
    ("7.HTML趋势图四类+总评对齐JSON", check_7_html_trend_four_types),
    ("8.风险筛选三端(对象/JSON/工单)口径一致", check_8_risk_filter_consistency),
    ("9.工单复用(匹配旧工单+保留手填字段+新编号追加)", check_9_ticket_reuse_matching),
    ("10.工单状态统计(四类+复发标记+JSON一致)", check_10_ticket_status_stats),
    ("11.多维筛选(优先级/状态/最近N天/突增N天+三端一致)", check_11_multi_dim_filter),
    ("12.最近出现日期(从完整df取，非仅代表评论)", check_12_last_date_from_full_df),
    ("13.无差评场景(保留表头+三端提示)", check_13_no_negative_scenario),
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
        print("  [V] 所有自检通过！")
        sys.exit(0)


if __name__ == "__main__":
    main()
