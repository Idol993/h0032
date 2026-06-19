from __future__ import annotations

import datetime
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import jieba
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


STOPWORDS = set(
    "的 了 和 是 就 都 而 及 与 着 或 一个 没有 我们 你们 他们 它们 这 那 这个 那个 "
    "这些 那些 自己 什么 怎么 怎样 为什么 可以 但是 然后 因为 所以 如果 虽然 不过 "
    "已经 还是 还有 只是 不是 就是 真的 感觉 觉得 知道 看到 听到 买到 收到 东西 "
    "商品 产品 宝贝 非常 特别 比较 有点 一些 一下 一样 一直 其实 以后 以前 现在 "
    "时候 时间 今天 昨天 明天 这样 那样 怎么样 好不好 行不行 会不会 能不能 大家 "
    "本人 人家 别人 对方 买家 卖家 客服 老板 店家 商家 京东 淘宝 天猫 拼多多 "
    "我 你 他 她 它 在 有 也 不 被 把 让 给 对 从 到 向 跟 同 为 以 用 按 按照 "
    "很 太 更 最 还 又 再 只 仅 刚 才 却 竟 居然 果然 其实 反正 总之 简直 "
    "说 看 想 做 买 卖 用 试 问 找 查 等 等 等会儿 一下 吧 吗 呢 啊 哦 嗯 哈 呀 "
    .split()
)


@dataclass
class SentimentResult:
    label: str
    confidence: float
    needs_review: bool


@dataclass
class ClusterInfo:
    cluster_id: int
    size: int
    keywords: list[str]
    representative_reviews: list[dict]
    ratio: float = 0.0
    last_appeared_date: str = ""
    linked_anomaly_dates: list[str] = field(default_factory=list)
    priority: str = "低优先级"
    ticket_id: str = ""
    first_seen_date: str = ""
    status: str = "待处理"
    assignee: str = ""
    notes: str = ""
    is_recurring: bool = False


@dataclass
class TimePoint:
    date: str
    positive_count: int
    negative_count: int
    neutral_count: int
    review_needed_count: int
    total_count: int
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    review_needed_ratio: float
    negative_growth: float
    is_anomaly: bool
    linked_clusters: list[int] = field(default_factory=list)


@dataclass
class TicketStatusStats:
    status: str
    cluster_count: int
    review_count: int


@dataclass
class RiskOverview:
    negative_rate: float
    review_needed_rate: float
    recent_anomaly_count: int
    top_cluster_ratio: float
    top_cluster_keywords: list[str]
    ticket_status_summary: list[TicketStatusStats] = field(default_factory=list)
    recurring_resolved_clusters: list[dict] = field(default_factory=list)


@dataclass
class AnalysisResult:
    product_name: str
    total_reviews: int
    positive_count: int
    negative_count: int
    neutral_count: int
    review_needed_count: int
    sentiment_distribution: dict
    clusters: list[ClusterInfo]
    time_trend: list[TimePoint]
    anomaly_points: list[TimePoint]
    positive_keywords: list[tuple[str, int]]
    risk_overview: RiskOverview
    raw_df: Optional[pd.DataFrame] = None


class SentimentAnalyzer:
    def __init__(self, model_name: str = "uer/roberta-base-finetuned-jd-binary"):
        self.model_name = model_name
        self._pipeline = None
        self._model_loaded = False

    def _load_model(self):
        if self._model_loaded:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            max_length=512,
        )
        self._model_loaded = True

    def analyze(self, text: str) -> SentimentResult:
        self._load_model()
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            return SentimentResult(label="中性", confidence=0.0, needs_review=True)

        try:
            result = self._pipeline(text[:512])[0]
            label_map = {"positive": "正面", "negative": "负面", "LABEL_1": "正面", "LABEL_0": "负面"}
            label = label_map.get(result["label"], result["label"])
            confidence = float(result["score"])
            needs_review = confidence < 0.6
            return SentimentResult(label=label, confidence=confidence, needs_review=needs_review)
        except Exception:
            return SentimentResult(label="中性", confidence=0.0, needs_review=True)

    def batch_analyze(self, texts: list[str]) -> list[SentimentResult]:
        self._load_model()
        results = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = [t[:512] if t and isinstance(t, str) else "" for t in texts[i : i + batch_size]]
            try:
                outputs = self._pipeline(batch)
                label_map = {"positive": "正面", "negative": "负面", "LABEL_1": "正面", "LABEL_0": "负面"}
                for output in outputs:
                    label = label_map.get(output["label"], output["label"])
                    confidence = float(output["score"])
                    needs_review = confidence < 0.6
                    results.append(SentimentResult(label=label, confidence=confidence, needs_review=needs_review))
            except Exception:
                for _ in batch:
                    results.append(SentimentResult(label="中性", confidence=0.0, needs_review=True))
        return results


def tokenize_chinese(text: str) -> list[str]:
    if not text or not isinstance(text, str):
        return []
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]+", " ", text)
    words = jieba.lcut(text)
    return [w.strip() for w in words if w.strip() and w.strip() not in STOPWORDS and len(w.strip()) > 1]


class NegativeReviewClusterer:
    def __init__(self, min_k: int = 2, max_k: int = 8):
        self.min_k = min_k
        self.max_k = max_k

    def _find_optimal_k(self, X) -> int:
        n_samples = X.shape[0]
        if n_samples <= self.min_k:
            return max(1, n_samples // 3) if n_samples >= 3 else 1

        upper_k = min(self.max_k, n_samples - 1)
        if upper_k <= self.min_k:
            return upper_k if upper_k >= 1 else 1

        best_k = self.min_k
        best_score = -1.0
        for k in range(self.min_k, upper_k + 1):
            try:
                km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
                labels = km.fit_predict(X)
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(X, labels, metric="cosine")
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue
        return best_k if best_score > -0.5 else self.min_k

    def cluster(self, negative_reviews: list[dict]) -> list[ClusterInfo]:
        if not negative_reviews:
            return []

        texts = [r.get("text", "") for r in negative_reviews]
        tokenized = [" ".join(tokenize_chinese(t)) for t in texts]

        if not any(tokenized):
            return [
                ClusterInfo(
                    cluster_id=0,
                    size=len(negative_reviews),
                    keywords=["其他问题"],
                    representative_reviews=negative_reviews[:3],
                )
            ]

        vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b",
            max_features=5000,
            min_df=max(1, len(texts) // 50),
        )
        try:
            X = vectorizer.fit_transform(tokenized)
        except ValueError:
            return [
                ClusterInfo(
                    cluster_id=0,
                    size=len(negative_reviews),
                    keywords=["其他问题"],
                    representative_reviews=negative_reviews[:3],
                )
            ]

        optimal_k = self._find_optimal_k(X)
        km = KMeans(n_clusters=optimal_k, random_state=42, n_init=20, max_iter=500)
        labels = km.fit_predict(X)

        feature_names = vectorizer.get_feature_names_out()
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

        clusters: list[ClusterInfo] = []
        for cluster_idx in range(optimal_k):
            mask = labels == cluster_idx
            cluster_reviews = [r for r, m in zip(negative_reviews, mask) if m]
            if not cluster_reviews:
                continue

            keywords = []
            for ind in order_centroids[cluster_idx, :20]:
                word = feature_names[ind]
                if len(word) > 1:
                    keywords.append(word)
                if len(keywords) >= 10:
                    break

            if not keywords:
                keywords = ["其他问题"]

            sorted_reviews = sorted(
                cluster_reviews,
                key=lambda r: len(r.get("text", "")),
                reverse=True,
            )

            # 给每条评论回写 cluster_id（按当前 clusters 长度分配，后面会整体重排）
            temp_id = len(clusters)
            for r in cluster_reviews:
                r["cluster_id"] = temp_id

            clusters.append(
                ClusterInfo(
                    cluster_id=temp_id,
                    size=len(cluster_reviews),
                    keywords=keywords,
                    representative_reviews=sorted_reviews[:3],
                )
            )

        clusters.sort(key=lambda c: c.size, reverse=True)
        # 建立 old_id -> new_id 映射
        id_map = {c.cluster_id: i for i, c in enumerate(clusters)}
        for i, c in enumerate(clusters):
            c.cluster_id = i
        # 同步回写每条评论的新 cluster_id
        for r in negative_reviews:
            old_id = r.get("cluster_id")
            if old_id is not None and old_id in id_map:
                r["cluster_id"] = id_map[old_id]

        return clusters


class TimeSeriesAnalyzer:
    def __init__(self, anomaly_threshold: float = 0.5):
        self.anomaly_threshold = anomaly_threshold

    def analyze(self, df: pd.DataFrame, clusters: list[ClusterInfo]) -> tuple[list[TimePoint], list[TimePoint]]:
        required_cols = {"sentiment_label"}
        date_col = None
        for col in ["date", "time", "timestamp", "create_time", "评论时间", "时间"]:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            return [], []

        work_df = df.copy()
        work_df["_parsed_date"] = pd.to_datetime(work_df[date_col], errors="coerce")
        work_df = work_df.dropna(subset=["_parsed_date"])
        if work_df.empty:
            return [], []

        work_df["_date"] = work_df["_parsed_date"].dt.strftime("%Y-%m-%d")
        daily = (
            work_df.groupby("_date")["sentiment_label"]
            .value_counts()
            .unstack(fill_value=0)
            .reset_index()
        )
        for col in ["正面", "负面", "中性", "待人工确认"]:
            if col not in daily.columns:
                daily[col] = 0

        daily = daily.sort_values("_date")
        daily["total"] = daily["正面"] + daily["负面"] + daily["中性"] + daily["待人工确认"]
        daily["positive_ratio"] = daily["正面"] / daily["total"].replace(0, 1)
        daily["negative_ratio"] = daily["负面"] / daily["total"].replace(0, 1)
        daily["neutral_ratio"] = daily["中性"] / daily["total"].replace(0, 1)
        daily["review_needed_ratio"] = daily["待人工确认"] / daily["total"].replace(0, 1)

        negative_series = daily["负面"].astype(float).values
        growth_rates = [0.0]
        for i in range(1, len(negative_series)):
            prev = negative_series[i - 1]
            curr = negative_series[i]
            if prev > 0:
                growth_rates.append((curr - prev) / prev)
            elif curr > 0:
                growth_rates.append(1.0)
            else:
                growth_rates.append(0.0)

        time_points: list[TimePoint] = []
        anomaly_points: list[TimePoint] = []
        cluster_keyword_map: dict[int, set[str]] = {
            c.cluster_id: set(c.keywords) for c in clusters
        }

        for i, (_, row) in enumerate(daily.iterrows()):
            is_anomaly = growth_rates[i] > self.anomaly_threshold and row["负面"] >= 2
            date_str = row["_date"]
            linked = []
            if is_anomaly:
                day_reviews = work_df[work_df["_date"] == date_str]
                day_neg = day_reviews[day_reviews["sentiment_label"] == "负面"]
                day_words = set()
                for txt in day_neg.get("text", day_neg.iloc[:, 0] if len(day_neg.columns) > 0 else []).tolist():
                    if isinstance(txt, str):
                        day_words.update(tokenize_chinese(txt))
                for cid, kw_set in cluster_keyword_map.items():
                    if day_words & kw_set:
                        linked.append(cid)

            tp = TimePoint(
                date=date_str,
                positive_count=int(row["正面"]),
                negative_count=int(row["负面"]),
                neutral_count=int(row["中性"]),
                review_needed_count=int(row["待人工确认"]),
                total_count=int(row["total"]),
                positive_ratio=float(row["positive_ratio"]),
                negative_ratio=float(row["negative_ratio"]),
                neutral_ratio=float(row["neutral_ratio"]),
                review_needed_ratio=float(row["review_needed_ratio"]),
                negative_growth=float(growth_rates[i]),
                is_anomaly=is_anomaly,
                linked_clusters=linked,
            )
            time_points.append(tp)
            if is_anomaly:
                anomaly_points.append(tp)

        return time_points, anomaly_points


def extract_positive_keywords(df: pd.DataFrame, top_n: int = 50) -> list[tuple[str, int]]:
    text_col = None
    for col in ["text", "content", "评论", "评论内容", "评价", "review"]:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        text_col = df.columns[0]

    pos_df = df[df["sentiment_label"] == "正面"]
    word_counter: Counter = Counter()
    for txt in pos_df[text_col].dropna().tolist():
        if isinstance(txt, str):
            words = tokenize_chinese(txt)
            word_counter.update(words)

    return word_counter.most_common(top_n)


class ReviewAnalyzer:
    def __init__(self, model_name: str = "uer/roberta-base-finetuned-jd-binary"):
        self.sentiment = SentimentAnalyzer(model_name)
        self.clusterer = NegativeReviewClusterer()
        self.time_analyzer = TimeSeriesAnalyzer()

    @staticmethod
    def _detect_columns(df: pd.DataFrame) -> dict:
        cols = {}
        for col in ["text", "content", "评论", "评论内容", "评价", "review", "comment"]:
            if col in df.columns:
                cols["text"] = col
                break
        if "text" not in cols:
            for col in df.columns:
                if df[col].dtype == object and df[col].astype(str).str.len().mean() > 5:
                    cols["text"] = col
                    break

        for col in ["rating", "score", "评分", "星级", "star"]:
            if col in df.columns:
                cols["rating"] = col
                break

        for col in ["date", "time", "timestamp", "create_time", "评论时间", "时间"]:
            if col in df.columns:
                cols["date"] = col
                break

        return cols

    @staticmethod
    def _calc_risk_overview(
        total: int,
        neg_count: int,
        review_count: int,
        clusters: list[ClusterInfo],
        anomalies: list[TimePoint],
        recent_days: int = 7,
    ) -> RiskOverview:
        neg_rate = neg_count / total if total > 0 else 0.0
        rev_rate = review_count / total if total > 0 else 0.0

        recent_anom = 0
        if anomalies:
            sorted_anom = sorted(anomalies, key=lambda a: a.date, reverse=True)
            if len(sorted_anom) <= recent_days:
                recent_anom = len(sorted_anom)
            else:
                recent_anom = recent_days

        top_ratio = 0.0
        top_keywords: list[str] = []
        if clusters and neg_count > 0:
            top_cluster = clusters[0]
            top_ratio = top_cluster.size / neg_count
            top_keywords = top_cluster.keywords[:5]

        return RiskOverview(
            negative_rate=round(neg_rate, 4),
            review_needed_rate=round(rev_rate, 4),
            recent_anomaly_count=recent_anom,
            top_cluster_ratio=round(top_ratio, 4),
            top_cluster_keywords=top_keywords,
        )

    def analyze(
        self,
        df: pd.DataFrame,
        product_name: str = "未命名商品",
    ) -> AnalysisResult:
        col_map = self._detect_columns(df)
        if "text" not in col_map:
            raise ValueError("无法识别评论文本列，请确保CSV包含评论文本字段")

        work_df = df.copy()
        work_df["text"] = work_df[col_map["text"]].astype(str).fillna("")

        texts = work_df["text"].tolist()
        sentiments = self.sentiment.batch_analyze(texts)

        final_labels = []
        confidences = []
        for s in sentiments:
            confidences.append(s.confidence)
            if s.needs_review:
                final_labels.append("待人工确认")
            else:
                final_labels.append(s.label)

        work_df["sentiment_label"] = final_labels
        work_df["sentiment_confidence"] = confidences
        work_df["needs_review"] = work_df["sentiment_label"] == "待人工确认"

        total = len(work_df)
        pos_count = int((work_df["sentiment_label"] == "正面").sum())
        neg_count = int((work_df["sentiment_label"] == "负面").sum())
        neu_count = int((work_df["sentiment_label"] == "中性").sum())
        review_count = int((work_df["sentiment_label"] == "待人工确认").sum())

        sentiment_dist = {
            "正面": pos_count,
            "负面": neg_count,
            "中性": neu_count,
            "待人工确认": review_count,
        }

        neg_mask = work_df["sentiment_label"] == "负面"
        negative_reviews = []
        for idx, row in work_df[neg_mask].iterrows():
            review_dict = {"text": row["text"], "index": int(idx)}
            if "rating" in col_map:
                review_dict["rating"] = row.get(col_map["rating"])
            if "date" in col_map:
                review_dict["date"] = str(row.get(col_map["date"]))
            negative_reviews.append(review_dict)

        clusters = self.clusterer.cluster(negative_reviews)

        # 把 cluster_id 写回 work_df，便于后续取完整日期
        work_df["cluster_id"] = pd.NA
        for rev in negative_reviews:
            ridx = rev.get("index")
            cid = rev.get("cluster_id")
            if ridx is not None and cid is not None and ridx in work_df.index:
                work_df.loc[ridx, "cluster_id"] = cid

        time_trend, anomalies = self.time_analyzer.analyze(work_df, clusters)
        pos_keywords = extract_positive_keywords(work_df)
        risk = self._calc_risk_overview(total, neg_count, review_count, clusters, anomalies)

        return AnalysisResult(
            product_name=product_name,
            total_reviews=total,
            positive_count=pos_count,
            negative_count=neg_count,
            neutral_count=neu_count,
            review_needed_count=review_count,
            sentiment_distribution=sentiment_dist,
            clusters=clusters,
            time_trend=time_trend,
            anomaly_points=anomalies,
            positive_keywords=pos_keywords,
            risk_overview=risk,
            raw_df=work_df,
        )

    def _build_result_from_labeled_df(
        self,
        labeled_df: pd.DataFrame,
        col_map: dict,
        product_name: str,
    ) -> AnalysisResult:
        """基于已经带有 sentiment_label 列的 DataFrame 重新构建分析结果（不跑大模型）"""
        work_df = labeled_df.copy()
        if "text" not in work_df.columns:
            work_df["text"] = work_df[col_map["text"]].astype(str).fillna("")

        total = len(work_df)
        pos_count = int((work_df["sentiment_label"] == "正面").sum())
        neg_count = int((work_df["sentiment_label"] == "负面").sum())
        neu_count = int((work_df["sentiment_label"] == "中性").sum())
        review_count = int((work_df["sentiment_label"] == "待人工确认").sum())

        sentiment_dist = {
            "正面": pos_count,
            "负面": neg_count,
            "中性": neu_count,
            "待人工确认": review_count,
        }

        neg_mask = work_df["sentiment_label"] == "负面"
        negative_reviews = []
        for idx, row in work_df[neg_mask].iterrows():
            review_dict = {"text": row.get("text", ""), "index": int(idx)}
            if "rating" in col_map:
                review_dict["rating"] = row.get(col_map["rating"])
            if "date" in col_map:
                review_dict["date"] = str(row.get(col_map["date"]))
            negative_reviews.append(review_dict)

        clusters = self.clusterer.cluster(negative_reviews)

        # 把 cluster_id 写回 work_df
        work_df["cluster_id"] = pd.NA
        for rev in negative_reviews:
            ridx = rev.get("index")
            cid = rev.get("cluster_id")
            if ridx is not None and cid is not None and ridx in work_df.index:
                work_df.loc[ridx, "cluster_id"] = cid

        time_trend, anomalies = self.time_analyzer.analyze(work_df, clusters)
        pos_keywords = extract_positive_keywords(work_df)
        risk = self._calc_risk_overview(total, neg_count, review_count, clusters, anomalies)

        return AnalysisResult(
            product_name=product_name,
            total_reviews=total,
            positive_count=pos_count,
            negative_count=neg_count,
            neutral_count=neu_count,
            review_needed_count=review_count,
            sentiment_distribution=sentiment_dist,
            clusters=clusters,
            time_trend=time_trend,
            anomaly_points=anomalies,
            positive_keywords=pos_keywords,
            risk_overview=risk,
            raw_df=work_df,
        )

    def analyze_incremental(
        self,
        previous_labeled_df: pd.DataFrame | None,
        new_df: pd.DataFrame,
        product_name: str = "未命名商品",
    ) -> AnalysisResult:
        """
        增量分析：只对 new_df 中的新增评论跑模型，合并 previous_labeled_df 后输出完整报告。
        若 previous_labeled_df 为 None，则等价于 analyze 全量分析。
        """
        col_map = self._detect_columns(new_df)
        if "text" not in col_map:
            raise ValueError("无法识别评论文本列，请确保CSV包含评论文本字段")

        work_new = new_df.copy()
        work_new["text"] = work_new[col_map["text"]].astype(str).fillna("")

        texts = work_new["text"].tolist()
        sentiments = self.sentiment.batch_analyze(texts)

        final_labels = []
        confidences = []
        for s in sentiments:
            confidences.append(s.confidence)
            if s.needs_review:
                final_labels.append("待人工确认")
            else:
                final_labels.append(s.label)

        work_new["sentiment_label"] = final_labels
        work_new["sentiment_confidence"] = confidences
        work_new["needs_review"] = work_new["sentiment_label"] == "待人工确认"

        if previous_labeled_df is not None and len(previous_labeled_df) > 0:
            merged = pd.concat(
                [previous_labeled_df.reset_index(drop=True), work_new.reset_index(drop=True)],
                ignore_index=True,
            )
        else:
            merged = work_new.reset_index(drop=True)

        return self._build_result_from_labeled_df(merged, col_map, product_name)


def enrich_cluster_metrics(
    result: AnalysisResult,
    existing_ticket_df: Optional[pd.DataFrame] = None,
) -> AnalysisResult:
    """补全每个差评簇的工单字段：占比、最近出现日期、突增关联日期、建议优先级；匹配旧工单。

    Args:
        result: 分析结果
        existing_ticket_df: 旧工单CSV（可选），用于匹配旧工单编号和保留手填字段
    """
    if not result.clusters or not result.negative_count:
        # 无差评主题时，也要生成默认空工单字段
        return result

    anomalies = result.anomaly_points or []

    # 建一个 dict: cluster_id -> 关联的异常日期列表
    anomaly_map: dict[int, list[str]] = {}
    for a in anomalies:
        for cid in a.linked_clusters:
            anomaly_map.setdefault(cid, []).append(a.date)

    # 从 raw_df 中按 cluster_id 分组取最大日期（最近出现）和最小日期（首次发现）
    df = result.raw_df
    cluster_date_map: dict[int, dict[str, str]] = {}
    if df is not None and "cluster_id" in df.columns and "date" in df.columns:
        grouped = df[df["sentiment_label"] == "负面"].groupby("cluster_id")["date"].agg(
            last_date="max",
            first_date="min",
        )
        for cid, row in grouped.iterrows():
            if pd.notna(cid):
                cluster_date_map[int(cid)] = {
                    "last": str(row["last_date"]) if pd.notna(row["last_date"]) else "",
                    "first": str(row["first_date"]) if pd.notna(row["first_date"]) else "",
                }

    # 先匹配旧工单（如果提供）
    matched_ids = set()
    if existing_ticket_df is not None and len(existing_ticket_df) > 0:
        for cluster in result.clusters:
            matched = _match_single_cluster(cluster, existing_ticket_df, matched_ids)
            if matched is not None:
                _apply_matched_ticket(cluster, matched)
                matched_ids.add(matched["ticket_id"])

    # 为未匹配到的新主题分配新编号
    max_id_num = 0
    if existing_ticket_df is not None and "主题编号" in existing_ticket_df.columns:
        for tid in existing_ticket_df["主题编号"].dropna():
            s = str(tid)
            if s.startswith("T") and s[1:].isdigit():
                n = int(s[1:])
                if n > max_id_num:
                    max_id_num = n

    for cluster in result.clusters:
        if not cluster.ticket_id:
            max_id_num += 1
            cluster.ticket_id = f"T{max_id_num:03d}"

    # 再补全其他字段
    for cluster in result.clusters:
        # 占比
        cluster.ratio = round(cluster.size / result.negative_count, 4)

        # 最近出现日期（从完整df取最新，即使已有也更新，因为数据更全更准确）
        date_info = cluster_date_map.get(cluster.cluster_id, {})
        if date_info.get("last"):
            cluster.last_appeared_date = date_info["last"]
        elif not cluster.last_appeared_date:
            dates = [str(r.get("date", "")) for r in cluster.representative_reviews if r.get("date") and str(r.get("date")) != "nan"]
            if dates:
                cluster.last_appeared_date = max(dates)

        # 首次发现日期（优先保留已有值：旧工单手填的 > 完整df取的 > 代表评论估算）
        if not cluster.first_seen_date:
            if date_info.get("first"):
                cluster.first_seen_date = date_info["first"]
            else:
                dates = [str(r.get("date", "")) for r in cluster.representative_reviews if r.get("date") and str(r.get("date")) != "nan"]
                if dates:
                    cluster.first_seen_date = min(dates)

        # 突增关联（合并已有值 + 新检测到的，不去重）
        new_anomalies = list(anomaly_map.get(cluster.cluster_id, []))
        if cluster.linked_anomaly_dates:
            existing_set = set(cluster.linked_anomaly_dates)
            for d in new_anomalies:
                if d not in existing_set:
                    cluster.linked_anomaly_dates.append(d)
        else:
            cluster.linked_anomaly_dates = new_anomalies

        # 已解决主题最近又突增 → 标记复发
        cluster.is_recurring = (cluster.status == "已解决" and bool(cluster.linked_anomaly_dates))

        # 优先级打分（保留已有值：旧工单手填的 > 自动计算）
        if not cluster.priority:
            score = 0.0
            score += cluster.ratio * 100  # 占比权重高
            if cluster.linked_anomaly_dates:
                score += 40  # 关联突增
            if len(cluster.keywords) >= 3:
                score += 10  # 主题明确
            if score >= 60:
                cluster.priority = "高优先级"
            elif score >= 30:
                cluster.priority = "中优先级"
            else:
                cluster.priority = "低优先级"

    # 补充 RiskOverview 的工单状态统计和复发主题
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

    return result


def _jaccard_similarity(a: set, b: set) -> float:
    """计算两个集合的Jaccard相似度"""
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / len(union) if union else 0.0


def _match_single_cluster(
    cluster: ClusterInfo,
    existing_df: pd.DataFrame,
    used_ticket_ids: set[str],
    threshold: float = 0.3,
) -> Optional[dict]:
    """从旧工单里找最匹配的一条（关键词Jaccard + 评论文本Jaccard），未被占用的。"""
    best_score = 0.0
    best_row = None

    cluster_kw = set(cluster.keywords)
    cluster_text_tokens = set()
    for r in cluster.representative_reviews:
        t = r.get("text", "")
        cluster_text_tokens.update(tokenize_chinese(t))

    for _, row in existing_df.iterrows():
        tid = str(row.get("主题编号", ""))
        if tid in used_ticket_ids:
            continue

        # 关键词相似度
        ticket_kw = set()
        kw_str = str(row.get("主题关键词", ""))
        if kw_str:
            ticket_kw = set(w.strip() for w in kw_str.replace("、", ",").split(",") if w.strip())
        kw_sim = _jaccard_similarity(cluster_kw, ticket_kw)

        # 代表评论相似度
        ticket_text_tokens = set()
        rep_str = str(row.get("代表评论", ""))
        if rep_str:
            ticket_text_tokens.update(tokenize_chinese(rep_str))
        text_sim = _jaccard_similarity(cluster_text_tokens, ticket_text_tokens)

        # 综合得分（关键词权重更高）
        total_score = kw_sim * 0.6 + text_sim * 0.4

        if total_score >= threshold and total_score > best_score:
            best_score = total_score
            best_row = row.to_dict()

    if best_row is not None:
        best_row["_match_score"] = best_score
        best_row["ticket_id"] = str(best_row.get("主题编号", ""))
    return best_row


def _apply_matched_ticket(cluster: ClusterInfo, matched: dict) -> None:
    """把旧工单字段合并到簇上，保留手填字段。"""
    cluster.ticket_id = matched.get("ticket_id", cluster.ticket_id)
    # 手填字段（覆盖）
    for src, dst in [
        ("处理状态", "status"),
        ("处理人", "assignee"),
        ("备注", "notes"),
        ("首次发现日期", "first_seen_date"),
    ]:
        val = matched.get(src)
        if val is not None and str(val) != "nan" and str(val).strip():
            setattr(cluster, dst, str(val).strip())

    # 系统字段（保留自算的值）
    if not cluster.priority and matched.get("建议优先级"):
        cluster.priority = str(matched["建议优先级"])


def calc_ticket_status_stats(result: AnalysisResult) -> list[TicketStatusStats]:
    """按状态统计主题数和评论数"""
    from collections import defaultdict

    status_map: dict[str, TicketStatusStats] = {}
    default_order = ["待处理", "处理中", "已解决", "观察中"]

    for status in default_order:
        status_map[status] = TicketStatusStats(status=status, cluster_count=0, review_count=0)

    for cluster in result.clusters:
        s = cluster.status or "待处理"
        if s not in status_map:
            status_map[s] = TicketStatusStats(status=s, cluster_count=0, review_count=0)
        status_map[s].cluster_count += 1
        status_map[s].review_count += cluster.size

    return list(status_map.values())


def build_ticket_dataframe(
    result: AnalysisResult,
    existing_ticket_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """生成差评主题工单 DataFrame；无差评主题时也保留固定表头"""
    enrich_cluster_metrics(result, existing_ticket_df=existing_ticket_df)

    columns = [
        "主题编号", "评论数", "占差评比例", "占比数值", "主题关键词",
        "代表评论", "首次发现日期", "最近出现日期",
        "关联差评突增日期", "是否关联差评突增",
        "建议优先级", "处理状态", "处理人", "备注",
    ]

    rows = []
    for c in result.clusters:
        rep_texts = " | ".join(
            r.get("text", "").replace("\n", " ").replace("\r", " ")[:200]
            for r in c.representative_reviews
        )
        rows.append({
            "主题编号": c.ticket_id,
            "评论数": c.size,
            "占差评比例": f"{c.ratio * 100:.1f}%",
            "占比数值": c.ratio,
            "主题关键词": "、".join(c.keywords),
            "代表评论": rep_texts,
            "首次发现日期": c.first_seen_date,
            "最近出现日期": c.last_appeared_date,
            "关联差评突增日期": "、".join(c.linked_anomaly_dates),
            "是否关联差评突增": "是" if c.linked_anomaly_dates else "否",
            "建议优先级": c.priority,
            "处理状态": c.status,
            "处理人": c.assignee,
            "备注": c.notes,
        })

    df = pd.DataFrame(rows, columns=columns)
    # 确保列类型合理
    if len(df) == 0:
        # 空DataFrame也要有正确的列顺序和类型
        df = pd.DataFrame(columns=columns)
    return df


def filter_clusters(
    result: AnalysisResult,
    high_priority_only: bool = False,
    with_anomaly_only: bool = False,
    priorities: Optional[list[str]] = None,
    statuses: Optional[list[str]] = None,
    appeared_last_n_days: Optional[int] = None,
    anomaly_last_n_days: Optional[int] = None,
    reference_date: Optional[str] = None,
    skip_enrich: bool = False,
) -> AnalysisResult:
    """按风险筛选簇。筛选不影响总评论数、情感分布等全局指标，只影响 clusters.

    Args:
        priorities: 例如 ["高优先级", "中优先级"]
        statuses: 例如 ["待处理", "处理中"]
        appeared_last_n_days: 只保留最近 N 天内出现过的主题（按最近出现日期）
        anomaly_last_n_days: 只保留最近 N 天内有关联突增的主题
        reference_date: 日期计算基准，默认用 time_trend 最后一天或今天
        skip_enrich: 跳过 enrich_cluster_metrics 调用（字段已手动设置时用）
    """
    has_any_filter = any([
        high_priority_only, with_anomaly_only,
        priorities, statuses,
        appeared_last_n_days is not None,
        anomaly_last_n_days is not None,
    ])
    if not has_any_filter:
        return result

    if not skip_enrich:
        enrich_cluster_metrics(result)

    # 确定基准日期
    ref_dt: Optional[datetime.date] = None
    if reference_date:
        try:
            ref_dt = datetime.datetime.strptime(str(reference_date)[:10], "%Y-%m-%d").date()
        except ValueError:
            pass
    if ref_dt is None and result.time_trend:
        try:
            last_date = result.time_trend[-1].date
            ref_dt = datetime.datetime.strptime(str(last_date)[:10], "%Y-%m-%d").date()
        except ValueError:
            pass
    if ref_dt is None:
        ref_dt = datetime.date.today()

    def _parse_date(s: str) -> Optional[datetime.date]:
        try:
            return datetime.datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return None

    filtered = []
    for c in result.clusters:
        keep = True

        if high_priority_only and c.priority != "高优先级":
            keep = False
        if with_anomaly_only and not c.linked_anomaly_dates:
            keep = False
        if priorities and c.priority not in priorities:
            keep = False
        if statuses and c.status not in statuses:
            keep = False

        # 最近 N 天出现过
        if keep and appeared_last_n_days is not None:
            last_dt = _parse_date(c.last_appeared_date)
            if last_dt is None or (ref_dt - last_dt).days > appeared_last_n_days:
                keep = False

        # 最近 N 天有突增关联
        if keep and anomaly_last_n_days is not None:
            has_recent_anomaly = False
            for d in c.linked_anomaly_dates:
                adt = _parse_date(d)
                if adt and (ref_dt - adt).days <= anomaly_last_n_days:
                    has_recent_anomaly = True
                    break
            if not has_recent_anomaly:
                keep = False

        if keep:
            filtered.append(c)

    # 构造新的 AnalysisResult（保持其他字段不变）
    return AnalysisResult(
        product_name=result.product_name,
        total_reviews=result.total_reviews,
        positive_count=result.positive_count,
        negative_count=result.negative_count,
        neutral_count=result.neutral_count,
        review_needed_count=result.review_needed_count,
        sentiment_distribution=result.sentiment_distribution,
        clusters=filtered,
        time_trend=result.time_trend,
        anomaly_points=result.anomaly_points,
        positive_keywords=result.positive_keywords,
        risk_overview=result.risk_overview,
        raw_df=result.raw_df,
    )


def compare_products(results: list[AnalysisResult]) -> dict:
    all_keywords: list[set[str]] = []
    for r in results:
        kw = set()
        for c in r.clusters:
            kw.update(c.keywords)
        all_keywords.append(kw)

    if len(all_keywords) < 2:
        return {"intersection": [], "differences": {}}

    intersection = set.intersection(*all_keywords) if all_keywords else set()
    differences = {}
    for i, r in enumerate(results):
        others = set.union(*[all_keywords[j] for j in range(len(results)) if j != i]) if len(results) > 1 else set()
        unique = all_keywords[i] - others
        differences[r.product_name] = list(unique)

    return {
        "intersection": list(intersection),
        "differences": differences,
        "per_product": [
            {
                "product": r.product_name,
                "total": r.total_reviews,
                "negative": r.negative_count,
                "clusters": [
                    {
                        "size": c.size,
                        "keywords": c.keywords,
                    }
                    for c in r.clusters
                ],
            }
            for r in results
        ],
    }
