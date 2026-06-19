from __future__ import annotations

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


@dataclass
class TimePoint:
    date: str
    positive_count: int
    negative_count: int
    total_count: int
    positive_ratio: float
    negative_ratio: float
    negative_growth: float
    is_anomaly: bool
    linked_clusters: list[int] = field(default_factory=list)


@dataclass
class AnalysisResult:
    product_name: str
    total_reviews: int
    positive_count: int
    negative_count: int
    review_needed_count: int
    sentiment_distribution: dict
    clusters: list[ClusterInfo]
    time_trend: list[TimePoint]
    anomaly_points: list[TimePoint]
    positive_keywords: list[tuple[str, int]]
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

            clusters.append(
                ClusterInfo(
                    cluster_id=len(clusters),
                    size=len(cluster_reviews),
                    keywords=keywords,
                    representative_reviews=sorted_reviews[:3],
                )
            )

        clusters.sort(key=lambda c: c.size, reverse=True)
        for i, c in enumerate(clusters):
            c.cluster_id = i

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
        for col in ["正面", "负面", "中性"]:
            if col not in daily.columns:
                daily[col] = 0

        daily = daily.sort_values("_date")
        daily["total"] = daily["正面"] + daily["负面"] + daily["中性"]
        daily["positive_ratio"] = daily["正面"] / daily["total"].replace(0, 1)
        daily["negative_ratio"] = daily["负面"] / daily["total"].replace(0, 1)

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
                total_count=int(row["total"]),
                positive_ratio=float(row["positive_ratio"]),
                negative_ratio=float(row["negative_ratio"]),
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
        time_trend, anomalies = self.time_analyzer.analyze(work_df, clusters)
        pos_keywords = extract_positive_keywords(work_df)

        return AnalysisResult(
            product_name=product_name,
            total_reviews=total,
            positive_count=pos_count,
            negative_count=neg_count,
            review_needed_count=review_count,
            sentiment_distribution=sentiment_dist,
            clusters=clusters,
            time_trend=time_trend,
            anomaly_points=anomalies,
            positive_keywords=pos_keywords,
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
        time_trend, anomalies = self.time_analyzer.analyze(work_df, clusters)
        pos_keywords = extract_positive_keywords(work_df)

        return AnalysisResult(
            product_name=product_name,
            total_reviews=total,
            positive_count=pos_count,
            negative_count=neg_count,
            review_needed_count=review_count,
            sentiment_distribution=sentiment_dist,
            clusters=clusters,
            time_trend=time_trend,
            anomaly_points=anomalies,
            positive_keywords=pos_keywords,
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
