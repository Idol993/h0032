from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .analyzer import AnalysisResult


def _to_python(obj):
    """递归把 numpy / pandas 类型转成 Python 原生类型，便于 JSON 序列化"""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_to_python(v) for v in obj.tolist()]
    return obj


def result_to_dict(result: AnalysisResult) -> dict:
    data = {
        "product_name": result.product_name,
        "total_reviews": int(result.total_reviews),
        "positive_count": int(result.positive_count),
        "negative_count": int(result.negative_count),
        "review_needed_count": int(result.review_needed_count),
        "sentiment_distribution": {k: int(v) for k, v in result.sentiment_distribution.items()},
        "clusters": [
            {
                "cluster_id": int(c.cluster_id),
                "size": int(c.size),
                "keywords": list(c.keywords),
                "representative_reviews": [_to_python(r) for r in c.representative_reviews],
            }
            for c in result.clusters
        ],
        "time_trend": [_to_python(asdict(t)) for t in result.time_trend],
        "anomaly_points": [_to_python(asdict(t)) for t in result.anomaly_points],
        "positive_keywords": [
            {"word": w, "count": int(cnt)} for w, cnt in result.positive_keywords
        ],
    }
    return _to_python(data)


def export_json(result: AnalysisResult, output_path: str) -> None:
    data = result_to_dict(result)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def export_compare_json(compare_data: dict, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_to_python(compare_data), f, ensure_ascii=False, indent=2)


class RichReporter:
    def __init__(self):
        self.console = Console()

    def print_summary(self, result: AnalysisResult) -> None:
        dist = result.sentiment_distribution
        total = result.total_reviews
        pct = lambda x: (x / total * 100) if total > 0 else 0

        title = f"📊 {result.product_name} - 评论分析报告"
        self.console.print(
            Panel.fit(
                f"[bold]总评论数:[/bold] {total}  |  "
                f"[green]正面:[/green] {dist.get('正面', 0)} ({pct(dist.get('正面', 0)):.1f}%)  |  "
                f"[red]负面:[/red] {dist.get('负面', 0)} ({pct(dist.get('负面', 0)):.1f}%)  |  "
                f"[yellow]待确认:[/yellow] {dist.get('待人工确认', 0)} ({pct(dist.get('待人工确认', 0)):.1f}%)",
                title=title,
                border_style="cyan",
            )
        )

    def print_sentiment_pie(self, result: AnalysisResult) -> None:
        dist = result.sentiment_distribution
        total = result.total_reviews
        if total == 0:
            return

        colors = {
            "正面": "green",
            "负面": "red",
            "中性": "dim",
            "待人工确认": "yellow",
        }

        table = Table(title="🥧 情感分布", show_header=True, header_style="bold magenta")
        table.add_column("类别", style="bold")
        table.add_column("数量", justify="center", style="cyan")
        table.add_column("占比", justify="center")
        table.add_column("分布", width=40)

        for label in ["正面", "负面", "中性", "待人工确认"]:
            count = dist.get(label, 0)
            pct = count / total * 100 if total > 0 else 0
            filled = int(pct / 5)
            bar = "█" * filled + "░" * (20 - filled)
            table.add_row(
                Text(label, style=colors.get(label, "white")),
                str(count),
                f"{pct:.1f}%",
                Text(bar, style=colors.get(label, "white")),
            )

        self.console.print(table)

    def print_top_clusters(self, result: AnalysisResult, top_n: int = 5) -> None:
        if not result.clusters:
            self.console.print("[yellow]暂无差评聚类结果[/yellow]")
            return

        table = Table(
            title=f"🔥 Top {min(top_n, len(result.clusters))} 差评主题",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("排名", style="dim", width=6, justify="center")
        table.add_column("评论数", style="cyan", width=8, justify="center")
        table.add_column("主题关键词", style="white")

        for i, cluster in enumerate(result.clusters[:top_n]):
            keywords_str = "、".join(cluster.keywords[:6])
            table.add_row(str(i + 1), str(cluster.size), keywords_str)

        self.console.print(table)

        self.console.print("\n[bold underline]📝 代表性评论:[/bold underline]\n")
        for i, cluster in enumerate(result.clusters[:top_n]):
            theme = "、".join(cluster.keywords[:5])
            self.console.print(
                f"[bold red]主题 #{i + 1}[/bold red] [dim]({cluster.size}条)[/dim] "
                f"[cyan]→ {theme}[/cyan]"
            )
            for j, review in enumerate(cluster.representative_reviews[:2]):
                text = review.get("text", "")
                if len(text) > 120:
                    text = text[:120] + "..."
                self.console.print(f"  [{j + 1}] {text}")
            self.console.print()

    def print_positive_keywords(self, result: AnalysisResult, top_n: int = 20) -> None:
        if not result.positive_keywords:
            return

        kws = result.positive_keywords[:top_n]
        self.console.print(f"\n[bold green]✨ 好评高频关键词 Top{len(kws)}:[/bold green]")
        words_line = []
        for w, cnt in kws:
            intensity = min(int(cnt / max(kws[0][1], 1) * 4), 4)
            styles = ["", "green", "bold green", "bold bright_green"]
            words_line.append(f"[{styles[intensity]}]{w}({cnt})[/]")
        self.console.print("  " + "  ".join(words_line))

    def print_anomalies(self, result: AnalysisResult) -> None:
        if not result.anomaly_points:
            self.console.print("\n[green]✅ 未检测到差评突增异常[/green]")
            return

        self.console.print(f"\n[bold red]⚠️  检测到 {len(result.anomaly_points)} 个差评突增时间点:[/bold red]")
        table = Table(show_header=True, header_style="bold red")
        table.add_column("日期", style="cyan")
        table.add_column("差评数", style="red", justify="center")
        table.add_column("环比增长", style="yellow", justify="center")
        table.add_column("关联问题簇", style="magenta")

        for tp in result.anomaly_points:
            growth_pct = f"{tp.negative_growth * 100:.0f}%"
            linked = (
                "、".join([f"#{cid}" for cid in tp.linked_clusters])
                if tp.linked_clusters
                else "无"
            )
            table.add_row(tp.date, str(tp.negative_count), growth_pct, linked)

        self.console.print(table)

    def print_time_trend_table(self, result: AnalysisResult, max_rows: int = 10) -> None:
        if not result.time_trend:
            return

        self.console.print(f"\n[bold]📈 时间趋势 (最近{min(max_rows, len(result.time_trend))}天):[/bold]")
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("日期", style="cyan")
        table.add_column("总评", justify="center")
        table.add_column("正面", style="green", justify="center")
        table.add_column("负面", style="red", justify="center")
        table.add_column("差评占比", justify="center")
        table.add_column("环比", justify="center")

        recent = result.time_trend[-max_rows:]
        for tp in recent:
            neg_pct = f"{tp.negative_ratio * 100:.1f}%"
            growth = f"{tp.negative_growth * 100:+.0f}%" if tp.negative_growth != 0 else "0%"
            style = "bold red" if tp.is_anomaly else ""
            row_style = style if style else None
            table.add_row(
                tp.date,
                str(tp.total_count),
                str(tp.positive_count),
                str(tp.negative_count),
                neg_pct,
                Text(growth, style=style),
            )

        self.console.print(table)

    def print_analysis(self, result: AnalysisResult) -> None:
        self.print_summary(result)
        self.print_sentiment_pie(result)
        self.print_top_clusters(result)
        self.print_positive_keywords(result)
        self.print_anomalies(result)
        self.print_time_trend_table(result)

    def print_compare(self, compare_data: dict) -> None:
        self.console.print(Panel.fit("[bold]🔍 多产品差评对比分析[/bold]", border_style="magenta"))

        if compare_data.get("intersection"):
            self.console.print(
                f"\n[bold cyan]🔗 共有问题关键词 ({len(compare_data['intersection'])}个):[/bold cyan]\n  "
                + "、".join(compare_data["intersection"][:15])
            )

        if compare_data.get("differences"):
            self.console.print("\n[bold yellow]🌟 各产品独有问题:[/bold yellow]")
            for product, keywords in compare_data["differences"].items():
                kws = keywords[:10] if keywords else ["无明显独有问题"]
                self.console.print(f"  [bold]{product}:[/bold] " + "、".join(kws))

        if compare_data.get("per_product"):
            self.console.print("\n[bold]📋 各产品差评簇详情:[/bold]")
            for info in compare_data["per_product"]:
                self.console.print(
                    f"\n  [bold]{info['product']}[/bold] "
                    f"[dim](总{info['total']}条, 差评{info['negative']}条)[/dim]"
                )
                for j, c in enumerate(info["clusters"][:3]):
                    theme = "、".join(c["keywords"][:5])
                    self.console.print(f"    #{j + 1} [{c['size']}条] → {theme}")


class HtmlReporter:
    def __init__(self):
        self.template_dir = os.path.join(os.path.dirname(__file__), "templates")

    def _load_template(self, name: str = "report.html"):
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader(self.template_dir))
        return env.get_template(name)

    def render(self, result: AnalysisResult, output_path: str) -> None:
        data = result_to_dict(result)
        template = self._load_template()
        html = template.render(
            result=data,
            clusters_json=json.dumps(data["clusters"], ensure_ascii=False),
            time_trend_json=json.dumps(data["time_trend"], ensure_ascii=False),
            keywords_json=json.dumps(data["positive_keywords"], ensure_ascii=False),
            anomalies_json=json.dumps(data["anomaly_points"], ensure_ascii=False),
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    def render_compare(self, compare_data: dict, output_path: str) -> None:
        template = self._load_template("compare_report.html")
        html = template.render(
            compare_data=compare_data,
            compare_json=json.dumps(compare_data, ensure_ascii=False),
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)


def print_to_console(result: AnalysisResult) -> None:
    reporter = RichReporter()
    reporter.print_analysis(result)


def print_compare_to_console(compare_data: dict) -> None:
    reporter = RichReporter()
    reporter.print_compare(compare_data)
