from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from . import __version__
from .analyzer import ReviewAnalyzer, compare_products, filter_clusters
from .reporter import (
    HtmlReporter,
    export_compare_json,
    export_json,
    export_ticket_csv,
    print_compare_to_console,
    print_to_console,
)

console = Console()


def _read_csv_safe(path: str) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    raise click.ClickException(f"读取CSV失败 {path}: {last_err}")


def _detect_product_name(filepath: str) -> str:
    name = Path(filepath).stem
    return name.replace("_", " ").replace("-", " ")


def _check_output_format(output: str | None) -> str | None:
    if not output:
        return None
    ext = Path(output).suffix.lower()
    if ext in (".html", ".htm"):
        return "html"
    elif ext == ".json":
        return "json"
    elif ext in (".csv", ".xlsx"):
        return ext[1:]
    return None


@click.group()
@click.version_option(version=__version__, prog_name="reviewscan")
def app():
    """商品评论情感分析与差评聚类命令行工具"""
    pass


@app.command()
@click.argument("csv_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", type=click.Path(), help="输出报告路径 (.html 或 .json)")
@click.option("--format", "-f", "fmt", type=click.Choice(["html", "json", "csv"]), help="输出格式（默认按扩展名推断）")
@click.option("--product-name", "-n", type=str, help="商品名称（默认使用文件名）")
@click.option("--model", type=str, default="uer/roberta-base-finetuned-jd-binary", help="HuggingFace情感模型")
@click.option("--no-console", is_flag=True, help="不打印终端报告")
@click.option("--ticket-csv", type=click.Path(), help="差评主题工单导出路径 (.csv)")
@click.option("--existing-ticket", type=click.Path(exists=True, dir_okay=False), help="已有工单CSV路径，用于匹配旧工单编号和保留手填字段")
@click.option("--high-risk-only", is_flag=True, help="仅导出高优先级差评主题（报告/JSON/工单均生效）")
@click.option("--anomaly-only", is_flag=True, help="仅导出关联了差评突增的主题")
@click.option("--priority", type=click.Choice(["高优先级", "中优先级", "低优先级"]), multiple=True, help="按优先级筛选，可重复指定")
@click.option("--status", type=click.Choice(["待处理", "处理中", "已解决", "观察中"]), multiple=True, help="按处理状态筛选，可重复指定")
@click.option("--appeared-last-n-days", type=int, help="仅保留最近N天内出现过的主题")
@click.option("--anomaly-last-n-days", type=int, help="仅保留最近N天内关联差评突增的主题")
def analyze(csv_file: str, output: str | None, fmt: str | None, product_name: str | None, model: str, no_console: bool,
            ticket_csv: str | None, existing_ticket: str | None,
            high_risk_only: bool, anomaly_only: bool,
            priority: tuple[str, ...], status: tuple[str, ...],
            appeared_last_n_days: int | None, anomaly_last_n_days: int | None):
    """分析单个商品的评论数据，输出情感分布、差评聚类、时间趋势报告。

    CSV_FILE: 评论数据CSV文件路径，需包含评论文本列（text/content/评论等）
    """
    product = product_name or _detect_product_name(csv_file)

    try:
        # 读取旧工单（如有）
        existing_ticket_df = None
        if existing_ticket:
            try:
                existing_ticket_df = _read_csv_safe(existing_ticket)
            except Exception as e:
                console.print(f"[yellow]⚠️  读取旧工单失败，将忽略: {e}[/yellow]")
                existing_ticket_df = None

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("读取评论数据...", total=100)

            df = _read_csv_safe(csv_file)
            progress.update(task, advance=10, description=f"已加载 {len(df)} 条评论")

            analyzer = ReviewAnalyzer(model_name=model)
            progress.update(task, advance=15, description="加载情感分析模型并分类...")

            result = analyzer.analyze(df, product_name=product)
            progress.update(task, advance=50, description="情感分类与差评聚类完成")

            # 先做一次 enrich，补全日期、匹配旧工单
            from .analyzer import enrich_cluster_metrics
            enrich_cluster_metrics(result, existing_ticket_df=existing_ticket_df)

            # 应用风险筛选（三端口径一致）
            priorities_list = list(priority) if priority else None
            statuses_list = list(status) if status else None
            has_filter = any([
                high_risk_only, anomaly_only,
                priorities_list, statuses_list,
                appeared_last_n_days is not None,
                anomaly_last_n_days is not None,
            ])
            if has_filter:
                result = filter_clusters(
                    result,
                    high_priority_only=high_risk_only,
                    with_anomaly_only=anomaly_only,
                    priorities=priorities_list,
                    statuses=statuses_list,
                    appeared_last_n_days=appeared_last_n_days,
                    anomaly_last_n_days=anomaly_last_n_days,
                )

            out_fmt = fmt or _check_output_format(output)
            ticket_saved = False
            if ticket_csv:
                progress.update(task, advance=10, description="生成差评主题工单 CSV...")
                export_ticket_csv(result, ticket_csv, existing_ticket_df=existing_ticket_df)
                ticket_saved = True

            if output:
                progress.update(task, advance=10, description=f"生成 {out_fmt or '文件'} 报告...")
                if out_fmt == "json":
                    export_json(result, output)
                elif out_fmt == "csv":
                    if result.raw_df is not None:
                        result.raw_df.to_csv(output, index=False, encoding="utf-8-sig")
                else:
                    reporter = HtmlReporter()
                    reporter.render(result, output)
                progress.update(task, advance=5)
            else:
                progress.update(task, advance=15)

            progress.update(task, completed=100, description="✅ 分析完成")

        if not no_console:
            console.print()
            print_to_console(result)

        if output:
            abs_out = os.path.abspath(output)
            console.print(f"\n[bold green]💾 报告已保存至:[/bold green] {abs_out}")
        if ticket_saved:
            abs_ticket = os.path.abspath(ticket_csv)
            console.print(f"[bold cyan]📋 差评主题工单已保存至:[/bold cyan] {abs_ticket}")

    except click.ClickException:
        raise
    except Exception as e:
        console.print(f"[bold red]❌ 分析失败:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@app.command()
@click.argument("csv_files", nargs=-1, type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--output", "-o", type=click.Path(), help="输出对比报告路径 (.html 或 .json)")
@click.option("--format", "-f", "fmt", type=click.Choice(["html", "json"]), help="输出格式")
@click.option("--model", type=str, default="uer/roberta-base-finetuned-jd-binary", help="HuggingFace情感模型")
@click.option("--no-console", is_flag=True, help="不打印终端报告")
def compare(csv_files: tuple[str, ...], output: str | None, fmt: str | None, model: str, no_console: bool):
    """对比多产品差评主题差异，找出共有问题和各产品独有问题。

    CSV_FILES: 多个评论CSV文件路径，每个文件对应一款产品
    """
    if len(csv_files) < 2:
        raise click.ClickException("请至少提供2个CSV文件进行对比")

    try:
        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            overall = progress.add_task(f"对比分析 {len(csv_files)} 款产品...", total=len(csv_files) * 100)

            analyzer = ReviewAnalyzer(model_name=model)
            for csv_path in csv_files:
                product = _detect_product_name(csv_path)
                progress.update(overall, description=f"分析 {product}...")
                df = _read_csv_safe(csv_path)
                result = analyzer.analyze(df, product_name=product)
                results.append(result)
                progress.advance(overall, 100)

            progress.update(overall, description="生成对比分析...")
            compare_data = compare_products(results)
            progress.update(overall, completed=len(csv_files) * 100, description="✅ 对比完成")

        if not no_console:
            console.print()
            print_compare_to_console(compare_data)

        out_fmt = fmt or _check_output_format(output)
        if output:
            if out_fmt == "json":
                export_compare_json(compare_data, output)
            else:
                reporter = HtmlReporter()
                reporter.render_compare(compare_data, output)
            abs_out = os.path.abspath(output)
            console.print(f"\n[bold green]💾 对比报告已保存至:[/bold green] {abs_out}")

    except click.ClickException:
        raise
    except Exception as e:
        console.print(f"[bold red]❌ 对比失败:[/bold red] {e}")
        sys.exit(1)


@app.command()
@click.argument("csv_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--interval", "-i", type=int, default=300, help="检查间隔秒数（默认300秒=5分钟）")
@click.option("--output", "-o", type=click.Path(), help="增量报告输出路径")
@click.option("--model", type=str, default="uer/roberta-base-finetuned-jd-binary", help="HuggingFace情感模型")
@click.option("--max-runs", type=int, default=0, help="最大运行次数（0=无限）")
def watch(csv_file: str, interval: int, output: str | None, model: str, max_runs: int):
    """监控模式：定期检查CSV文件变化，仅对新增评论做增量分析。

    CSV_FILE: 监控的评论CSV文件路径（程序会定时读取，只分析新增行）
    """
    if interval < 10:
        raise click.ClickException("检查间隔不能小于10秒")

    analyzer = ReviewAnalyzer(model_name=model)
    last_row_count = 0
    last_mtime = 0.0
    run_count = 0
    accumulated_labeled_df: pd.DataFrame | None = None

    product = _detect_product_name(csv_file)
    console.print(f"[bold cyan]👀 监控模式启动[/bold cyan]  产品: {product}  间隔: {interval}秒")
    console.print(f"[dim]按 Ctrl+C 停止监控[/dim]\n")

    try:
        while True:
            run_count += 1
            if max_runs > 0 and run_count > max_runs:
                console.print("[yellow]已达到最大运行次数，退出监控[/yellow]")
                break

            try:
                current_mtime = os.path.getmtime(csv_file)
                df = _read_csv_safe(csv_file)
                current_rows = len(df)

                if current_mtime > last_mtime and current_rows > last_row_count:
                    added_count = current_rows - last_row_count
                    console.print(
                        f"\n[bold green]🔄 检测到更新:[/bold green] "
                        f"+{added_count} 条新评论 (共{current_rows}条)"
                    )

                    if last_row_count == 0:
                        new_df = df
                    else:
                        new_df = df.iloc[last_row_count:].copy()

                    result = analyzer.analyze_incremental(
                        previous_labeled_df=accumulated_labeled_df,
                        new_df=new_df,
                        product_name=product,
                    )
                    accumulated_labeled_df = result.raw_df
                    print_to_console(result)

                    if output:
                        try:
                            if Path(output).suffix.lower() == ".json":
                                export_json(result, output)
                            else:
                                reporter = HtmlReporter()
                                reporter.render(result, output)
                            console.print(f"[green]💾 完整报告已更新 (共{result.total_reviews}条):[/green] {os.path.abspath(output)}")
                        except Exception as e:
                            console.print(f"[yellow]⚠️  写入报告失败: {e}[/yellow]")

                    last_row_count = current_rows
                    last_mtime = current_mtime
                else:
                    console.print(f"[dim]⏸  暂无新评论 ({time.strftime('%H:%M:%S')})[/dim]")

            except FileNotFoundError:
                console.print(f"[yellow]⚠️  文件不存在: {csv_file}[/yellow]")
            except Exception as e:
                console.print(f"[yellow]⚠️  读取/分析失败: {e}[/yellow]")

            if max_runs > 0 and run_count >= max_runs:
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        console.print("\n[cyan]👋 监控已停止[/cyan]")


def main():
    app()


if __name__ == "__main__":
    main()
