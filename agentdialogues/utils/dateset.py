import json
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.progress import track
from rich.table import Table

console = Console()


def flatten_dialogue(chat_data):
    """Flatten dialogue."""
    chat_id = chat_data.get("chat_id", "unknown_chat")
    batch_id = chat_data.get("batch_id", "unknown_batch")
    seed = chat_data.get("seed", "unknown_seed")
    dialogue = chat_data.get("dialogue", [])

    rows = []
    for idx, msg in enumerate(dialogue):
        base = {
            "chat_id": chat_id,
            "batch_id": batch_id,
            "seed": seed,
            "round": idx // 2 + 1,
            "role": msg.get("role"),
            "name": msg.get("name"),
            "message": msg.get("message"),
        }

        for evaluator in msg.get("meta", []):
            ev_name = evaluator.get("evaluator", "unknown")
            score_dict = evaluator.get("score", {})
            for k, v in score_dict.items():
                base[f"{ev_name}_{k}"] = v

        rows.append(base)

    return rows


def prepare_simulation_dataset(batch_id: str, logs_base_path: str = "logs"):
    """Process simulation batch logs."""
    input_dir = Path(logs_base_path) / batch_id
    output_file = input_dir / "aggregated_scores.csv"

    if not input_dir.exists():
        console.print(f"[red]Directory not found:[/red] {input_dir}")
        return

    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        console.print(f"[red]No .json files found in:[/red] {input_dir}")
        return

    all_rows = []
    console.print(f"[bold green]Processing logs from:[/bold green] {input_dir}")

    for file_path in track(json_files, description="Processing dialogue logs"):
        try:
            with file_path.open("r", encoding="utf-8") as f:
                chat_log = json.load(f)
                flat_rows = flatten_dialogue(chat_log)
                all_rows.extend(flat_rows)
        except Exception as e:
            console.print(f"[yellow]Failed to process {file_path.name}:[/yellow] {e}")

    if not all_rows:
        console.print("[red]❌ No messages extracted. Aborting.[/red]")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(output_file, index=False)
    console.print(f"[bold cyan]Aggregated data saved to:[/bold cyan] {output_file}")

    # Summary table
    evaluator_columns = [
        col
        for col in df.columns
        if any(
            metric in col
            for metric in [
                "toxicity",
                "insult",
                "threat",
                "obscene",
                "identity_attack",
                "sexual_explicit",
            ]
        )
    ]
    if evaluator_columns:
        table = Table(
            title="Toxicity Metric Averages (All Messages)",
            header_style="bold magenta",
        )
        table.add_column("Metric", style="bold")
        table.add_column("Mean Score", justify="right")

        for col in evaluator_columns:
            mean_val = df[col].mean()
            table.add_row(col, f"{mean_val:.6f}")

        console.print(table)
