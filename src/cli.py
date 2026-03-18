"""
CLI Interface — Premium Rich terminal UI for the Bet Prediction Agent.
"""

import asyncio
from collections import defaultdict
from datetime import date, datetime
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box

from src.agent import PredictionAgent
from src.models import Sport, BetType, Prediction, ParlayPrediction, SPORT_EMOJIS

console = Console()

BET_TYPE_LABELS = {
    BetType.MONEYLINE: "🏆 Winner",
    BetType.GAME_RESULT_90: "⚽ Game Result (90'+ST)",
    BetType.SPREAD: "📊 Spread / Handicap",
    BetType.OVER_UNDER: "⬆️⬇️ Total O/U",
    BetType.TEAM_TOTAL: "🎯 Team Total",
    BetType.BOTH_TEAMS_SCORE: "⚽ BTTS",
    BetType.DOUBLE_CHANCE: "🛡️ Double Chance",
    BetType.CORNERS: "🔲 Corners",
    BetType.HALFTIME_RESULT: "⏱️ HT Result",
    BetType.HALFTIME_OVER_UNDER: "⏱️ HT O/U",
    BetType.PLAYER_PROPS: "🎯 Player Props",
    BetType.DRAW_NO_BET: "🔄 Draw No Bet",
    BetType.FIRST_HALF: "⏱️ First Half",
    BetType.CORRECT_SCORE: "🎯 Correct Score",
}


def display_predictions(
    predictions: dict[Sport, list[Prediction]], title: str = "Today's Predictions"
):
    """Display predictions grouped by match with all markets."""
    console.print()
    now = datetime.now()
    console.print(Panel(
        f"[bold cyan]{title}[/]\n[dim]📅 {now.strftime('%A, %B %d, %Y')} • ⏰ {now.strftime('%I:%M %p')}[/]",
        expand=False,
        border_style="cyan",
    ))

    for sport, preds in predictions.items():
        if not preds:
            continue

        emoji = SPORT_EMOJIS.get(sport, "🏆")
        console.print(f"\n[bold yellow]{'━' * 60}[/]")
        console.print(f"[bold yellow]  {emoji} {sport.value.upper().replace('-', ' ')}[/]")
        console.print(f"[bold yellow]{'━' * 60}[/]")

        # Group by match
        matches = defaultdict(list)
        for p in preds:
            matches[p.event.id].append(p)

        for event_id, match_preds in matches.items():
            ev = match_preds[0].event
            kick_off = ev.start_time.strftime("%b %d, %Y • %I:%M %p")

            console.print()
            console.print(Panel(
                f"[bold white]{ev.home_team.name}[/] [dim]vs[/] [bold white]{ev.away_team.name}[/]\n"
                f"[dim]🏟️  {ev.tournament.name} ({ev.tournament.country})[/]\n"
                f"[dim]📅 {kick_off}[/]",
                border_style="blue",
                expand=False,
            ))

            # Group by bet type
            by_type = defaultdict(list)
            for p in match_preds:
                by_type[p.bet_type].append(p)

            for bt, bt_preds in by_type.items():
                label = BET_TYPE_LABELS.get(bt, bt.value)
                table = Table(
                    title=f"  {label}",
                    box=box.SIMPLE_HEAVY,
                    show_lines=False,
                    title_style="bold",
                    padding=(0, 1),
                )
                table.add_column("Pick", style="bold green", min_width=30)
                table.add_column("Odds", justify="center", min_width=10)
                table.add_column("Conf", justify="center", min_width=8)
                table.add_column("Value", justify="center", min_width=8)
                table.add_column("Note", style="dim", min_width=12)

                for pred in bt_preds[:4]:
                    conf = pred.confidence
                    conf_style = (
                        "[bold green]" if conf >= 75
                        else "[bold yellow]" if conf >= 65
                        else "[bold red]"
                    )

                    odds_str = pred.american_odds if pred.american_odds else (
                        f"{pred.odds:.2f}" if pred.odds > 0 else "-"
                    )

                    val = pred.value_rating
                    val_str = f"{val:+.3f}" if val != 0 else "-"
                    if val > 0.05:
                        val_str = f"[green]{val_str}[/]"
                    elif val < 0:
                        val_str = f"[red]{val_str}[/]"

                    push_str = f"⚠️ {pred.push_note}" if pred.push_note else ""

                    table.add_row(
                        pred.pick,
                        odds_str,
                        f"{conf_style}{conf:.0f}%[/]",
                        val_str,
                        push_str,
                    )

                console.print(table)


def display_parlay(parlay: ParlayPrediction, title: str = "Recommended Parlay"):
    """Display a premium parlay recommendation."""
    if not parlay.legs:
        console.print("[yellow]⚠️ No parlay available.[/]")
        return

    # Count sports
    sports_used = set(leg.event.tournament.sport for leg in parlay.legs)
    mix_label = "🌐 MIXED SPORTS" if len(sports_used) > 1 else ""
    sports_str = " ".join(SPORT_EMOJIS.get(s, "🏆") for s in sports_used)

    console.print()
    table = Table(
        title=f"🎯 {title} {mix_label} {sports_str}",
        box=box.DOUBLE_EDGE,
        show_lines=True,
        title_style="bold magenta",
    )
    table.add_column("#", justify="center", style="bold", width=3)
    table.add_column("Match", style="white", min_width=28)
    table.add_column("Sport", justify="center", width=6)
    table.add_column("Pick", style="bold green", min_width=22)
    table.add_column("Odds", justify="center", width=8)
    table.add_column("Conf", justify="center", width=6)
    table.add_column("Date & Time", justify="center", style="dim", width=18)

    for i, leg in enumerate(parlay.legs, 1):
        sport = leg.event.tournament.sport
        emoji = SPORT_EMOJIS.get(sport, "🏆")

        conf = leg.confidence
        conf_style = (
            "[bold green]" if conf >= 75
            else "[bold yellow]" if conf >= 65
            else "[bold red]"
        )

        odds_str = leg.american_odds if leg.american_odds else (
            f"{leg.odds:.2f}" if leg.odds > 0 else "-"
        )

        push_str = f"\n[dim yellow]⚠️ {leg.push_note}[/]" if leg.push_note else ""

        dt = leg.event.start_time.strftime("%b %d • %I:%M %p")

        table.add_row(
            str(i),
            f"{leg.event.home_team.name} vs\n{leg.event.away_team.name}\n[dim]{leg.event.tournament.name}[/]",
            f"{emoji}",
            f"{leg.pick}{push_str}",
            odds_str,
            f"{conf_style}{conf:.0f}%[/]",
            dt,
        )

    console.print(table)

    # Summary
    console.print(Panel(
        f"[bold]📊 Combined Confidence:[/] {parlay.combined_confidence:.1f}%\n"
        f"[bold]💰 Combined Odds:[/] {parlay.combined_odds:.2f}x\n"
        f"[bold]📈 Expected Value:[/] {parlay.expected_value:+.4f}\n"
        f"[bold]⚠️  Risk Level:[/] {parlay.risk_level.upper()}\n"
        f"[bold]💵 Recommended Stake:[/] ${parlay.recommended_stake:.2f}",
        title="📊 Parlay Summary",
        border_style="green",
    ))

    if parlay.reasoning:
        console.print(Panel(
            parlay.reasoning,
            title="💡 Analysis",
            border_style="blue",
        ))


async def run_cli():
    """Main CLI interaction loop."""
    agent = PredictionAgent()
    now = datetime.now()

    console.print(Panel(
        "[bold cyan]🏆 PREMIUM BET PREDICTION AI AGENT[/]\n"
        "Powered by SofaScore data + Statistical Analysis + AI\n"
        f"[dim]📅 {now.strftime('%A, %B %d, %Y')} • ⏰ {now.strftime('%I:%M %p')}[/]\n"
        "[dim]Mix any sports together — Soccer ⚽ + Basketball 🏀 + Tennis 🎾 + More![/]",
        border_style="cyan",
    ))

    while True:
        console.print("\n[bold]Choose an option:[/]")
        console.print("  [1] 📊 Today's Predictions (All Markets by Match)")
        console.print("  [2] 🎯 Build Parlay (Mix Any Sports)")
        console.print("  [3] 💰 Find Value Bets")
        console.print("  [4] 🔍 Analyze Single Game")
        console.print("  [5] 📋 Full Daily Report")
        console.print("  [6] ⚙️  Select Sports Filter")
        console.print("  [7] 🚪 Exit")

        choice = console.input("\n[bold cyan]Enter choice (1-7):[/] ").strip()

        if choice == "1":
            with console.status("[bold green]🔄 Fetching & analyzing all markets across all sports..."):
                preds = await agent.get_todays_predictions(min_confidence=50)
            display_predictions(preds, "📊 Today's Full Market Analysis")

        elif choice == "2":
            legs = console.input("Number of legs (default 6): ").strip()
            num_legs = int(legs) if legs.isdigit() else 6

            strategy = console.input("Strategy [safe/balanced/value] (default balanced): ").strip()
            if strategy not in {"safe", "balanced", "value"}:
                strategy = "balanced"

            console.print("[dim]🌐 Building mixed-sport parlay across ALL sports...[/]")
            with console.status(f"[bold green]🔄 Building {num_legs}-leg {strategy} parlay..."):
                parlay = await agent.build_parlay(num_legs=num_legs, strategy=strategy)
            display_parlay(parlay, f"{num_legs}-Leg {strategy.capitalize()} Parlay")

        elif choice == "3":
            with console.status("[bold green]🔄 Scanning for value bets across all sports..."):
                value_bets = await agent.find_value_bets()
            if value_bets:
                table = Table(title="💰 Value Bets", box=box.ROUNDED, show_lines=True)
                table.add_column("Match", min_width=25)
                table.add_column("Pick", style="bold green", min_width=25)
                table.add_column("Sport", justify="center")
                table.add_column("Odds", justify="center")
                table.add_column("Our Prob", justify="center")
                table.add_column("Value", justify="center", style="bold green")
                table.add_column("Date/Time", style="dim")
                for vb in value_bets[:20]:
                    sport_e = SPORT_EMOJIS.get(vb.event.tournament.sport, "🏆")
                    odds_str = vb.american_odds if vb.american_odds else f"{vb.odds:.2f}"
                    table.add_row(
                        f"{vb.event.home_team.name} vs\n{vb.event.away_team.name}",
                        vb.pick,
                        sport_e,
                        odds_str,
                        f"{vb.probability*100:.0f}%",
                        f"{vb.value_rating:+.3f}",
                        vb.event.start_time.strftime("%b %d • %I:%M %p"),
                    )
                console.print(table)
            else:
                console.print("[yellow]⚠️ No value bets found.[/]")

        elif choice == "4":
            eid = console.input("Enter SofaScore Event ID: ").strip()
            if eid.isdigit():
                with console.status("[bold green]🔄 Deep analyzing game..."):
                    preds = await agent.analyze_single_game(int(eid))
                for pred in preds:
                    am = f" ({pred.american_odds})" if pred.american_odds else ""
                    push = f"\n⚠️ {pred.push_note}" if pred.push_note else ""
                    console.print(Panel(
                        f"[bold]{pred.pick}{am}[/] — {pred.confidence:.0f}% confidence\n"
                        f"Type: {pred.bet_type.value} | Odds: {pred.odds:.2f}"
                        f"{push}\n\n{pred.reasoning}",
                        title=f"{pred.event.home_team.name} vs {pred.event.away_team.name} • {pred.event.start_time.strftime('%b %d • %I:%M %p')}",
                        border_style="green",
                    ))
            else:
                console.print("[red]Invalid event ID.[/]")

        elif choice == "5":
            with console.status("[bold green]🔄 Generating comprehensive daily report..."):
                report = await agent.generate_daily_report()
            console.print(report)

        elif choice == "6":
            console.print("\n[bold]Available sports:[/]")
            for i, sport in enumerate(Sport, 1):
                emoji = SPORT_EMOJIS.get(sport, "🏆")
                console.print(f"  {emoji} [{i}] {sport.value}")
            console.print("\n[dim]💡 Tip: Use option [2] Build Parlay to mix ALL sports together automatically![/]")

        elif choice == "7":
            await agent.close()
            console.print("[bold green]Goodbye! Bet responsibly. 🍀[/]")
            break

        else:
            console.print("[red]Invalid choice.[/]")


def main():
    """Entry point."""
    asyncio.run(run_cli())


if __name__ == "__main__":
    main()
