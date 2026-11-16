"""
Chess.com-style Game Review: Accuracy calculation, move classification
(Brilliant / Great / Best / Excellent / Good / Book / Inaccuracy / Mistake / Miss / Blunder),
and a terminal dashboard powered by Rich.

Dependencies:  pip install chess requests rich
Requires Stockfish (stockfish.exe next to this script or "stockfish" in PATH).

Usage examples:
  python main.py
  python main.py https://www.chess.com/game/live/123456789 --depth 16 --threads 4
  python main.py game.pgn --json report.json
"""
import argparse
import datetime
import io
import json
import math
import os
import re
import shutil
import statistics
from typing import Optional

import chess
import chess.engine
import chess.pgn
import requests
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

console = Console()

import sys

BASE_DIR = (
    os.path.dirname(sys.executable)
    if getattr(sys, "frozen", False)
    else os.path.dirname(os.path.abspath(__file__))
)
_local_sf = os.path.join(BASE_DIR, "stockfish.exe")
STOCKFISH_PATH = _local_sf if os.path.exists(_local_sf) else (shutil.which("stockfish") or _local_sf)

MATE = 10000
MAX_BOOK_PLIES = 24       # Fallback ply cap for opening book detection
BOOK_MIN_GAMES = 50       # Minimum games in Lichess Explorer to count as theory
BRILLIANT_ALT_CAP = 0.97  # If already completely winning without the sacrifice -> not Brilliant
DEBUG_BRILLIANT = False   # True -> print why a sacrifice candidate was rejected as Brilliant

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}

SEE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100,
}

CLASS_META = {
    "brilliant":  ("!!", "Brilliant",  "bold bright_cyan"),
    "great":      ("!",  "Great",      "bold dodger_blue1"),
    "best":       ("★",  "Best",       "bold green"),
    "excellent":  ("✓",  "Excellent",  "green"),
    "good":       ("•",  "Good",       "sea_green2"),
    "book":       ("📖", "Book",       "tan"),
    "inaccuracy": ("?!", "Inaccuracy", "yellow"),
    "mistake":    ("?",  "Mistake",    "dark_orange"),
    "miss":       ("✕",  "Miss",       "indian_red1"),
    "blunder":    ("??", "Blunder",    "bold red"),
}

ORDER = [
    "brilliant", "great", "best", "excellent", "good", "book",
    "inaccuracy", "mistake", "miss", "blunder",
]

HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python/ChessAnalyzer"}


# ----------------------------------------------------------------- PGN ----
def get_game_pgn(url: str) -> str:
    match = re.search(r'(?:live|daily|game/live|game/daily)/(\d+)', url)
    if not match:
        raise ValueError("Could not extract game ID from the provided URL.")

    game_id = match.group(1)
    res = requests.get(
        f"https://www.chess.com/callback/live/game/{game_id}",
        headers=HTTP_HEADERS,
        timeout=10,
    )

    white_player, year, month = None, None, None
    if res.status_code == 200:
        game_data = res.json().get("game", {})
        hdrs = game_data.get("pgnHeaders", {})
        white_player = hdrs.get("White") or game_data.get("white", {}).get("username")
        date_str = hdrs.get("Date")
        if date_str:
            parts = date_str.split('.')
            if len(parts) >= 2:
                year, month = parts[0], parts[1].zfill(2)

    if not year or not month:
        now = datetime.datetime.now()
        year, month = str(now.year), f"{now.month:02d}"

    if white_player:
        pub = requests.get(
            f"https://api.chess.com/pub/player/{white_player}/games/{year}/{month}",
            headers=HTTP_HEADERS,
            timeout=10,
        )
        if pub.status_code == 200:
            for g in pub.json().get("games", []):
                if game_id in g.get("url", "") and g.get("pgn"):
                    return g["pgn"]

    raise RuntimeError(f"Failed to fetch PGN for game {game_id}.")


# ------------------------------------------------------------ Opening Book ----
def is_book_move_api(board: chess.Board, move: chess.Move, session: requests.Session) -> Optional[bool]:
    """
    Checks via Lichess Opening Explorer whether the move is established theory.
    Returns True/False, or None if a network error occurs.
    """
    try:
        resp = session.get(
            "https://explorer.lichess.ovh/lichess",
            params={
                "fen": board.fen(),
                "ratings": "1600,1800,2000,2200,2500",
                "speeds": "blitz,rapid,classical",
                "topGames": 0,
                "recentGames": 0,
            },
            timeout=2.5,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        uci = move.uci()
        for m in data.get("moves", []):
            if m.get("uci") == uci:
                total = m.get("white", 0) + m.get("draws", 0) + m.get("black", 0)
                return total >= BOOK_MIN_GAMES
        return False
    except requests.RequestException:
        return None


# -------------------------------------------------------------------- Math ----
def win_pct(cp: float) -> float:
    """Win probability (0..100) from centipawn evaluation (Lichess logistic curve)."""
    cp = max(-MATE, min(MATE, cp))
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def move_accuracy(win_before: float, win_after: float) -> float:
    """Single-move accuracy (0..100) from the mover's perspective."""
    drop = max(0.0, win_before - win_after)
    acc = 103.1668 * math.exp(-0.04354 * drop) - 3.1669 + 1
    return max(0.0, min(100.0, acc))


def game_accuracy(values_weights) -> float:
    """Mean of volatility-weighted and harmonic accuracy."""
    if not values_weights:
        return 0.0
    tw = sum(w for _, w in values_weights)
    weighted = sum(a * w for a, w in values_weights) / tw
    harmonic = len(values_weights) / sum(1 / max(a, 1.0) for a, _ in values_weights)
    return (weighted + harmonic) / 2


def material_diff(board: chess.Board, color: bool) -> int:
    total = 0
    for piece in board.piece_map().values():
        v = PIECE_VALUES[piece.piece_type]
        total += v if piece.color == color else -v
    return total


# ------------------------------------------------------------------ Engine ----
def analyse_position(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    limit: chess.engine.Limit,
    prev_cp: Optional[int] = None,
) -> dict:
    """
    Deterministic position analysis:
    - Game-over positions return immediately without invoking the engine.
    - Forced single-move positions use a fast shallow search (depth=6).
    - Standard positions always use multipv=2 for consistent Great/Brilliant checks.
    """
    if board.is_game_over():
        if board.is_checkmate():
            cp = -MATE if board.turn == chess.WHITE else MATE
            return {"cp": cp, "eval": "#", "pv": [], "lines": []}
        return {"cp": 0, "eval": "0.00", "pv": [], "lines": []}

    if board.legal_moves.count() == 1:
        only_move = next(iter(board.legal_moves))
        infos = engine.analyse(board, chess.engine.Limit(depth=6), multipv=1)
        sc = infos[0]["score"].white()
        cp0 = sc.score(mate_score=MATE)
        ev = f"{'+' if sc.mate() > 0 else '-'}M{abs(sc.mate())}" if sc.is_mate() else f"{cp0 / 100:+.2f}"
        pv = infos[0].get("pv") or [only_move]
        return {"cp": cp0, "eval": ev, "pv": pv, "lines": [(pv, cp0)]}

    infos = engine.analyse(board, limit, multipv=2)

    lines = []
    for inf in infos:
        sc = inf["score"].white()
        lines.append((inf.get("pv", []), sc.score(mate_score=MATE), sc))

    pv0, cp0, sc0 = lines[0]
    if sc0.is_mate():
        m = sc0.mate()
        ev = f"{'+' if m > 0 else '-'}M{abs(m)}"
    else:
        ev = f"{cp0 / 100:+.2f}"
    return {"cp": cp0, "eval": ev, "pv": pv0, "lines": [(p, c) for p, c, _ in lines]}


# ---------------------------------------------------------- Classification ----
def _see(board: chess.Board, sq: int, target_val: int) -> int:
    """Static Exchange Evaluation (SEE) gain on target square."""
    caps = [m for m in board.legal_moves if m.to_square == sq and board.is_capture(m)]
    if not caps:
        return 0
    m = min(caps, key=lambda mv: SEE_VALUES[board.piece_type_at(mv.from_square)])
    attacker_val = SEE_VALUES[board.piece_type_at(m.from_square)]
    board.push(m)
    gain = target_val - _see(board, sq, attacker_val)
    board.pop()
    return max(0, gain)


def static_sacrifice(board_before: chess.Board, move: chess.Move) -> int:
    """Net material sacrificed on the destination square after immediate recaptures."""
    piece = board_before.piece_at(move.from_square)
    captured = board_before.piece_at(move.to_square)
    cap_val = PIECE_VALUES[captured.piece_type] if captured else 0
    b = board_before.copy()
    b.push(move)
    gain = _see(b, move.to_square, SEE_VALUES[piece.piece_type])
    return gain - cap_val


def is_brilliant(board_before, move, res, next_res, s, ep_after, loss):
    piece = board_before.piece_at(move.from_square)
    if not piece or piece.piece_type == chess.KING:
        return False, None

    color = board_before.turn
    base = material_diff(board_before, color)
    b = board_before.copy()
    b.push(move)
    pv = list(next_res["pv"][:8])
    if len(pv) > 1 and len(pv) % 2:
        pv = pv[:-1]
    for m in pv:
        b.push(m)
    pv_net = material_diff(b, color) - base
    static = static_sacrifice(board_before, move)

    b_after = board_before.copy()
    b_after.push(move)
    promo_sac = (
        move.promotion is not None
        and len(b_after.attackers(not color, move.to_square)) > 0
    )

    if not (static >= 2 or pv_net <= -2 or promo_sac):
        return False, None

    if loss > 0.03:
        return False, f"move is not best (loss {loss:.3f} > 0.03)"
    if ep_after < 0.40:
        return False, f"resulting position is bad (ep={ep_after:.2f})"
    alts = [c for p, c in res["lines"] if p and p[0] != move]
    if alts:
        alt_ep = win_pct(s * alts[0]) / 100
        if alt_ep >= BRILLIANT_ALT_CAP:
            return False, f"already completely winning without sacrifice (alt ep={alt_ep:.3f} >= {BRILLIANT_ALT_CAP})"
    return True, None


def is_great(board_before: chess.Board, res: dict, played: chess.Move,
             best_move: Optional[chess.Move], ep_before: float,
             prev_loss: float, s: int) -> bool:
    if len(res["lines"]) < 2 or played != best_move or ep_before > 0.92:
        return False

    ep_second = win_pct(s * res["lines"][1][1]) / 100
    gap = ep_before - ep_second
    if gap < 0.17:
        return False

    if board_before.move_stack:
        last_opp_move = board_before.peek()
        if board_before.is_capture(played) and played.to_square == last_opp_move.to_square:
            if prev_loss < 0.15:
                return False

    punished_mistake = prev_loss >= 0.12 and ep_before >= 0.45
    turning_point = ep_before >= 0.68 and ep_second <= 0.50
    holding_draw = ep_before >= 0.45 and ep_second <= 0.28
    return punished_mistake or turning_point or holding_draw


def classify(loss: float, played_is_best: bool, is_book: bool,
             brilliant: bool, great: bool, ep_before: float,
             ep_after: float, prev_loss: float) -> str:
    if is_book:
        return "book"
    if brilliant:
        return "brilliant"
    if great:
        return "great"
    if played_is_best:
        return "best"
    if loss <= 0.02:
        return "excellent"
    if loss <= 0.06:
        return "good"
    if loss <= 0.12:
        return "inaccuracy"

    if prev_loss >= 0.14 and ep_before >= 0.45 and loss >= 0.12:
        return "miss"

    if loss <= 0.26:
        return "mistake"

    if ep_after >= 0.85:
        return "mistake"

    return "blunder"


# --------------------------------------------------------------- Rich UI ----
def acc_color(acc: float) -> str:
    if acc >= 85:
        return "bold bright_green"
    if acc >= 75:
        return "bold green"
    if acc >= 65:
        return "bold yellow"
    return "bold red"


def render_summary(white_name: str, black_name: str, result: str, depth: int,
                   white_acc: float, black_acc: float, counts: dict):
    w_card = Panel(
        Text.assemble(
            ("⬜ ", ""),
            (f"{white_name}\n", "bold white"),
            ("Accuracy: ", "dim"),
            (f"{white_acc:.1f}%", acc_color(white_acc)),
        ),
        border_style="white",
        padding=(0, 2),
    )
    b_card = Panel(
        Text.assemble(
            ("⬛ ", ""),
            (f"{black_name}\n", "bold white"),
            ("Accuracy: ", "dim"),
            (f"{black_acc:.1f}%", acc_color(black_acc)),
        ),
        border_style="steel_blue1",
        padding=(0, 2),
    )
    console.print(Columns([w_card, b_card]))

    table = Table(
        title=f"Game Review Summary  •  Result: {result}  •  Depth: {depth}",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="grey50",
    )
    table.add_column("Classification", justify="left", min_width=24)
    table.add_column(white_name, justify="right", min_width=10)
    table.add_column(black_name, justify="right", min_width=10)

    for k in ORDER:
        w_cnt = counts[True][k]
        b_cnt = counts[False][k]
        if w_cnt or b_cnt:
            sym, name_en, style = CLASS_META[k]
            label_cell = Text.assemble((f"{sym:<3} ", style), (name_en, style))
            table.add_row(label_cell, str(w_cnt), str(b_cnt))

    console.print(table)


# -------------------------------------------------------------------- Main ----
def analyze_game(source: str, depth: int = 14, threads: int = 2,
                 use_book_api: bool = True, json_out: Optional[str] = None):
    if not os.path.exists(STOCKFISH_PATH):
        console.print(f"[bold red]Error:[/bold red] Stockfish executable not found ({STOCKFISH_PATH}).")
        return

    try:
        with console.status("[cyan]Fetching game PGN...[/cyan]"):
            if os.path.isfile(source):
                with open(source, encoding="utf-8") as f:
                    pgn_text = f.read()
            else:
                pgn_text = get_game_pgn(source)
    except Exception as e:
        console.print(f"[bold red]Load error:[/bold red] {e}")
        return

    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if not game:
        console.print("[bold red]Failed to parse PGN.[/bold red]")
        return

    moves = list(game.mainline_moves())
    boards = [game.board()]
    for m in moves:
        b = boards[-1].copy()
        b.push(m)
        boards.append(b)

    white_name = game.headers.get("White", "White")
    black_name = game.headers.get("Black", "Black")
    result = game.headers.get("Result", "*")

    console.print(
        Panel(
            f"[bold white]{white_name}[/bold white]  [dim]vs[/dim]  [bold steel_blue1]{black_name}[/bold steel_blue1]\n"
            f"[dim]Event:[/dim] {game.headers.get('Event', 'Live Chess')}   "
            f"[dim]Date:[/dim] {game.headers.get('Date', '—')}   "
            f"[dim]Depth:[/dim] {depth}   [dim]Threads:[/dim] {threads}",
            title="[bold cyan]♟ CHESS GAME REVIEW[/bold cyan]",
            border_style="cyan",
            expand=False,
        )
    )

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    try:
        engine.configure({"Threads": 1, "Hash": 64})
    except Exception:
        pass
    limit = chess.engine.Limit(depth=depth)

    results = []
    last_cp = None
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=32),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Analyzing positions with Stockfish...", total=len(boards))
            for b in boards:
                res = analyse_position(engine, b, limit, prev_cp=last_cp)
                results.append(res)
                last_cp = res["cp"]
                progress.advance(task)
    finally:
        engine.quit()

    wins_white = [win_pct(r["cp"]) for r in results]
    window = max(2, min(8, len(moves) // 10))

    counts = {True: {k: 0 for k in ORDER}, False: {k: 0 for k in ORDER}}
    acc_data = {True: [], False: []}
    move_log = []

    prev_loss = 0.0
    in_book = True
    http_session = requests.Session()
    http_session.headers.update(HTTP_HEADERS)

    for i, move in enumerate(moves):
        ply = i + 1
        color = boards[i].turn
        s = 1 if color == chess.WHITE else -1
        res, nxt = results[i], results[i + 1]

        best_move = res["pv"][0] if res["pv"] else None
        played_is_best = move == best_move

        wb = win_pct(s * res["cp"])
        wa = wb if played_is_best else win_pct(s * nxt["cp"])
        ep_before, ep_after = wb / 100, wa / 100
        loss = max(0.0, ep_before - ep_after)

        move_is_book = False
        if in_book and ply <= MAX_BOOK_PLIES:
            if use_book_api:
                api_res = is_book_move_api(boards[i], move, http_session)
                if api_res is True:
                    move_is_book = True
                elif api_res is False:
                    in_book = False
                else:
                    move_is_book = ply <= 6 and loss <= 0.02
                    if not move_is_book:
                        in_book = False
            else:
                move_is_book = ply <= 6 and loss <= 0.02
                if not move_is_book:
                    in_book = False
        else:
            in_book = False

        brilliant, why = (False, None) if move_is_book else is_brilliant(
            boards[i], move, res, nxt, s, ep_after, loss
        )
        great = (not move_is_book) and (not brilliant) and is_great(
            boards[i], res, move, best_move, ep_before, prev_loss, s
        )
        label = classify(
            loss, played_is_best, move_is_book, brilliant, great,
            ep_before, ep_after, prev_loss
        )
        prev_loss = loss

        counts[color][label] += 1
        seg = wins_white[i:i + window]
        w = max(0.5, min(12.0, statistics.pstdev(seg) if len(seg) > 1 else 0.5))
        acc_data[color].append((move_accuracy(wb, wa), w))

        san = boards[i].san(move)
        best_san = boards[i].san(best_move) if best_move else None
        num = (ply + 1) // 2

        move_log.append({
            "ply": ply,
            "move_number": num,
            "color": "white" if color == chess.WHITE else "black",
            "san": san,
            "uci": move.uci(),
            "eval": nxt["eval"],
            "win_pct_white": round(wins_white[i + 1], 2),
            "classification": label,
            "best_move": best_san,
            "brilliant_debug": why if (DEBUG_BRILLIANT and why and loss <= 0.10) else None,
        })

    http_session.close()

    moves_table = Table(
        title="Move-by-Move Analysis",
        box=box.SIMPLE_HEAVY,
        header_style="bold dim",
    )
    moves_table.add_column("#", justify="right", style="dim", width=4)
    moves_table.add_column("White", min_width=26)
    moves_table.add_column("Eval", justify="right", width=7)
    moves_table.add_column("Black", min_width=26)
    moves_table.add_column("Eval", justify="right", width=7)

    def format_move_cell(entry: Optional[dict]) -> Text:
        if not entry:
            return Text("")
        lbl = entry["classification"]
        sym, _, style = CLASS_META[lbl]
        cell = Text()
        cell.append(f"{entry['san']:<7} ", style="bold white")
        cell.append(f"[{sym}]", style=style)
        if lbl in ("inaccuracy", "mistake", "miss", "blunder") and entry["best_move"]:
            cell.append(f" → {entry['best_move']}", style="dim green")
        return cell

    for idx in range(0, len(move_log), 2):
        w_entry = move_log[idx]
        b_entry = move_log[idx + 1] if idx + 1 < len(move_log) else None
        moves_table.add_row(
            f"{w_entry['move_number']}.",
            format_move_cell(w_entry),
            w_entry["eval"],
            format_move_cell(b_entry),
            b_entry["eval"] if b_entry else "",
        )

    console.print(moves_table)
    console.print()

    white_acc = round(game_accuracy(acc_data[True]), 1)
    black_acc = round(game_accuracy(acc_data[False]), 1)
    render_summary(white_name, black_name, result, depth, white_acc, black_acc, counts)

    if json_out:
        report = {
            "white": {"name": white_name, "accuracy": white_acc, "counts": counts[True]},
            "black": {"name": black_name, "accuracy": black_acc, "counts": counts[False]},
            "result": result,
            "depth": depth,
            "moves": move_log,
        }
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        console.print(f"\n[green]JSON report saved to:[/green] [bold]{json_out}[/bold]")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chess.com-style Game Review analyzer with terminal UI."
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="Chess.com game URL or path to a local .pgn file",
    )
    parser.add_argument(
        "-d", "--depth",
        type=int,
        default=14,
        help="Stockfish search depth (default: 14)",
    )
    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=2,
        help="CPU threads allocated to Stockfish (default: 2)",
    )
    parser.add_argument(
        "--no-book-api",
        action="store_true",
        help="Disable Lichess Opening Explorer requests (offline mode)",
    )
    parser.add_argument(
        "--json",
        dest="json_out",
        metavar="FILE",
        help="Export detailed analysis report to a JSON file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    interactive_mode = args.source is None
    src = args.source

    if interactive_mode:
        console.print("[bold cyan]♟ Chess Game Review[/bold cyan] [dim](ready)[/dim]")
        src = console.input("[bold cyan]Chess.com game URL (or path to .pgn): [/bold cyan]").strip()

    try:
        if src:
            analyze_game(
                src,
                depth=args.depth,
                threads=args.threads,
                use_book_api=not args.no_book_api,
                json_out=args.json_out,
            )
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error:[/bold red] {e}")
    finally:
        if interactive_mode and getattr(sys, "frozen", False):
            console.input("\n[dim]Press Enter to exit...[/dim]")