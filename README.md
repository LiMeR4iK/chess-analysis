# ♟ Chess Game Review

A Chess.com-style post-game analyzer for the terminal, powered by Stockfish and Rich. Fetches games directly from Chess.com URLs or local `.pgn` files, calculates player accuracy, and classifies every move.

## Features

- **Move Classification:** Detects `Brilliant (!!)`, `Great (!)`, `Best (★)`, `Excellent (✓)`, `Good (•)`, `Book (📖)`, `Inaccuracy (?!)`, `Mistake (?)`, `Miss (✕)`, and `Blunder (??)` using Win% delta and Static Exchange Evaluation (SEE).
- **Accuracy Calculation:** Combines volatility-weighted and harmonic means based on centipawn-to-win-probability logistic curves.
- **Opening Book Integration:** Verifies theory moves via the Lichess Opening Explorer API (with an offline fallback).
- **Terminal Dashboard:** Clean move-by-move table, progress bar, and summary cards built with `rich`.
- **JSON Export:** Optional structured report export for further analysis.

## Requirements

- Python 3.10+
- [Stockfish](https://stockfishchess.org/download/) executable

## Installation

```bash
git clone https://github.com/LiMeR4iK/chess-analysis.git
cd chess-analysis
pip install -r requirements.txt
```

## Usage

**Interactive mode:**
```bash
python main.py
```

**CLI arguments:**
```bash
# Analyze a Chess.com game URL with custom depth
python main.py https://www.chess.com/game/live/123456789 --depth 16

# Analyze a local PGN file and export JSON report
python main.py game.pgn --json report.json

# Run offline without opening book API requests
python main.py game.pgn --no-book-api
```

| Flag | Description | Default |
| :--- | :--- | :--- |
| `-d`, `--depth` | Stockfish search depth | `14` |
| `-t`, `--threads` | CPU threads allocated to Stockfish | `1` |
| `--no-book-api` | Disable Lichess Opening Explorer queries | `False` |
| `--json FILE` | Export full analysis report to a JSON file | `None` |

## License

MIT