#!/usr/bin/env python3
"""Phase 1 validation checkpoint."""

from src.data.database import PrismDatabase


def main() -> None:
    db = PrismDatabase()
    results = db.phase1_checkpoint()

    assert results["score_consistency_check"] == 0, "Score consistency failed"
    assert results["nfl_game_states"] > 10000, "Insufficient NFL data"
    assert results["nba_game_states"] > 10000, "Insufficient NBA data"
    assert results["high_confidence_matches"] > 100, "Insufficient market matches"

    print("Phase 1 checkpoint PASSED")
    print(results)


if __name__ == "__main__":
    main()
