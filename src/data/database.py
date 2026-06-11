"""DuckDB interface and schema management for PRISM."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from config.settings import Settings

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS game_states (
    game_id        VARCHAR NOT NULL,
    sport          VARCHAR NOT NULL,
    season         INTEGER NOT NULL,
    game_date      DATE NOT NULL,
    home_team      VARCHAR NOT NULL,
    away_team      VARCHAR NOT NULL,
    seconds_remaining INTEGER NOT NULL,
    game_period    INTEGER NOT NULL,
    score_differential INTEGER NOT NULL,
    home_score     INTEGER NOT NULL,
    away_score     INTEGER NOT NULL,
    possession     VARCHAR,
    is_scoring_event BOOLEAN DEFAULT FALSE,
    event_type     VARCHAR,
    created_at     TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (game_id, seconds_remaining)
);

CREATE TABLE IF NOT EXISTS contracts (
    contract_id    VARCHAR NOT NULL,
    market_source  VARCHAR NOT NULL,
    sport          VARCHAR NOT NULL,
    home_team      VARCHAR NOT NULL,
    away_team      VARCHAR NOT NULL,
    game_date      DATE NOT NULL,
    contract_type  VARCHAR NOT NULL,
    resolved_outcome VARCHAR,
    resolution_price FLOAT,
    created_at     TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (contract_id, market_source)
);

CREATE TABLE IF NOT EXISTS market_prices (
    contract_id    VARCHAR NOT NULL,
    market_source  VARCHAR NOT NULL,
    timestamp      TIMESTAMP NOT NULL,
    yes_price      FLOAT NOT NULL,
    no_price       FLOAT NOT NULL,
    yes_bid        FLOAT,
    yes_ask        FLOAT,
    volume         FLOAT,
    PRIMARY KEY (contract_id, market_source, timestamp)
);

CREATE TABLE IF NOT EXISTS game_contract_map (
    game_id        VARCHAR NOT NULL,
    contract_id    VARCHAR NOT NULL,
    market_source  VARCHAR NOT NULL,
    match_confidence FLOAT NOT NULL,
    match_method   VARCHAR NOT NULL,
    PRIMARY KEY (game_id, contract_id, market_source)
);

CREATE TABLE IF NOT EXISTS model_estimates (
    game_id        VARCHAR NOT NULL,
    model_name     VARCHAR NOT NULL,
    seconds_remaining INTEGER NOT NULL,
    home_win_prob  FLOAT NOT NULL,
    model_version  VARCHAR NOT NULL,
    created_at     TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (game_id, model_name, seconds_remaining)
);

CREATE TABLE IF NOT EXISTS edge_signals (
    game_id        VARCHAR NOT NULL,
    contract_id    VARCHAR NOT NULL,
    market_source  VARCHAR NOT NULL,
    seconds_remaining INTEGER NOT NULL,
    model_prob     FLOAT NOT NULL,
    market_price   FLOAT NOT NULL,
    edge           FLOAT NOT NULL,
    edge_type      VARCHAR NOT NULL,
    created_at     TIMESTAMP DEFAULT NOW()
);
"""


class PrismDatabase:
  """DuckDB wrapper for PRISM analytical storage."""

  def __init__(self, db_path: Path | str | None = None, settings: Settings | None = None):
      self.settings = settings or Settings()
      self.db_path = Path(db_path) if db_path else self.settings.prism_db_path
      self._conn: duckdb.DuckDBPyConnection | None = None

  @property
  def conn(self) -> duckdb.DuckDBPyConnection:
      """Lazy-connect to DuckDB and ensure schema exists."""
      if self._conn is None:
          self.db_path.parent.mkdir(parents=True, exist_ok=True)
          self._conn = duckdb.connect(str(self.db_path))
          self.initialize_schema()
      return self._conn

  def initialize_schema(self) -> None:
      """Create all tables if they do not exist."""
      self.conn.execute(SCHEMA_SQL)
      logger.info("DuckDB schema initialized at %s", self.db_path)

  def close(self) -> None:
      """Close the database connection."""
      if self._conn is not None:
          self._conn.close()
          self._conn = None

  def execute(self, sql: str, params: list[Any] | None = None) -> duckdb.DuckDBPyConnection:
      """Execute SQL with optional parameters."""
      if params:
          return self.conn.execute(sql, params)
      return self.conn.execute(sql)

  def query_df(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
      """Run a query and return a DataFrame."""
      result = self.execute(sql, params)
      return result.df()

  def upsert_dataframe(
      self,
      table: str,
      df: pd.DataFrame,
      primary_key: list[str],
  ) -> int:
      """
      Insert or replace rows in a table keyed by primary_key columns.

      Uses DELETE + INSERT for simplicity on analytical workloads.
      """
      if df.empty:
          logger.warning("No rows to upsert into %s", table)
          return 0

      temp = f"_tmp_{table}"
      self.conn.register(temp, df)
      cols = ", ".join(df.columns)
      key_clause = " AND ".join(f"{table}.{k} = {temp}.{k}" for k in primary_key)
      self.conn.execute(
          f"DELETE FROM {table} WHERE EXISTS (SELECT 1 FROM {temp} WHERE {key_clause})"
      )
      self.conn.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM {temp}")
      self.conn.unregister(temp)
      logger.info("Upserted %d rows into %s", len(df), table)
      return len(df)

  def insert_dataframe(self, table: str, df: pd.DataFrame) -> int:
      """Bulk insert a DataFrame into a table."""
      if df.empty:
          return 0
      self.conn.register("_insert_df", df)
      cols = ", ".join(df.columns)
      self.conn.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM _insert_df")
      self.conn.unregister("_insert_df")
      logger.info("Inserted %d rows into %s", len(df), table)
      return len(df)

  def count(self, table: str, where: str | None = None) -> int:
      """Return row count for a table with optional WHERE clause."""
      sql = f"SELECT COUNT(*) FROM {table}"
      if where:
          sql += f" WHERE {where}"
      row = self.execute(sql).fetchone()
      assert row is not None
      return int(row[0])

  def phase4_checkpoint(self) -> pd.DataFrame:
      """Phase 4 edge signal summary by market source and edge type."""
      return self.query_df(
          """
          SELECT
              market_source,
              edge_type,
              COUNT(*) AS n_observations,
              AVG(edge) AS mean_edge,
              STDDEV(edge) AS std_edge,
              AVG(ABS(edge)) AS mean_abs_edge
          FROM edge_signals
          GROUP BY market_source, edge_type
          ORDER BY mean_abs_edge DESC
          """
      )

  def phase1_checkpoint(self) -> dict[str, int]:
      """Run Phase 1 validation counts."""
      return {
          "nfl_game_states": self.count("game_states", "sport='NFL'"),
          "nba_game_states": self.count("game_states", "sport='NBA'"),
          "kalshi_contracts": self.count("contracts", "market_source='kalshi'"),
          "polymarket_contracts": self.count("contracts", "market_source='polymarket'"),
          "high_confidence_matches": self.count(
              "game_contract_map", "match_confidence > 0.8"
          ),
          "score_consistency_check": self.count(
              "game_states",
              "score_differential != home_score - away_score",
          ),
      }
