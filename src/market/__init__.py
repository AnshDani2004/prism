"""Market interface: price alignment, edge detection, adverse selection."""

from src.market.adverse_selection import AdverseSelectionDetector
from src.market.edge import EdgeCalculator
from src.market.interface import ContractResolver

__all__ = ["ContractResolver", "EdgeCalculator", "AdverseSelectionDetector"]
