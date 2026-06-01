"""Shared argparse helpers for BGF experiment scripts.

Import and call :func:`add_standard_args` to add the canonical set of
CLI arguments instead of redefining them in every script.

Usage::

    from scripts.common_args import add_standard_args

    parser = argparse.ArgumentParser(description="My experiment")
    add_standard_args(parser, rounds=30, agents=100)
    args = parser.parse_args()
"""

from __future__ import annotations

import argparse


def add_standard_args(
    parser: argparse.ArgumentParser,
    *,
    rounds: int = 10,
    agents: int = 50,
    seeds: str = "1",
) -> None:
    """Add the standard BGF experiment CLI arguments to *parser*.

    Args:
        parser: An existing ArgumentParser to extend.
        rounds: Default number of simulation rounds.
        agents: Default population size.
        seeds: Default seed string (comma-separated, e.g. "1,2,3").
    """
    parser.add_argument(
        "--rounds",
        type=int,
        default=rounds,
        help=f"Number of simulation rounds (default: {rounds}).",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=agents,
        help=f"Population size (default: {agents}).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=seeds,
        help='Comma-separated random seeds (default: "%(default)s").',
    )
    parser.add_argument(
        "--include-llm",
        action="store_true",
        help="Run with LLM policy instead of rule-based proxy.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Root directory for experiment output (default: experiments/).",
    )


def parse_seeds(seeds_str: str) -> list[int]:
    """Parse a comma-separated seed string into a list of ints.

    Args:
        seeds_str: e.g. "1,2,3" or "42".

    Returns:
        List of integer seeds.
    """
    return [int(s.strip()) for s in seeds_str.split(",") if s.strip()]
