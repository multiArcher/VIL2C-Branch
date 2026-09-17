"""Regression for SMAC statistics during incomplete YMAPPO rollouts."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("smac")
from envs.smac_wrapper import SMACWrapper


def test_stats_before_first_completed_episode():
    wrapper = SMACWrapper.__new__(SMACWrapper)
    upstream = Mock(side_effect=ZeroDivisionError)
    wrapper.env = SimpleNamespace(
        battles_game=0, battles_won=0, timeouts=0, force_restarts=2,
        get_stats=upstream,
    )
    assert wrapper.get_stats() == {
        "battles_game": 0, "battles_won": 0, "battles_draw": 0,
        "win_rate": 0.0, "timeouts": 0, "restarts": 2,
    }
    upstream.assert_not_called()


def test_completed_episode_stats_still_use_smac():
    wrapper = SMACWrapper.__new__(SMACWrapper)
    expected = {"battles_game": 2, "battles_won": 1, "win_rate": 0.5}
    wrapper.env = SimpleNamespace(battles_game=2, get_stats=Mock(return_value=expected))
    assert wrapper.get_stats() == expected
    wrapper.env.get_stats.assert_called_once_with()
