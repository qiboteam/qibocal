import pytest

from qibocal.auto.runcard import OUTDATED_ACTION_ATTRS, Action, Id
from qibocal.config import logging


@pytest.mark.parametrize("attribute", OUTDATED_ACTION_ATTRS)
def test_outdated_action_parameters(attribute, caplog):
    """Testing warning."""
    caplog.set_level(logging.WARNING)
    test = {"id": "id", attribute: Id("test") if attribute != "priority" else 0}
    action = Action(**test)
