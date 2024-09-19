# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Exceptions specific to closed loop control and reinforcement learning.
"""


class PsipyException(Exception):
    pass


class PsipyRLException(PsipyException):
    pass


class NotNotifiedOfEpisodeStart(PsipyRLException):
    pass


class AgentOutsideSpecs(PsipyRLException):
    """Raised when agent is requested to operate outside its design specifications."""

    pass


class PlantShutdown(PsipyRLException):
    """Raised when the plant suddenly shut down."""

    pass
