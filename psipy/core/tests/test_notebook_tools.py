# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.


from psipy.core.notebook_tools import is_notebook


def test_is_notebook():
    assert not is_notebook()
