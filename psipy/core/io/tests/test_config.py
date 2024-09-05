# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

from configparser import ConfigParser

from psipy.core.io.config import config_to_dict

CONFIG_STR = """\
[SECTION]
astr = strval
anint = 1
afloat = 1.1
"""


def test_type_conversions():
    # ConfigParser, when initialized from dict, can have true ints and floats.
    config = ConfigParser()
    config.read_dict(
        dict(
            SECTION=dict(
                astr="strval",
                anint="1",
                afloat="1.1",
                arealint=2,
                arealfloat=2.2,
            ),
        )
    )
    # config_to_dict retains real ints and floats and coerces string int and
    # floats.
    dct = config_to_dict(config)
    assert dct["SECTION"]["astr"] == "strval"
    assert isinstance(dct["SECTION"]["anint"], int)
    assert dct["SECTION"]["anint"] == 1
    assert isinstance(dct["SECTION"]["afloat"], float)
    assert dct["SECTION"]["afloat"] == 1.1
    assert isinstance(dct["SECTION"]["arealint"], int)
    assert dct["SECTION"]["arealint"] == 2
    assert isinstance(dct["SECTION"]["arealfloat"], float)
    assert dct["SECTION"]["arealfloat"] == 2.2

    # ConfigPrasers, when initialized from string or file, cannot have real ints
    # and floats but only have strings.
    config = ConfigParser()
    config.read_string(CONFIG_STR)
    assert isinstance(config.get("SECTION", "anint"), str)
    assert config.get("SECTION", "anint") == "1"
    assert isinstance(config.get("SECTION", "afloat"), str)
    assert config.get("SECTION", "afloat") == "1.1"
    # Which config_to_dict converts to ints and floats.
    dct = config_to_dict(config)
    assert dct["SECTION"]["astr"] == "strval"
    assert isinstance(dct["SECTION"]["anint"], int)
    assert dct["SECTION"]["anint"] == 1
    assert isinstance(dct["SECTION"]["afloat"], float)
    assert dct["SECTION"]["afloat"] == 1.1
