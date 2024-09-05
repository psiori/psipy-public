# PSIORI Reinforcement Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Snap7 communication with a Siemens PLC.

Snap7 installation: ``pip install python-snap7==0.10``

- macOS: ``brew install snap7``
- ubuntu: TODO @johnberroa

`snap7` is used to communicate with a Siemens PLC with python code.
Interaction with the PLC involves reading and writing to memory.  When doing
either, snap7 needs to know the following parameters:

Area: String version of the relevant PLC area code
Address: The integer address, e.g. Q0.1 := 0
Bit at address: The integer bit location within the address, e.g. Q0.1 := 1
Datatype: Snap7 datatype, which is a wrapper for read/write block sizes

The following information is highly useful:
- http://snap7.sourceforge.net/sharp7.html
- https://github.com/gijzelaerr/python-snap7/blob/master/snap7/util.py
- http://simplyautomationized.blogspot.com/2016/02/
  python-snap7-s7-1200-simple-example.html

TODO @johnberroa: reStructured text

"""

import logging
from typing import Any, Dict, List, Union

import snap7
from snap7.util import get_bool, get_dword, get_int, get_real
from snap7.util import set_bool, set_dword, set_int, set_real

LOG = logging.getLogger(__name__)

# The following is copy-pasted from internal snap7 code.
# This must be done in order to get pytest to pass because
# using snap7's "ADict" causes pytest to crash.
# https://github.com/gijzelaerr/python-snap7/blob/
# f645608140825c5903b3f16d225d52fd50cfe0b2/snap7/snap7types.py#L62-L78

# Area codes:
# PE:= 0x81 Process Inputs
# PA:= 0x82, Process Outputs
# MK:= 0x83, Merkers
# DB:= 0x84: DB
# CT:= 0x1C: Counters
# TM:= 0x1D: Timers

# -- start copy --

areas = {
    "PE": 0x81,
    "PA": 0x82,
    "MK": 0x83,
    "DB": 0x84,
    "CT": 0x1C,
    "TM": 0x1D,
}

# Word Length
S7WLBit = 0x01
S7WLByte = 0x02
S7WLWord = 0x04
S7WLDWord = 0x06
S7WLReal = 0x08
S7WLCounter = 0x1C
S7WLTimer = 0x1D

# -- end copy --

# For your information:
# PLC_TYPES = Union[S7WLBit, S7WLByte, S7WLWord, S7WLReal, S7WLDWord]
TYPE_BYTEARRAYS = {
    S7WLBit: bytearray(b"\x00"),
    S7WLByte: bytearray(b"\x00\x00"),
    S7WLWord: bytearray(b"\x00\x00\x00\x00"),
    S7WLReal: bytearray(b"\x00\x00\x00\x00\x00\x00\x00\x00"),
    S7WLDWord: bytearray(b"\x00\x00\x00\x00\x00\x00"),
}


def read_memory(plc, area: str, address: int, bit_at_address: int, datatype: Any):
    """Read memory at address x bit_at_address

    Args:
        plc: snap7 Client
        area: str version of the area on the PLC (from area codes)
        address: the int representation of the address, e.g. %Q0.1, address == 0
        bit_at_address: the int representation of the position at the address,
                        e.g. %Q0.1, bit==1
        datatype: datatype from snap7, determines the size of the bytes to write

    Returns:
        What is written in memory, converted to datatype

    """
    result = plc.read_area(area=areas[area], dbnumber=1, start=address, size=datatype)
    if datatype == S7WLBit:
        return get_bool(result, 0, bit_at_address)
    elif datatype == S7WLByte or datatype == S7WLWord:
        return get_int(result, 0)
    elif datatype == S7WLReal:
        return get_real(result, 0)
    elif datatype == S7WLDWord:
        return get_dword(result, 0)
    else:
        return None


def write_memory(
    plc, area: str, address: int, bit_at_address: int, datatype: Any, value: Any
):
    """Write memory at address x bit_at_address

    Args:
        plc: snap7 Client
        area: str version of the area on the PLC (from area codes)
        address: the int representation of the address, e.g. %Q0.1, address == 0
        bit_at_address: the int representation of the position at the address,
                        e.g. %Q0.1, bit==1
        datatype: datatype from snap7, determines the size of the bytes to write
        value: the value to write
    """
    if datatype != S7WLBit:
        # Need to read-modify-write, so we get the reference by reading -- slower!
        result = plc.read_area(areas[area], 1, address, datatype)
    else:
        result = TYPE_BYTEARRAYS[datatype]
    if datatype == S7WLBit:
        set_bool(result, 0, bit_at_address, value)
    elif datatype == S7WLByte or datatype == S7WLWord:
        set_int(result, 0, value)
    elif datatype == S7WLReal:
        set_real(result, 0, value)
    elif datatype == S7WLDWord:
        set_dword(result, 0, value)
    plc.write_area(areas[area], 1, address, result)


class DBReader:
    """Read a datablock off of a PLC with a specific datablock definition.

    Since reading each individual key requires a whole read cycletime, by placing
    all relevant variables in a datablock allows us to read them all at once.
    However, the datablock seems to have a certain size, and only up to 10.0
    bits (unconfirmed) can be read from the DB. Any combination of data sizes
    can make up these 10.0 bits (i.e. 5 bools, 2 reals and 1 int, etc.)

    Currently reads:
        DBCart - raw cart position encoder values (INACCURATE! can't find proper dtype)
        DBPole - raw pole angle
        DBLeftStop - bool for the left stop trigger; true = not on
        DBPos - converted position values
    """

    offsets = {"Bool": 2, "Int": 2, "Real": 4, "DInt": 6, "String": 256}
    datablock = """
DBPos\tInt\t0.0
DBPole\tInt\t2.0
DBLeftStop\tBool\t4.0
"""

    def __init__(self, plc_address: str, datablock_num: int):
        self.plc = snap7.client.Client()
        self.plc.connect(plc_address, 0, 1)
        self.db_num = datablock_num

    def __len__(self):
        return self.length

    def read(self) -> Dict[str, Union[Any]]:
        data = self.plc.read_area(areas["DB"], self.db_num, 0, self.length)
        db = {}
        for item in self.items:
            value = None
            offset = int(item["bytebit"].split(".")[0])

            if item["datatype"] == "Real":
                value = get_real(data, offset)

            if item["datatype"] == "Bool":
                bit = int(item["bytebit"].split(".")[1])
                value = get_bool(data, offset, bit)

            if item["datatype"] == "Int":
                value = get_int(data, offset)

            if item["datatype"] == "DWord":
                value = get_dword(data, offset)

            db[item["name"]] = value

        return db

    def get_db_size(self, array, bytekey, datatypekey) -> int:
        seq, length = [x[bytekey] for x in array], [x[datatypekey] for x in array]
        idx = seq.index(max(seq))
        lastByte = int(max(seq).split(".")[0]) + (self.offsets[length[idx]])
        return lastByte

    @property
    def items(self) -> List[Dict[str, str]]:
        itemlist = filter(lambda a: a != "", self.datablock.split("\n"))
        deliminator = "\t"
        return [
            {
                "name": x.split(deliminator)[0],
                "datatype": x.split(deliminator)[1],
                "bytebit": x.split(deliminator)[2],
            }
            for x in itemlist
        ]

    @property
    def length(self) -> int:
        return self.get_db_size(self.items, "bytebit", "datatype")


if __name__ == "__main__":
    try:
        import time

        time.time()
        # The below will turn on the first output light on the PLC
        plc = snap7.client.Client()
        print("Connecting...")
        plc.connect("192.168.0.1", 0, 1)
        d = DBReader("192.168.0.1", 9)
        while True:
            o = d.read()
            print(o["DBCart"])

    except KeyboardInterrupt:
        print("PLC disconnected.")
        plc.disconnect()
