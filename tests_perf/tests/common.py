from enum import Enum


class Implementation(Enum):
    native = 'native'
    njit = 'njit'
    hpat = 'hpat'
