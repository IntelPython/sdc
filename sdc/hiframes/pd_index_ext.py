# *****************************************************************************
# Copyright (c) 2020, Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************


from numba import types

_dt_index_data_typ = types.Array(types.NPDatetime('ns'), 1, 'C')
_timedelta_index_data_typ = types.Array(types.NPTimedelta('ns'), 1, 'C')


class DatetimeIndexType(types.IterableType):
    """Temporary type class for DatetimeIndex objects.
    """

    def __init__(self, is_named=False):
        # TODO: support other properties like freq/tz/dtype/yearfirst?
        self.is_named = is_named
        super(DatetimeIndexType, self).__init__(
            name="DatetimeIndex(is_named = {})".format(is_named))

    def copy(self):
        # XXX is copy necessary?
        return DatetimeIndexType(self.is_named)

    @property
    def key(self):
        # needed?
        return self.is_named

    def unify(self, typingctx, other):
        # needed?
        return super(DatetimeIndexType, self).unify(typingctx, other)

    @property
    def iterator_type(self):
        # same as Buffer
        # TODO: fix timestamp
        return types.iterators.ArrayIterator(_dt_index_data_typ)


# similar to DatetimeIndex
class TimedeltaIndexType(types.IterableType):
    """Temporary type class for TimedeltaIndex objects.
    """

    def __init__(self, is_named=False):
        # TODO: support other properties like unit/freq?
        self.is_named = is_named
        super(TimedeltaIndexType, self).__init__(
            name="TimedeltaIndexType(is_named = {})".format(is_named))

    def copy(self):
        # XXX is copy necessary?
        return TimedeltaIndexType(self.is_named)

    @property
    def key(self):
        # needed?
        return self.is_named

    def unify(self, typingctx, other):
        # needed?
        return super(TimedeltaIndexType, self).unify(typingctx, other)

    @property
    def iterator_type(self):
        # same as Buffer
        # TODO: fix timedelta
        return types.iterators.ArrayIterator(_timedelta_index_data_typ)
