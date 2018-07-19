from numba import types
from numba.extending import models, register_model

class SeriesType(types.Array):
    """Temporary type class for Series objects.
    """
    def __init__(self, dtype, ndim, layout, readonly=False, name=None,
                 aligned=True):
        # same as types.Array, except name is Series
        assert ndim == 1, "Series() should be one dimensional"
        assert name is None
        if readonly:
            self.mutable = False
        if (not aligned or
            (isinstance(dtype, types.Record) and not dtype.aligned)):
            self.aligned = False
        if name is None:
            type_name = "series"
            if not self.mutable:
                type_name = "readonly " + type_name
            if not self.aligned:
                type_name = "unaligned " + type_name
            name = "%s(%s, %sd, %s)" % (type_name, dtype, ndim, layout)
        super(SeriesType, self).__init__(dtype, ndim, layout, name=name)

    def copy(self, dtype=None, ndim=None, layout=None, readonly=None):
        # same as types.Array, except Series return type
        if dtype is None:
            dtype = self.dtype
        if ndim is None:
            ndim = self.ndim
        if layout is None:
            layout = self.layout
        if readonly is None:
            readonly = not self.mutable
        return SeriesType(dtype=dtype, ndim=ndim, layout=layout, readonly=readonly,
                     aligned=self.aligned)

    def unify(self, typingctx, other):
        # same as types.Array, except returns Series for Series/Series
        # If other is array and the ndim matches
        if isinstance(other, SeriesType) and other.ndim == self.ndim:
            # If dtype matches or other.dtype is undefined (inferred)
            if other.dtype == self.dtype or not other.dtype.is_precise():
                if self.layout == other.layout:
                    layout = self.layout
                else:
                    layout = 'A'
                readonly = not (self.mutable and other.mutable)
                aligned = self.aligned and other.aligned
                return SeriesType(dtype=self.dtype, ndim=self.ndim, layout=layout,
                             readonly=readonly, aligned=aligned)

        # XXX: unify Series/Array as Array
        return super(SeriesType, self).unify(typingctx, other)

register_model(SeriesType)(models.ArrayModel)