import numpy as np
import pandas as pd
import numba
from numba.extending import overload
from hpat.utils import empty_like_type, alloc_arr_tup

# ported from Spark to Numba-compilable Python
# A port of the Android TimSort class, which utilizes a "stable, adaptive, iterative mergesort."
# https:# github.com/apache/spark/blob/master/core/src/main/java/org/apache/spark/util/collection/TimSort.java
# https:# github.com/python/cpython/blob/master/Objects/listobject.c


# This is the minimum sized sequence that will be merged.  Shorter
# sequences will be lengthened by calling binarySort.  If the entire
# array is less than this length, no merges will be performed.
# # This constant should be a power of two.  It was 64 in Tim Peter's C
# implementation, but 32 was empirically determined to work better in
# this implementation.  In the unlikely event that you set this constant
# to be a number that's not a power of two, you'll need to change the
# minRunLength computation.
# # If you decrease this constant, you must change the stackLen
# computation in the TimSort constructor, or you risk an
# ArrayOutOfBounds exception.  See listsort.txt for a discussion
# of the minimum stack length required as a function of the length
# of the array being sorted and the minimum merge sequence length.

MIN_MERGE = 32


# A stable, adaptive, iterative mergesort that requires far fewer than
# n lg(n) comparisons when running on partially sorted arrays, while
# offering performance comparable to a traditional mergesort when run
# on random arrays.  Like all proper mergesorts, this sort is stable and
# runs O(n log n) time (worst case).  In the worst case, this sort requires
# temporary storage space for n/2 object references, in the best case,
# it requires only a small constant amount of space.
#
# This implementation was adapted from Tim Peters's list sort for
# Python, which is described in detail here:
#
#  http:# svn.python.org/projects/python/trunk/Objects/listsort.txt
#
# Tim's C code may be found here:
#
#  http:# svn.python.org/projects/python/trunk/Objects/listobject.c
#
# The underlying techniques are described in this paper (and may have
# even earlier origins):
#
# "Optimistic Sorting and Information Theoretic Complexity"
# Peter McIlroy
# SODA (Fourth Annual ACM-SIAM Symposium on Discrete Algorithms),
# pp 467-474, Austin, Texas, 25-27 January 1993.
#
# While the API to this class consists solely of static methods, it is
# (privately) instantiable, a TimSort instance holds the state of an ongoing
# sort, assuming the input array is large enough to warrant the full-blown
# TimSort. Small arrays are sorted in place, using a binary insertion sort.

@numba.njit
def sort(sortState, key_arr, lo, hi, data):  # pragma: no cover

    nRemaining  = hi - lo
    if nRemaining < 2:
        return  # Arrays of size 0 and 1 are always sorted

    # If array is small, do a "mini-TimSort" with no merges
    if nRemaining < MIN_MERGE:
        initRunLen = countRunAndMakeAscending(key_arr, lo, hi, data)
        binarySort(key_arr, lo, hi, lo + initRunLen, data)
        return

    # March over the array once, left to right, finding natural runs,
    # extending short natural runs to minRun elements, and merging runs
    # to maintain stack invariant.

    minRun = minRunLength(nRemaining)
    while True:  # emulating do-while
        # Identify next run
        runLen = countRunAndMakeAscending(key_arr, lo, hi, data)

        # If run is short, extend to min(minRun, nRemaining)
        if runLen < minRun:
            force = nRemaining if nRemaining <= minRun else minRun
            binarySort(key_arr, lo, lo + force, lo + runLen, data)
            runLen = force

        # Push run onto pending-run stack, and maybe merge
        sortState.pushRun(lo, runLen)
        sortState.mergeCollapse()

        # Advance to find next run
        lo += runLen
        nRemaining -= runLen
        if nRemaining == 0:
            break

    # Merge all remaining runs to complete sort
    assert lo == hi
    sortState.mergeForceCollapse()
    assert sortState.stackSize == 1



# Sorts the specified portion of the specified array using a binary
# insertion sort.  This is the best method for sorting small numbers
# of elements.  It requires O(n log n) compares, but O(n^2) data
# movement (worst case).
#
# If the initial part of the specified range is already sorted,
# this method can take advantage of it: the method assumes that the
# elements from index {@code lo}, inclusive, to {@code start},
# exclusive are already sorted.
#
# @param a the array in which a range is to be sorted
# @param lo the index of the first element in the range to be sorted
# @param hi the index after the last element in the range to be sorted
# @param start the index of the first element in the range that is
#       not already known to be sorted ({@code lo <= start <= hi})
# @param c comparator to used for the sort

@numba.njit
def binarySort(key_arr, lo, hi, start, data):  # pragma: no cover
    assert lo <= start and start <= hi
    if start == lo:
        start += 1

    # Buffer pivotStore = s.allocate(1)

    while start < hi:
        #pivotStore = key_arr[start]  # TODO: copy data to pivot
        pivot = key_arr[start]
        pivot_data = getitem_arr_tup(data, start)

        # Set left (and right) to the index where key_arr[start] (pivot) belongs
        left = lo
        right = start
        assert left <= right

        # Invariants:
        #  pivot >= all in [lo, left).
        #  pivot <  all in [right, start).

        while left < right:
            mid = (left + right) >> 1
            if pivot < key_arr[mid]:
                right = mid
            else:
                left = mid + 1

        assert left == right

        # The invariants still hold: pivot >= all in [lo, left) and
        # pivot < all in [left, start), so pivot belongs at left.  Note
        # that if there are elements equal to pivot, left points to the
        # first slot after them -- that's why this sort is stable.
        # Slide elements over to make room for pivot.


        n = start - left  # The number of elements to move
        # TODO: optimize for n==1 and n==2
        # TODO: data
        # FIXME: is slicing ok?
        #key_arr[left+1:left+1+n] = key_arr[left:left+n]
        copyRange(key_arr, left, key_arr, left + 1, n)
        copyRange_tup(data, left, data, left + 1, n)

        #copyElement(pivotStore, 0, key_arr, left)
        key_arr[left] = pivot
        setitem_arr_tup(data, left, pivot_data)
        start += 1



# Returns the length of the run beginning at the specified position in
# the specified array and reverses the run if it is descending (ensuring
# that the run will always be ascending when the method returns).
#
# A run is the longest ascending sequence with:
#
#   a[lo] <= a[lo + 1] <= a[lo + 2] <= ...
#
# or the longest descending sequence with:
#
#   a[lo] >  a[lo + 1] >  a[lo + 2] >  ...
#
# For its intended use in a stable mergesort, the strictness of the
# definition of "descending" is needed so that the call can safely
# reverse a descending sequence without violating stability.
#
# @param a the array in which a run is to be counted and possibly reversed
# @param lo index of the first element in the run
# @param hi index after the last element that may be contained in the run.
# It is required that {@code lo < hi}.
# @param c the comparator to used for the sort
# @return  the length of the run beginning at the specified position in
#         the specified array

@numba.njit
def countRunAndMakeAscending(key_arr, lo, hi, data):  # pragma: no cover
    assert lo < hi
    runHi = lo + 1
    if runHi == hi:
        return 1

    # Find end of run, and reverse range if descending
    if key_arr[runHi] < key_arr[lo]:  # Descending
        runHi += 1
        while runHi < hi and key_arr[runHi] < key_arr[runHi-1]:
            runHi += 1
        reverseRange(key_arr, lo, runHi, data)
    else:                     # Ascending
        runHi += 1
        while runHi < hi and key_arr[runHi] >= key_arr[runHi-1]:
            runHi += 1


    return runHi - lo


# Reverse the specified range of the specified array.
# @param a the array in which a range is to be reversed
# @param lo the index of the first element in the range to be reversed
# @param hi the index after the last element in the range to be reversed

@numba.njit
def reverseRange(key_arr, lo, hi, data):  # pragma: no cover
    hi -= 1
    while lo < hi:
        # swap, TODO: copy data
        tmp = key_arr[lo]
        key_arr[lo] = key_arr[hi]
        key_arr[hi] = tmp

        # TODO: add support for map and use it
        swap_arrs(data, lo, hi)
        # for arr in data:
        #     tmp_v = arr[lo]
        #     arr[lo] = arr[hi]
        #     arr[hi] = tmp_v

        lo += 1
        hi -= 1


# Returns the minimum acceptable run length for an array of the specified
# length. Natural runs shorter than this will be extended with
# {@link #binarySort}.
#
# Roughly speaking, the computation is:
#
# If n < MIN_MERGE, return n (it's too small to bother with fancy stuff).
# Else if n is an exact power of 2, return MIN_MERGE/2.
# Else return an k, MIN_MERGE/2 <= k <= MIN_MERGE, such that n/k
#  is close to, but strictly less than, an exact power of 2.
#
# For the rationale, see listsort.txt.
#
# @param n the length of the array to be sorted
# @return the length of the minimum run to be merged


@numba.njit
def minRunLength(n):  # pragma: no cover
    assert n >= 0
    r = 0      # Becomes 1 if any 1 bits are shifted off
    while n >= MIN_MERGE:
        r |= (n & 1)
        n >>= 1

    return n + r

# When we get into galloping mode, we stay there until both runs win less
# often than MIN_GALLOP consecutive times.

MIN_GALLOP = 7

# Maximum initial size of tmp array, which is used for merging.  The array
# can grow to accommodate demand.
#
# Unlike Tim's original C version, we do not allocate this much storage
# when sorting smaller arrays.  This change was required for performance.

INITIAL_TMP_STORAGE_LENGTH = 256


# spec = [
#     ('key_arr', numba.float64[:]),
#     ('aLength', numba.intp),
#     ('minGallop', numba.intp),
#     ('tmpLength', numba.intp),
#     ('tmp', numba.float64[:]),
#     ('stackSize', numba.intp),
#     ('runBase', numba.int64[:]),
#     ('runLen', numba.int64[:]),
# ]

# Creates a TimSort instance to maintain the state of an ongoing sort.
#@numba.jitclass(spec)
class SortState:
    def __init__(self, key_arr, aLength, data):
        self.key_arr = key_arr
        self.data = data
        self.aLength = aLength

        # This controls when we get *into* galloping mode.  It is initialized
        # to MIN_GALLOP.  The mergeLo and mergeHi methods nudge it higher for
        # random data, and lower for highly structured data.
        self.minGallop = MIN_GALLOP

        arr_len = aLength
        # Allocate temp storage (which may be increased later if necessary)
        self.tmpLength = arr_len >> 1 if  arr_len < 2 * INITIAL_TMP_STORAGE_LENGTH else INITIAL_TMP_STORAGE_LENGTH
        self.tmp = empty_like_type(self.tmpLength, self.key_arr)
        self.tmp_data = alloc_arr_tup(self.tmpLength, data)


        # A stack of pending runs yet to be merged.  Run i starts at
        # address base[i] and extends for len[i] elements.  It's always
        # true (so long as the indices are in bounds) that:
        #
        #    runBase[i] + runLen[i] == runBase[i + 1]
        #
        # so we could cut the storage for this, but it's a minor amount,
        # and keeping all the info explicit simplifies the code.

        # Allocate runs-to-be-merged stack (which cannot be expanded).  The
        # stack length requirements are described in listsort.txt.  The C
        # version always uses the same stack length (85), but this was
        # measured to be too expensive when sorting "mid-sized" arrays (e.g.,
        # 100 elements) in Java.  Therefore, we use smaller (but sufficiently
        # large) stack lengths for smaller arrays.  The "magic numbers" in the
        # computation below must be changed if MIN_MERGE is decreased.  See
        # the MIN_MERGE declaration above for more information.

        self.stackSize = 0  # Number of pending runs on stack
        stackLen = 5 if arr_len < 120 else (
                   10 if arr_len < 1542 else (
                   19 if arr_len < 119151 else 40
                   ))
        self.runBase = np.empty(stackLen, np.int64)
        self.runLen = np.empty(stackLen, np.int64)

    # Pushes the specified run onto the pending-run stack.

    # @param runBase index of the first element in the run
    # @param runLen  the number of elements in the run

    def pushRun(self, runBase, runLen):
        self.runBase[self.stackSize] = runBase
        self.runLen[self.stackSize] = runLen
        self.stackSize += 1

    # Examines the stack of runs waiting to be merged and merges adjacent runs
    # until the stack invariants are reestablished:

    #    1. runLen[i - 3] > runLen[i - 2] + runLen[i - 1]
    #    2. runLen[i - 2] > runLen[i - 1]

    # This method is called each time a new run is pushed onto the stack,
    # so the invariants are guaranteed to hold for i < stackSize upon
    # entry to the method.

    def mergeCollapse(self):
        while self.stackSize > 1:
            n = self.stackSize - 2
            if ((n >= 1 and self.runLen[n-1] <= self.runLen[n] + self.runLen[n+1])
                    or (n >= 2 and self.runLen[n-2] <= self.runLen[n] + self.runLen[n-1])):
                if self.runLen[n - 1] < self.runLen[n + 1]:
                    n -= 1
            elif self.runLen[n] > self.runLen[n + 1]:
                break  # Invariant is established

            self.mergeAt(n)


    # Merges all runs on the stack until only one remains.  This method is
    # called once, to complete the sort.

    def mergeForceCollapse(self):
        while self.stackSize > 1:
            n = self.stackSize - 2
            if n > 0 and self.runLen[n-1] < self.runLen[n+1]:
                n -= 1
            self.mergeAt(n)


    # Merges the two runs at stack indices i and i+1.  Run i must be
    # the penultimate or antepenultimate run on the stack.  In other words,
    # i must be equal to stackSize-2 or stackSize-3.

    # @param i stack index of the first of the two runs to merge

    def mergeAt(self, i):
        assert self.stackSize >= 2
        assert i >= 0
        assert i == self.stackSize - 2 or i == self.stackSize - 3

        base1 = self.runBase[i]
        len1 = self.runLen[i]
        base2 = self.runBase[i+1]
        len2 = self.runLen[i+1]
        assert len1 > 0 and len2 > 0
        assert base1 + len1 == base2


        # Record the length of the combined runs. if i is the 3rd-last
        # run now, also slide over the last run (which isn't involved
        # in this merge).  The current run (i+1) goes away in any case.

        self.runLen[i] = len1 + len2
        if i == self.stackSize - 3:
            self.runBase[i+1] = self.runBase[i+2]
            self.runLen[i+1] = self.runLen[i+2]

        self.stackSize -= 1

        # Find where the first element of run2 goes in run1. Prior elements
        # in run1 can be ignored (because they're already in place).

        k = self.gallopRight(self.key_arr[base2], self.key_arr, base1, len1, 0)
        assert k >= 0
        base1 += k
        len1 -= k
        if len1 == 0:
            return


        # Find where the last element of run1 goes in run2. Subsequent elements
        # in run2 can be ignored (because they're already in place).

        len2 = self.gallopLeft(self.key_arr[base1+len1-1], self.key_arr, base2, len2, len2 - 1)
        assert len2 >= 0
        if len2 == 0:
            return

        # Merge remaining runs, using tmp array with min(len1, len2) elements
        if len1 <= len2:
            self.mergeLo(base1, len1, base2, len2)
        else:
            self.mergeHi(base1, len1, base2, len2)


    # Locates the position at which to insert the specified key into the
    # specified sorted range, if the range contains an element equal to key,
    # returns the index of the leftmost equal element.

    # @param key the key whose insertion point to search for
    # @param a the array in which to search
    # @param base the index of the first element in the range
    # @param len the length of the range, must be > 0
    # @param hint the index at which to begin the search, 0 <= hint < n.
    #    The closer hint is to the result, the faster this method will run.
    # @param c the comparator used to order the range, and to search
    # @return the k,  0 <= k <= n such that a[b + k - 1] < key <= a[b + k],
    #   pretending that a[b - 1] is minus infinity and a[b + n] is infinity.
    #   In other words, key belongs at index b + k, or in other words,
    #   the first k elements of a should precede key, and the last n - k
    #   should follow it.

    def gallopLeft(self, key, arr, base, _len, hint):
        assert _len > 0 and hint >= 0 and hint < _len
        lastOfs = 0
        ofs = 1

        if key > arr[base+hint]:
            # Gallop right until a[base+hint+lastOfs] < key <= a[base+hint+ofs]
            maxOfs = _len - hint
            while ofs < maxOfs and key > arr[base+hint+ofs]:
                lastOfs = ofs
                ofs = (ofs << 1) + 1
                if ofs <= 0:   # overflow
                    ofs = maxOfs

            if ofs > maxOfs:
                ofs = maxOfs

            # Make offsets relative to base
            lastOfs += hint
            ofs += hint
        else:  # key <= a[base + hint]
            # Gallop left until a[base+hint-ofs] < key <= a[base+hint-lastOfs]
            maxOfs = hint + 1
            while ofs < maxOfs and key <= arr[base+hint-ofs]:
                lastOfs = ofs
                ofs = (ofs << 1) + 1
                if ofs <= 0:   # overflow
                    ofs = maxOfs

            if ofs > maxOfs:
                ofs = maxOfs

            # Make offsets relative to base
            tmp = lastOfs
            lastOfs = hint - ofs
            ofs = hint - tmp

        assert -1 <= lastOfs and lastOfs < ofs and ofs <= _len


        # Now a[base+lastOfs] < key <= a[base+ofs], so key belongs somewhere
        # to the right of lastOfs but no farther right than ofs.  Do a binary
        # search, with invariant a[base + lastOfs - 1] < key <= a[base + ofs].

        lastOfs += 1
        while lastOfs < ofs:
            m = lastOfs + ((ofs - lastOfs) >> 1)

            if key > arr[base+m]:
                lastOfs = m + 1  # a[base + m] < key
            else:
                ofs = m          # key <= a[base + m]

        assert lastOfs == ofs    # so a[base + ofs - 1] < key <= a[base + ofs]
        return ofs



    # Like gallopLeft, except that if the range contains an element equal to
    # key, gallopRight returns the index after the rightmost equal element.

    # @param key the key whose insertion point to search for
    # @param a the array in which to search
    # @param base the index of the first element in the range
    # @param len the length of the range must be > 0
    # @param hint the index at which to begin the search, 0 <= hint < n.
    #    The closer hint is to the result, the faster this method will run.
    # @param c the comparator used to order the range, and to search
    # @return the k,  0 <= k <= n such that a[b + k - 1] <= key < a[b + k]

    def gallopRight(self, key, arr, base, _len, hint):
        assert _len > 0 and hint >= 0 and hint < _len

        ofs = 1
        lastOfs = 0

        if key < arr[base + hint]:
            # Gallop left until a[b+hint - ofs] <= key < a[b+hint - lastOfs]
            maxOfs = hint + 1
            while ofs < maxOfs and key < arr[base + hint - ofs]:
                lastOfs = ofs
                ofs = (ofs << 1) + 1
                if ofs <= 0:   # overflow
                    ofs = maxOfs

            if ofs > maxOfs:
                ofs = maxOfs

            # Make offsets relative to b
            tmp = lastOfs
            lastOfs = hint - ofs
            ofs = hint - tmp
        else:  #  a[b + hint] <= key
            # Gallop right until a[b+hint + lastOfs] <= key < a[b+hint + ofs]
            maxOfs = _len - hint
            while ofs < maxOfs and key >= arr[base+hint+ofs]:
                lastOfs = ofs
                ofs = (ofs << 1) + 1
                if ofs <= 0:   # overflow
                    ofs = maxOfs

            if ofs > maxOfs:
                ofs = maxOfs

            # Make offsets relative to b
            lastOfs += hint
            ofs += hint

        assert -1 <= lastOfs and lastOfs < ofs and ofs <= _len


        # Now a[b + lastOfs] <= key < a[b + ofs], so key belongs somewhere to
        # the right of lastOfs but no farther right than ofs.  Do a binary
        # search, with invariant a[b + lastOfs - 1] <= key < a[b + ofs].

        lastOfs += 1
        while lastOfs < ofs:
            m = lastOfs + ((ofs - lastOfs) >> 1)

            if key < arr[base + m]:
                ofs = m          # key < a[b + m]
            else:
                lastOfs = m + 1  # a[b + m] <= key

        assert lastOfs == ofs    # so a[b + ofs - 1] <= key < a[b + ofs]
        return ofs

    # Merges two adjacent runs in place, in a stable fashion.  The first
    # element of the first run must be greater than the first element of the
    # second run (a[base1] > a[base2]), and the last element of the first run
    # (a[base1 + len1-1]) must be greater than all elements of the second run.

    # For performance, this method should be called only when len1 <= len2
    # its twin, mergeHi should be called if len1 >= len2.  (Either method
    # may be called if len1 == len2.)

    # @param base1 index of first element in first run to be merged
    # @param len1  length of first run to be merged (must be > 0)
    # @param base2 index of first element in second run to be merged
    #       (must be aBase + aLen)
    # @param len2  length of second run to be merged (must be > 0)

    def mergeLo(self, base1, len1, base2, len2):
        assert len1 > 0 and len2 > 0 and base1 + len1 == base2

        # Copy first run into temp array
        arr = self.key_arr
        arr_data = self.data
        tmp = self.ensureCapacity(len1)
        tmp_data = self.tmp_data
        copyRange(arr, base1, tmp, 0, len1)
        #tmp[:len1] = arr[base1:base1+len1]
        copyRange_tup(arr_data, base1, tmp_data, 0, len1)

        cursor1 = 0       # Indexes into tmp array
        cursor2 = base2   # Indexes a
        dest = base1      # Indexes a

        # Move first element of second run and deal with degenerate cases
        # copyElement(arr, cursor2, arr, dest)
        arr[dest] = arr[cursor2]
        copyElement_tup(arr_data, cursor2, arr_data, dest)

        cursor2 += 1
        dest += 1
        len2 -= 1
        if len2 == 0:
            copyRange(tmp, cursor1, arr, dest, len1)
            copyRange_tup(tmp_data, cursor1, arr_data, dest, len1)
            #arr[dest:dest+len1] = tmp[cursor1:cursor1+len1]
            return

        if len1 == 1:
            copyRange(arr, cursor2, arr, dest, len2)
            copyRange_tup(arr_data, cursor2, arr_data, dest, len2)
            copyElement(tmp, cursor1, arr, dest + len2) # Last elt of run 1 to end of merge
            copyElement_tup(tmp_data, cursor1, arr_data, dest + len2)
            return


        minGallop = self.minGallop

        # XXX *************** refactored nested break into func
        len1, len2, cursor1, cursor2, dest, minGallop = self.mergeLo_inner(
            len1, len2, tmp, cursor1, cursor2, dest, minGallop)
        # XXX *****************


        self.minGallop = 1 if minGallop < 1 else minGallop  # Write back to field

        if len1 == 1:
            assert len2 > 0
            copyRange(arr, cursor2, arr, dest, len2)
            copyRange_tup(arr_data, cursor2, arr_data, dest, len2)
            copyElement(tmp, cursor1, arr, dest + len2) #  Last elt of run 1 to end of merge
            copyElement_tup(tmp_data, cursor1, arr_data, dest + len2)
        elif len1 == 0:
            raise ValueError("Comparison method violates its general contract!")
        else:
            assert len2 == 0
            assert len1 > 1
            copyRange(tmp, cursor1, arr, dest, len1)
            copyRange_tup(tmp_data, cursor1, arr_data, dest, len1)


    def mergeLo_inner(self, len1, len2, tmp, cursor1, cursor2, dest, minGallop):
        arr = self.key_arr
        arr_data = self.data
        tmp_data = self.tmp_data

        while True:
            count1 = 0 # Number of times in a row that first run won
            count2 = 0 # Number of times in a row that second run won


            # Do the straightforward thing until (if ever) one run starts
            # winning consistently.

            while True:
                assert len1 > 1 and len2 > 0
                if arr[cursor2] < tmp[cursor1]:
                    copyElement(arr, cursor2, arr, dest)
                    copyElement_tup(arr_data, cursor2, arr_data, dest)
                    cursor2 += 1
                    dest += 1
                    count2 += 1
                    count1 = 0
                    len2 -= 1
                    if len2 == 0:
                        return len1, len2, cursor1, cursor2, dest, minGallop
                else:
                    copyElement(tmp, cursor1, arr, dest)
                    copyElement_tup(tmp_data, cursor1, arr_data, dest)
                    cursor1 += 1
                    dest += 1
                    count1 += 1
                    count2 = 0
                    len1 -= 1
                    if len1 == 1:
                        return len1, len2, cursor1, cursor2, dest, minGallop

                if not ((count1 | count2) < minGallop):
                    break


            # One run is winning so consistently that galloping may be a
            # huge win. So try that, and continue galloping until (if ever)
            # neither run appears to be winning consistently anymore.

            while True:
                assert len1 > 1 and len2 > 0
                count1 = self.gallopRight(arr[cursor2], tmp, cursor1, len1, 0)
                if count1 != 0:
                    copyRange(tmp, cursor1, arr, dest, count1)
                    copyRange_tup(tmp_data, cursor1, arr_data, dest, count1)
                    dest += count1
                    cursor1 += count1
                    len1 -= count1
                    if len1 <= 1: # len1 == 1 or len1 == 0
                        return len1, len2, cursor1, cursor2, dest, minGallop

                copyElement(arr, cursor2, arr, dest)
                copyElement_tup(arr_data, cursor2, arr_data, dest)
                cursor2 += 1
                dest += 1
                len2 -= 1
                if len2 == 0:
                    return len1, len2, cursor1, cursor2, dest, minGallop

                count2 = self.gallopLeft(tmp[cursor1], arr, cursor2, len2, 0)
                if count2 != 0:
                    copyRange(arr, cursor2, arr, dest, count2)
                    copyRange_tup(arr_data, cursor2, arr_data, dest, count2)
                    dest += count2
                    cursor2 += count2
                    len2 -= count2
                    if len2 == 0:
                        return len1, len2, cursor1, cursor2, dest, minGallop

                copyElement(tmp, cursor1, arr, dest)
                copyElement_tup(tmp_data, cursor1, arr_data, dest)
                cursor1 += 1
                dest += 1
                len1 -= 1
                if len1 == 1:
                    return len1, len2, cursor1, cursor2, dest, minGallop
                minGallop -= 1

                if not (count1 >= MIN_GALLOP | count2 >= MIN_GALLOP):
                    break

            if minGallop < 0:
                minGallop = 0

            minGallop += 2  # Penalize for leaving gallop mode

        return len1, len2, cursor1, cursor2, dest, minGallop



    # Like mergeLo, except that this method should be called only if
    # len1 >= len2 mergeLo should be called if len1 <= len2.  (Either method
    # may be called if len1 == len2.)

    # @param base1 index of first element in first run to be merged
    # @param len1  length of first run to be merged (must be > 0)
    # @param base2 index of first element in second run to be merged
    #       (must be aBase + aLen)
    # @param len2  length of second run to be merged (must be > 0)

    def mergeHi(self, base1, len1, base2, len2):
        assert len1 > 0 and len2 > 0 and base1 + len1 == base2

        # Copy second run into temp array
        arr = self.key_arr
        arr_data = self.data
        tmp = self.ensureCapacity(len2)
        tmp_data = self.tmp_data
        copyRange(arr, base2, tmp, 0, len2)
        copyRange_tup(arr_data, base2, tmp_data, 0, len2)

        cursor1 = base1 + len1 - 1  # Indexes into arr
        cursor2 = len2 - 1          # Indexes into tmp array
        dest = base2 + len2 - 1     # Indexes into arr

        # Move last element of first run and deal with degenerate cases
        copyElement(arr, cursor1, arr, dest)
        copyElement_tup(arr_data, cursor1, arr_data, dest)
        cursor1 -= 1
        dest -= 1
        len1 -= 1
        if len1 == 0:
            copyRange(tmp, 0, arr, dest - (len2 - 1), len2)
            copyRange_tup(tmp_data, 0, arr_data, dest - (len2 - 1), len2)
            return

        if len2 == 1:
            dest -= len1
            cursor1 -= len1
            copyRange(arr, cursor1 + 1, arr, dest + 1, len1)
            copyRange_tup(arr_data, cursor1 + 1, arr_data, dest + 1, len1)
            copyElement(tmp, cursor2, arr, dest)
            copyElement_tup(tmp_data, cursor2, arr_data, dest)
            return

        minGallop = self.minGallop

        # XXX *************** refactored nested break into func
        len1, len2, tmp, cursor1, cursor2, dest, minGallop = self.mergeHi_inner(
            base1, len1, len2, tmp, cursor1, cursor2, dest, minGallop)
        # XXX *****************

        self.minGallop = 1 if minGallop < 1 else minGallop  # Write back to field

        if len2 == 1:
            assert len1 > 0
            dest -= len1
            cursor1 -= len1
            copyRange(arr, cursor1 + 1, arr, dest + 1, len1)
            copyRange_tup(arr_data, cursor1 + 1, arr_data, dest + 1, len1)
            copyElement(tmp, cursor2, arr, dest) # Move first elt of run2 to front of merge
            copyElement_tup(tmp_data, cursor2, arr_data, dest)
        elif len2 == 0:
            raise ValueError("Comparison method violates its general contract!")
        else:
            assert len1 == 0
            assert len2 > 0
            copyRange(tmp, 0, arr, dest - (len2 - 1), len2)
            copyRange_tup(tmp_data, 0, arr_data, dest - (len2 - 1), len2)


    # XXX refactored nested loop break
    def mergeHi_inner(self, base1, len1, len2, tmp, cursor1, cursor2, dest, minGallop):
        arr = self.key_arr
        arr_data = self.data
        tmp_data = self.tmp_data

        while True:
            count1 = 0 # Number of times in a row that first run won
            count2 = 0 # Number of times in a row that second run won

            # Do the straightforward thing until (if ever) one run
            # appears to win consistently.

            while True:
                assert len1 > 0 and len2 > 1
                if tmp[cursor2] < arr[cursor1]:
                    copyElement(arr, cursor1, arr, dest)
                    copyElement_tup(arr_data, cursor1, arr_data, dest)
                    cursor1 -= 1
                    dest -= 1
                    count1 += 1
                    count2 = 0
                    len1 -= 1
                    if len1 == 0:
                        return len1, len2, tmp, cursor1, cursor2, dest, minGallop
                else:
                    copyElement(tmp, cursor2, arr, dest)
                    copyElement_tup(tmp_data, cursor2, arr_data, dest)
                    cursor2 -=1
                    dest -= 1
                    count2 += 1
                    count1 = 0
                    len2 -= 1
                    if len2 == 1:
                        return len1, len2, tmp, cursor1, cursor2, dest, minGallop

                if not ((count1 | count2) < minGallop):
                    break


            # One run is winning so consistently that galloping may be a
            # huge win. So try that, and continue galloping until (if ever)
            # neither run appears to be winning consistently anymore.

            while True:
                assert len1 > 0 and len2 > 1
                count1 = len1 - self.gallopRight(tmp[cursor2], arr, base1, len1, len1 - 1)
                if count1 != 0:
                    dest -= count1
                    cursor1 -= count1
                    len1 -= count1
                    copyRange(arr, cursor1 + 1, arr, dest + 1, count1)
                    copyRange_tup(arr_data, cursor1 + 1, arr_data, dest + 1, count1)
                    if len1 == 0:
                        return len1, len2, tmp, cursor1, cursor2, dest, minGallop

                copyElement(tmp, cursor2, arr, dest)
                copyElement_tup(tmp_data, cursor2, arr_data, dest)
                cursor2 -= 1
                dest -= 1
                len2 -= 1
                if len2 == 1:
                    return len1, len2, tmp, cursor1, cursor2, dest, minGallop

                count2 = len2 - self.gallopLeft(arr[cursor1], tmp, 0, len2, len2 - 1)
                if count2 != 0:
                    dest -= count2
                    cursor2 -= count2
                    len2 -= count2
                    copyRange(tmp, cursor2 + 1, arr, dest + 1, count2)
                    copyRange_tup(tmp_data, cursor2 + 1, arr_data, dest + 1, count2)
                    if len2 <= 1:  # len2 == 1 or len2 == 0
                        return len1, len2, tmp, cursor1, cursor2, dest, minGallop

                copyElement(arr, cursor1, arr, dest)
                copyElement_tup(arr_data, cursor1, arr_data, dest)
                cursor1 -= 1
                dest -= 1
                len1 -= 1
                if len1 == 0:
                    return len1, len2, tmp, cursor1, cursor2, dest, minGallop
                minGallop -= 1
                if not (count1 >= MIN_GALLOP | count2 >= MIN_GALLOP):
                    break

            if minGallop < 0:
                minGallop = 0
            minGallop += 2  # Penalize for leaving gallop mode

        return len1, len2, tmp, cursor1, cursor2, dest, minGallop


    # Ensures that the external array tmp has at least the specified
    # number of elements, increasing its size if necessary.  The size
    # increases exponentially to ensure amortized linear time complexity.

    # @param minCapacity the minimum required capacity of the tmp array
    # @return tmp, whether or not it grew

    def ensureCapacity(self, minCapacity):
        if self.tmpLength < minCapacity:
            # Compute smallest power of 2 > minCapacity
            newSize = minCapacity
            newSize |= newSize >> 1
            newSize |= newSize >> 2
            newSize |= newSize >> 4
            newSize |= newSize >> 8
            newSize |= newSize >> 16
            newSize += 1

            if newSize < 0:  # Not bloody likely!
                newSize = minCapacity
            else:
                newSize = min(newSize, self.aLength >> 1)

            self.tmp = empty_like_type(newSize, self.key_arr)
            self.tmp_data = alloc_arr_tup(newSize, self.data)
            self.tmpLength = newSize

        return self.tmp


################### Utils #############

def swap_arrs(data, lo, hi):
    for arr in data:
        tmp_v = arr[lo]
        arr[lo] = arr[hi]
        arr[hi] = tmp_v

@overload(swap_arrs)
def swap_arrs_overload(arr_tup_t, l_typ, h_typ):
    count = arr_tup_t.count

    func_text = "def f(arr_tup, lo, hi):\n"
    for i in range(count):
        func_text += "  tmp_v_{} = arr_tup[{}][lo]\n".format(i, i)
        func_text += "  arr_tup[{}][lo] = arr_tup[{}][hi]\n".format(i, i)
        func_text += "  arr_tup[{}][hi] = tmp_v_{}\n".format(i, i)
    func_text += "  return\n"

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    swap_impl = loc_vars['f']
    return swap_impl


@numba.njit
def copyRange(src_arr, src_pos, dst_arr, dst_pos, n):  # pragma: no cover
    dst_arr[dst_pos:dst_pos+n] = src_arr[src_pos:src_pos+n]

def copyRange_tup(src_arr_tup, src_pos, dst_arr_tup, dst_pos, n):  # pragma: no cover
    for src_arr, dst_arr in zip(src_arr_tup, dst_arr_tup):
        dst_arr[dst_pos:dst_pos+n] = src_arr[src_pos:src_pos+n]

@overload(copyRange_tup)
def copyRange_tup_overload(src_arr_tup_t, src_pos_t, dst_arr_tup_t, dst_pos_t, n_t):
    count = src_arr_tup_t.count
    assert count == dst_arr_tup_t.count

    func_text = "def f(src_arr_tup, src_pos, dst_arr_tup, dst_pos, n):\n"
    for i in range(count):
        func_text += "  copyRange(src_arr_tup[{}], src_pos, dst_arr_tup[{}], dst_pos, n)\n".format(i, i)
    func_text += "  return\n"

    loc_vars = {}
    exec(func_text, {'copyRange': copyRange}, loc_vars)
    copy_impl = loc_vars['f']
    return copy_impl

@numba.njit
def copyElement(src_arr, src_pos, dst_arr, dst_pos):  # pragma: no cover
    dst_arr[dst_pos] = src_arr[src_pos]

def copyElement_tup(src_arr_tup, src_pos, dst_arr_tup, dst_pos):  # pragma: no cover
    for src_arr, dst_arr in zip(src_arr_tup, dst_arr_tup):
        dst_arr[dst_pos] = src_arr[src_pos]

@overload(copyElement_tup)
def copyElement_tup_overload(src_arr_tup_t, src_pos_t, dst_arr_tup_t, dst_pos_t):
    count = src_arr_tup_t.count
    assert count == dst_arr_tup_t.count

    func_text = "def f(src_arr_tup, src_pos, dst_arr_tup, dst_pos):\n"
    for i in range(count):
        func_text += "  copyElement(src_arr_tup[{}], src_pos, dst_arr_tup[{}], dst_pos)\n".format(i, i)
    func_text += "  return\n"

    loc_vars = {}
    exec(func_text, {'copyElement': copyElement}, loc_vars)
    copy_impl = loc_vars['f']
    return copy_impl

def getitem_arr_tup(arr_tup, ind):  # pragma: no cover
    l = [arr[ind] for arr in arr_tup]
    return tuple(l)

@overload(getitem_arr_tup)
def getitem_arr_tup_overload(arr_tup_t, ind_t):
    count = arr_tup_t.count

    func_text = "def f(arr_tup, ind):\n"
    func_text += "  return ({}{})\n".format(
        ','.join(["arr_tup[{}][ind]".format(i) for i in range(count)]),
        "," if count == 1 else "")  # single value needs comma to become tuple

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars['f']
    return impl

def setitem_arr_tup(arr_tup, ind, val_tup):  # pragma: no cover
    for arr, val in zip(arr_tup, val_tup):
        arr[ind] = val

@overload(setitem_arr_tup)
def setitem_arr_tup_overload(arr_tup_t, ind_t, val_tup_t):
    count = arr_tup_t.count

    func_text = "def f(arr_tup, ind, val_tup):\n"
    for i in range(count):
        func_text += "  arr_tup[{}][ind] = val_tup[{}]\n".format(i, i)
    func_text += "  return\n"

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars['f']
    return impl


def test():  # pragma: no cover
    import time
    #SortStateCL = SortState #numba.jitclass(spec)(SortState)
    # warm up
    t1 = time.time()
    T = np.ones(3)
    data = (np.arange(3), np.ones(3),)
    spec = [
    ('key_arr', numba.float64[:]),
    ('aLength', numba.intp),
    ('minGallop', numba.intp),
    ('tmpLength', numba.intp),
    ('tmp', numba.float64[:]),
    ('stackSize', numba.intp),
    ('runBase', numba.int64[:]),
    ('runLen', numba.int64[:]),
    ('data', numba.typeof(data)),
    ('tmp_data', numba.typeof(data)),
    ]
    SortStateCL = numba.jitclass(spec)(SortState)
    sortState = SortStateCL(T, 3, data)
    sort(sortState, T, 0, 3, data)
    print("compile time", time.time()-t1)
    n = 210000
    np.random.seed(2)
    data = (np.arange(n), np.random.ranf(n))
    A = np.random.ranf(n)
    df = pd.DataFrame({'A': A, 'B': data[0], 'C': data[1]})
    t1 = time.time()
    #B = np.sort(A)
    df2 = df.sort_values('A', inplace=False)
    t2 = time.time()
    sortState = SortStateCL(A, n, data)
    sort(sortState, A, 0, n, data)
    print("HPAT", time.time()-t2, "Numpy", t2-t1)
    # print(df2.B)
    # print(data)
    np.testing.assert_almost_equal(data[0], df2.B.values)
    np.testing.assert_almost_equal(data[1], df2.C.values)

if __name__ == '__main__':
    test()
