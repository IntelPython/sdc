#ifndef SDC_COMMON_H_
#define SDC_COMMON_H_

#if defined(__GNUC__)
#define __UNUSED__ __attribute__((unused))
#else
#define __UNUSED__
#endif

struct SDC_CTypes
{
    enum SDC_CTypeEnum
    {
        INT8 = 0,
        UINT8 = 1,
        INT32 = 2,
        UINT32 = 3,
        INT64 = 4,
        UINT64 = 7,
        FLOAT32 = 5,
        FLOAT64 = 6,
        INT16 = 8,
        UINT16 = 9,
    };
};

#endif /* SDC_COMMON_H_ */
