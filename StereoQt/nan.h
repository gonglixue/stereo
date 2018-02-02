#ifndef NAN_H
#define NAN_H

#include <math.h>

static const float NaN = sqrt(-1.0f);

inline bool is_number(float x) {
	return (x == x);
}

#endif
