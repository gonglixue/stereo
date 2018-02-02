#ifndef IMAGE_H
#define IMAGE_H

#include <stdlib.h>
#include <opencv2/opencv.hpp>

typedef enum
{
	IMAGE_GRAY,
	IMAGE_RGB,
	IMAGE_INT,
	IMAGE_FLOAT
}ImageType;

inline cv::Vec3b imgRef(cv::Mat im, int y, int x)
{
	cv::Vec3b value = im.at<cv::Vec3b>(y, x);
	return value;
}

inline int imGetXSize(cv::Mat im)
{
	return im.cols;
}

inline int imGetYSize(cv::Mat im)
{
	return im.rows;
}

struct Coord
{
	int x, y;
	Coord() {}
	Coord(int a, int b) { x = a; y = b; }

	Coord operator+ (Coord a) const { return Coord(x + a.x, y + a.y); }
	Coord operator+ (int a) const { return Coord(x + a, y); }
	Coord operator- (int a) const { return Coord(x - a, y); }

	bool operator< (Coord a) const { return (x < a.x) && (y < a.y); }
	bool operator<= (Coord a) const { return (x <= a.x) && (y <= a.y); }
	bool operator!= (Coord a) const { return (x != a.x) || (y != a.y); }
};

inline cv::Vec3b imgRef(cv::Mat im, Coord p)
{
	return imgRef(im, p.y, p.x);
}

// p是否在矩形r里
inline bool inRect(Coord p, Coord r)
{
	return (Coord(0, 0) <= p && p < r);
}

// rectangle iterator
class RectIterator {
	Coord p;		///< current point
	int w;			///< width of rectangle
public:
	RectIterator(Coord rect) :p(0, 0), w(rect.x) {}
	const Coord& operator*() const { return p; };
	bool operator!= (const RectIterator& it) const { return (p != it.p); }
	RectIterator& operator++() {
		if (++p.x == w) {
			p.x = 0;
			++p.y;
		}
		return *this;
	}

	friend RectIterator rectBegin(Coord rect);
	friend RectIterator rectEnd(Coord rect);
};

inline RectIterator rectBegin(Coord rect) {
	return RectIterator(rect);
}

inline RectIterator rectEnd(Coord rect)
{
	RectIterator it(rect);
	it.p.y = rect.y;
	return it;
}




#endif
