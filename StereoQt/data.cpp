#include "match.h"
#include <algorithm>

/// upper bound for intensity level difference when computing data cost
static int CUTOFF = 30;

/// distance from v to interval [min, max]
inline int dist_interval(int v, int min, int max)
{
	if (v < min)
		return (min - v);
	if (v > max)
		return (v - max);

	return 0;
}

/// birchfield-Tomasi color distance between pixel p and q
int Match::data_penalty_color(Coord p, Coord q) const {
	int dSum = 0;
	// Loop over the 3 channels
	for (int i = 0; i < 3; i++) {
		int Ip = imgRef(imColorLeft, p)[i];
		int Iq = imgRef(imColorRight, q)[i];
		int IpMin = imgRef(imColorLeftMin, p)[i];
		int IqMin = imgRef(imColorRightMin, q)[i];
		int IpMax = imgRef(imColorLeftMax, p)[i];
		int IqMax = imgRef(imColorRightMax, q)[i];

		int dp = dist_interval(Ip, IqMin, IqMax);
		int dq = dist_interval(Iq, IpMin, IpMax);
		int d = std::min(dp, dq);
		if (d > CUTOFF)
			d = CUTOFF;
		if (params.dataCost == Parameters::L2)
			d = d*d;
		dSum += d;
	}

	return dSum / 3;
}


/// Fill ImMin and ImMax from Im
static void SubPixelColor(cv::Mat& Im, cv::Mat& ImMin, cv::Mat& ImMax)
{
	int I, I1, I2, I3, I4, IMin, IMax;

	Coord p;
	int xmax = imGetXSize(ImMin), ymax = imGetYSize(ImMin);

	for (p.y = 0; p.y < ymax; p.y++) {
		for (p.x = 0; p.x < xmax; p.x++)
		{
			for (int i = 0; i < 3; i++)	// channel
			{
				I = IMin = IMax = imgRef(Im, p.y, p.x)[i];
				I1 = (p.x > 0 ? (imgRef(Im, p.y, p.x - 1)[i] + I) / 2 : I);		//左邻接像素和该像素的平均
				I2 = (p.x + 1 < xmax ? (imgRef(Im, p.y, p.x + 1)[i] + I) / 2 : I);		//右邻接像素和该像素的平均
				I3 = (p.y > 0 ? (imgRef(Im, p.y - 1, p.x)[i] + I) / 2 : I);		//上
				I4 = (p.y + 1 < ymax ? (imgRef(Im, p.y + 1, p.x)[i] + I) / 2 : I);

				// IMin = 最小邻接像素和该像素的平均
				if (IMin > I1)	IMin = I1;
				if (IMin > I2)	IMin = I2;
				if (IMin > I3)	IMin = I3;
				if (IMin > I4)	IMin = I4;
				if (IMax < I1) IMax = I1;
				if (IMax < I2) IMax = I2;
				if (IMax < I3) IMax = I3;
				if (IMax < I4) IMax = I4;

				ImMin.at<cv::Vec3b>(p.y, p.x)[i] = IMin;
				ImMin.at<cv::Vec3b>(p.y, p.x)[i] = IMax;
			}
		}
	}
}

void Match::InitSubPixel()
{
	imColorLeftMin = cv::Mat(imSizeL.y, imSizeL.x, CV_8UC3);
	imColorLeftMax = cv::Mat(imSizeL.y, imSizeL.x, CV_8UC3);
	imColorRightMin = cv::Mat(imSizeR.y, imSizeR.x, CV_8UC3);
	imColorRightMax = cv::Mat(imSizeR.y, imSizeR.x, CV_8UC3);

	SubPixelColor(imColorLeft, imColorLeftMin, imColorLeftMax);
	SubPixelColor(imColorRight, imColorRightMin, imColorRightMax);
}


/// smoothness panalty between assignements (p1, P1+disp) and (p2, p2+disp)
int Match::smoothness_penalty_color(Coord p1, Coord p2, int disp) const
{
	int d, dMax = 0;
	for (int i = 0; i < 3; i++)
	{
		d = imgRef(imColorLeft, p1)[i] - imgRef(imColorLeft, p2)[i];
		if (d < 0)
			d = -d;
		if (dMax < d)
			dMax = d;

		d = imgRef(imColorRight, p1 + disp)[i] - imgRef(imColorRight, p2 + disp)[i];
		if (d < 0)
			d = -d;
		if (dMax < d)
			dMax = d;
	}

	return (dMax < params.edgeTresh) ? params.lambda1 : params.lambda2;
}

/// set parameters
void Match::SetParameters(Parameters *_params)
{
	params = *_params;
	InitSubPixel();
}
