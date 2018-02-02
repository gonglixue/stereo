#ifndef MATCH_H
#define MATCH_H

#include "image.h"

class Energy;

class Match {
public:
	Match(cv::Mat left, cv::Mat right, bool color = true);
	Match() {}
	~Match();

	void SetDispRange(int dMin, int dMax);

	/// Parameters
	struct Parameters
	{
		enum { L1, L2 } dataCost;
		int denominator;			///< Data term must be multiplied by denominator

		int edgeTresh;				///< intensity level diff for edge ?
		int lambda1;				///< smoothness cost not across edge
		int lambda2;				///< smoothness cost across edge
		int K;						///< penalty for inactive assignment

		int maxIter;
		bool bRandomizeEveryIteration;		///< random alpha order at each iter
	};
	enum { SAD, NCC, GRAPH } method;

	void InitMatch(cv::Mat left, cv::Mat right);

	float GetK();
	void SetParameters(Parameters *params);
	void KZ2();

	void SaveXLeft(const char* filename);
	void SaveScaledXLeft(const char* filename, bool flag);

private:
	Coord imSizeL, imSizeR;
	int originalHeightL;	///< left image height befor possible crop
	cv::Mat imColorLeft, imColorRight;
	cv::Mat imLeftMin, imLeftMax;	///< range of gray based on neighbors
	cv::Mat imRightMin, imRightMax;
	cv::Mat imColorLeftMin, imColorLeftMax;
	cv::Mat imColorRightMin, imColorRightMax;
	int dispMin, dispMax;

	static const int OCCLUDED;

	cv::Mat d_left;	// CV_32SC1 int32
	Parameters params;
	

	int E;
	cv::Mat vars0;		///< varaibales befor alpha expansion. int32
	cv::Mat varsA;		///< variables after alpha expansion

	void run();
	void InitSubPixel();

	// data penalty functions
	int data_penalty_color(Coord l, Coord r) const;
	// smoothness penalty functions
	int smoothness_penalty_color(Coord p, Coord np, int d) const;

	// KZ algorithm
	int data_occlusion_penalty(Coord l, Coord r) const;
	int smoothness_penalty(Coord p, Coord np, int d) const;
	int ComputeEnerty() const;
	bool ExpansionMove(int a);

	// graph construction
	void build_nodes(Energy& e, Coord p, int a);
	void build_smoothness(Energy& e, Coord p, Coord np, int a);
	void build_uniqueness(Energy& e, Coord p, int a);
	void update_disparity(const Energy& e, int a);
};

#endif
