#include "match.h"
#include <limits>
#include <cmath>
#include <ctime>

static const int MAX_DENOM = 1 << 4;

/// Store in \a params fractions approximating the last 3 parameters.
///
/// They have the same denominator (up to \c MAX_DENOM), chosen so that the sum
/// of relative errors is minimized.
void set_fractions(Match::Parameters& params,
	float K, float lambda1, float lambda2) {
	float minError = std::numeric_limits<float>::max();
	for (int i = 1; i <= MAX_DENOM; i++) {
		float e = 0;
		int numK = 0, num1 = 0, num2 = 0;
		if (K>0)
			e += std::abs((numK = int(i*K + .5f)) / (i*K) - 1.0f);
		if (lambda1>0)
			e += std::abs((num1 = int(i*lambda1 + .5f)) / (i*lambda1) - 1.0f);
		if (lambda2>0)
			e += std::abs((num2 = int(i*lambda2 + .5f)) / (i*lambda2) - 1.0f);
		if (e<minError) {
			minError = e;
			params.denominator = i;
			params.K = numK;
			params.lambda1 = num1;
			params.lambda2 = num2;
		}
	}
}

/// Make sure parameters K, lambda1 and lambda2 are non-negative.
///
/// - K may be computed automatically and lambda set to K/5.
/// - lambda1=3*lambda, lambda2=lambda
/// As the graph requires integer weights, use fractions and common denominator.
void fix_parameters(Match& m, Match::Parameters& params,
	float& K, float& lambda, float& lambda1, float& lambda2) {
	if (K<0) { // Automatic computation of K
		m.SetParameters(&params);
		K = m.GetK();
	}
	if (lambda<0) // Set lambda to K/5
		lambda = K / 5;
	if (lambda1<0) lambda1 = 3 * lambda;
	if (lambda2<0) lambda2 = lambda;
	set_fractions(params, K, lambda1, lambda2);
	m.SetParameters(&params);
}


int main()
{
	Match::Parameters params = {
		Match::Parameters::L2, 1,
		8, -1, -1,
		-1,
		4, false
	};

	float K = -1, lambda = -1, lambda1 = -1, lambda2 = -1;

	cv::Mat im1 = cv::imread("left.png");
	cv::Mat im2 = cv::imread("right.png");

	Match m(im1, im2, true);
	m.SetDispRange(-60, 0);

	time_t seed = time(NULL);
	srand((unsigned int)seed);
	fix_parameters(m, params, K, lambda, lambda1, lambda2);

	m.KZ2();
	m.SaveXLeft("disparity.png");

	im1.release();
	im2.release();
	return 0;
}

