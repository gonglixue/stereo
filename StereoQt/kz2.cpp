#include "match.h"
#include "energy.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cassert>

/// (half of) the neighborhood system
/// the full neighborhood system is edges in NEIGHBORS plus reversed edges
const struct Coord NEIGHBORS[] = { Coord(-1,0), Coord(0,1) };
#define	NEIGHBOR_NUM (sizeof(NEIGHBORS) / sizeof(Coord))

/// compute the data + occlusion penalty (D(a) - K)
int Match::data_occlusion_penalty(Coord p, Coord q) const {
	int D = data_penalty_color(p, q);
	return params.denominator * D - params.K;
}

/// smoothness penalty for assignments(p1, p1+d) and (p2, p2+d)
int Match::smoothness_penalty(Coord p1, Coord p2, int d) const {
	return smoothness_penalty_color(p1, p2, d);
}

/// compute energy
int Match::ComputeEnerty() const {
	int E = 0;

	RectIterator end = rectEnd(imSizeL);
	for (RectIterator p1 = rectBegin(imSizeL); p1 != end; ++p1) {
		int d1 = d_left.at<int>((*p1).y, (*p1).x);
		if (d1 != OCCLUDED)
			E += data_occlusion_penalty(*p1, *p1 + d1);

		for (uint k = 0; k < NEIGHBOR_NUM; k++)
		{
			Coord p2 = *p1 + NEIGHBORS[k];
			if (inRect(p2, imSizeL)) {
				int d2 = d_left.at<int>(p2.y, p2.x);

				if (d1 == d2)
					continue;
				if (d1 != OCCLUDED && inRect(p2 + d1, imSizeR))
					E += smoothness_penalty(*p1, p2, d1);
				if (d2 != OCCLUDED && inRect(*p1 + d2, imSizeR))
					E += smoothness_penalty(*p1, p2, d2);
			}
		}
	}

	return E;
}


/// VAR_ALPHA means disparity alpha before expansion move( in vars0 and varsA)
static const Energy::Var VAR_ALPHA = ((Energy::Var) - 1);
/// VAR_ABSENT means occlusion in vars0, and p+alhpa outside image in varsA
static const Energy::Var VAR_ABSENT = ((Energy::Var) - 2);
/// if the variable has a regular value
inline bool IS_VAR(Energy::Var var) {
	return (var >= 0);
}

/// build nodes for pixel p
/// for assignments in A0: SOURCE means active, SINK means inactive.
/// for assignments in Aalpha: SOURCE means inactive, SINK means active
void Match::build_nodes(Energy& e, Coord p, int a) {
	int d = d_left.at<int>(p.y, p.x);
	Coord q = p + d;
	if (a == d) {	// active assinments (p, p+a) in A^a will remain active
		vars0.at<int>(p.y, p.x) = VAR_ALPHA;
		varsA.at<int>(p.y, p.x) = VAR_ALPHA;
		e.add_constant(data_occlusion_penalty(p, q));
		return;
	}

	vars0.at<int>(p.y, p.x) = (d != OCCLUDED) ?	// (p, p+d) in A^0 can remain active
		e.add_variable(data_occlusion_penalty(p, q), 0) : VAR_ABSENT;

	q = p + a;
	varsA.at<int>(p.y, p.x) = inRect(q, imSizeR) ?
		e.add_variable(0, data_occlusion_penalty(p, q)) : VAR_ABSENT;
}

/// build smoothness term for neighbor pixels p1 and p2 with disparity a.
void Match::build_smoothness(Energy&e, Coord p1, Coord p2, int a)
{
	int d1 = d_left.at<int>(p1.y, p1.x);
	Energy::Var o1 = (Energy::Var)vars0.at<int>(p1.y, p1.x);
	Energy::Var a1 = (Energy::Var)varsA.at<int>(p1.y, p1.x);

	int d2 = d_left.at<int>(p2.y, p2.x);
	Energy::Var o2 = (Energy::Var)vars0.at<int>(p2.y, p2.x);
	Energy::Var a2 = (Energy::Var)varsA.at<int>(p2.y, p2.x);

	// disparity a
	if (a1 != VAR_ABSENT && a2 != VAR_ABSENT)
	{
		int delta = smoothness_penalty(p1, p2, a);
		if (a1 != VAR_ALPHA) {	//(p1, p1+a) is variable
			if (a2 != VAR_ALPHA)
				e.add_term2(a1, a2, 0, delta, delta, 0);
			else // penalize (p1, p1+a) inactive
				e.add_term1(a1, delta, 0);
		}
		else if (a2 != VAR_ALPHA)	//(p1, p1+a) active, (p2, p2+a) variable
			e.add_term1(a2, delta, 0);	// penalize (p2, p2+a) inactive
	}

	// disparity d==nd!=a
	if (d1 == d2 && IS_VAR(o1) && IS_VAR(o2)) {
		assert(a1 != a && d1 != OCCLUDED);
		int delta = smoothness_penalty(p1, p2, d1);
		e.add_term2(o1, o2, 0, delta, delta, 0);	// penalize different activity
	}

	// disparity d1, a!=d1!=d2, (p2, p2+d1) inactive neighbor assignment
	if (d1 != d2 && IS_VAR(o1) && inRect(p2 + d1, imSizeR))
		e.add_term1(o1, smoothness_penalty(p1, p2, d1), 0);

	// diaprity d2, a!=d2!=d1, (p1, p1+d2) inactive neighbor assignment
	if (d2 != d1 && IS_VAR(o2) && inRect(p1 + d2, imSizeR))
		e.add_term1(o2, smoothness_penalty(p1, p2, d2), 0);
}

/// build edges in graph enforcing uniqueness at pixels p and p+d;
/// -prevent (p, p+d) and (p, p+a) from being both active.
/// -prevent (p, p+d) and (p+d-alpha, p+d) from being both active.
void Match::build_uniqueness(Energy& e, Coord p, int alpha)
{
	Energy::Var o = (Energy::Var)vars0.at<int>(p.y, p.x);
	if (!IS_VAR(o))
		return;

	// enfore unique image of p
	Energy::Var a = (Energy::Var)varsA.at<int>(p.y, p.x);
	if (a != VAR_ABSENT)
		e.forbid01(o, a);

	// enforce unique antecedent of p+d
	int d = d_left.at<int>(p.y, p.x);
	assert(d != OCCLUDED);
	p = p + (d - alpha);

	if (inRect(p, imSizeL))
	{
		a = (Energy::Var)varsA.at<int>(p.y, p.x);
		assert(IS_VAR(a));
		e.forbid01(o, a);
	}
}

/// update the disparity map according to min cut of energy
void Match::update_disparity(const Energy& e, int alpha) {
	RectIterator end = rectEnd(imSizeL);
	for (RectIterator p = rectBegin(imSizeL); p != end; ++p)
	{
		Energy::Var o = (Energy::Var)vars0.at<int>((*p).y, (*p).x);
		if (IS_VAR(o) && e.get_var(o) == 1)
			d_left.at<int>((*p).y, (*p).x) = OCCLUDED;
	}

	for (RectIterator p = rectBegin(imSizeL); p != end; ++p)
	{
		Energy::Var a = (Energy::Var)varsA.at<int>((*p).y, (*p).x);
		if (IS_VAR(a) && e.get_var(a) == 1)		// new disparity
			d_left.at<int>((*p).y, (*p).x) = alpha;
	}
}

/// compute the minimum a-expansion configuration
/// return whether the move is different from identity.
/// a: disparity
bool Match::ExpansionMove(int a)
{
	// factors 2 and 12 are minimal ensuring no reallocation
	Energy e(2 * imSizeL.x*imSizeL.y, 12 * imSizeL.x * imSizeL.y);

	// build graph
	RectIterator endL = rectEnd(imSizeL);
	for (RectIterator p = rectBegin(imSizeL); p != endL; ++p)
		build_nodes(e, *p, a);

	for (RectIterator p1 = rectBegin(imSizeL); p1 != endL; ++p1) {
		for (uint k = 0; k < NEIGHBOR_NUM; k++)
		{
			Coord p2 = *p1 + NEIGHBORS[k];
			if (inRect(p2, imSizeL))
				build_smoothness(e, *p1, p2, a);
		}
	}

	for (RectIterator p = rectBegin(imSizeL); p != endL; ++p)
		build_uniqueness(e, *p, a);

	int oldE = E;
	E = e.minimize();

	if (E < oldE)	// accept expansion move
	{
		update_disparity(e, a);
		assert(ComputeEnerty() == E);
		return true;
	}
	return false;
}

/// random permutation
static void generate_permutaion(int *buf, int n)
{
	for (int i = 0; i < n; i++)
		buf[i] = i;
	for (int i = 0; i < n - 1; i++) {
		int j = i + (int)(((double)rand() / RAND_MAX)*(n - i));
		if (j >= n)
			continue;
		std::swap(buf[i], buf[j]);
	}
}

/// main algorithm: a series of alpha-expansions
void Match::run()
{
	std::cout << std::fixed << std::setprecision(1);

	const int dispSize = dispMax - dispMin + 1;
	int* permutation = new int[dispSize];

	E = ComputeEnerty();
	std::cout << "E=" << E << std::endl;

	bool* done = new bool[dispSize]; // can expansion of label decrease energy?
	std::fill_n(done, dispSize, false);
	int nDone = dispSize;		// number of false entries in 'done'

	int step = 0; //?
	for (int iter = 0; iter < params.maxIter && nDone>0; iter++) {
		if (iter == 0 || params.bRandomizeEveryIteration)
			generate_permutaion(permutation, dispSize);

		// 遍历所有label，决定是否expansion
		for (int index = 0; index < dispSize; index++) {
			int label = permutation[index];
			if (done[label])
				continue;
			++step;

			if (ExpansionMove(dispMin + label)) {
				std::fill_n(done, dispSize, false);
				nDone = dispSize;
				std::cout << "*";
			}
			else
				std::cout << "-";
			std::cout << std::flush;
			done[label] = true;
			--nDone;
		}
		std::cout << "E=" << E << std::endl;
	}

	std::cout << (float)step / dispSize << " iteration" << std::endl;

	delete[] permutation;
	delete[] done;
}

void Match::KZ2()
{
	if (params.K < 0 || params.edgeTresh < 0 ||
		params.lambda1 < 0 || params.lambda2 < 0 || params.denominator < 1) {
		std::cerr << "Error in KZ2: wrong parameter!" << std::endl;
		exit(1);
	}

	std::string strDenom;
	if (params.denominator != 1) {
		std::ostringstream s;
		s << params.denominator;
		strDenom = "/" + s.str();
	}

	std::cout << "KZ2:	K=" << params.K << strDenom << std::endl
		<< "	edgeThreshol=" << params.edgeTresh
		<< ", lambda1=" << params.lambda1 << strDenom
		<< ", lambda2=" << params.lambda2 << strDenom
		<< ", dataCosnt = L" <<
		((params.dataCost == Parameters::L1) ? '1' : '2') << std::endl;

	run();
}