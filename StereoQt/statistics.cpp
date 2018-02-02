#include <algorithm>
#include <iostream>
#include "match.h"

/// Heuristic for selecting parameter 'K'
/// Details are described in K's thesis
float Match::GetK()
{
	int i = dispMax - dispMin + 1;
	int k = (i + 2) / 4;
	if (k < 3)
		k = 3;

	int *array = new int[k];
	std::fill_n(array, k, 0);
	int sum = 0, num = 0;

	int xmin = std::max(0, -dispMin);
	int xmax = std::min(imSizeL.x, imSizeR.x - dispMax);
	Coord p;

	for (p.y = 0; p.y<imSizeL.y && p.y<imSizeR.y; p.y++)
		for (p.x = xmin; p.x<xmax; p.x++) {
			// compute k'th smallest value among data_penalty(p, p+d) for all d
			for (int i = 0, d = dispMin; d <= dispMax; d++) {
				int delta = data_penalty_color(p, p + d);
				if (i<k) array[i++] = delta;
				else for (i = 0; i<k; i++)
					if (delta<array[i])
						std::swap(delta, array[i]);
			}
			sum += *std::max_element(array, array + k);
			num++;
		}

	delete[] array;
	if (num == 0) { std::cerr << "GetK: Not enough samples!" << std::endl; exit(1); }
	if (sum == 0) { std::cerr << "GetK failed: K is 0!" << std::endl; exit(1); }

	float K = ((float)sum) / num;
	std::cout << "Computing statistics: K(data_penalty noise) =" << K << std::endl;
	return K;
}