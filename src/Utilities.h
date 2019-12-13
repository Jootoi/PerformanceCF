#include<vector>
namespace Utilities {

	void reduceToUniques(void* vec);


	template<typename T>
	T sumVectorElements(std::vector<T>& a) {
		T sum = 0;
		#pragma omp parallel for reduction (+:sum)
		for (int i = 0; i < a.size(); i++) {
			sum = sum + a[i];
		}
	}
}

