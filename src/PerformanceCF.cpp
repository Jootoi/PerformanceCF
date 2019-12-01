// PerformanceCF.cpp : Defines the entry point for the application.
//

#include"LatentFactorModel.h"
#include "PerformanceCF.h"

using namespace std;

int main()
{
	cout << " Eigen version : " << EIGEN_MAJOR_VERSION << " . "
		<< EIGEN_MINOR_VERSION << endl;
	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
	Eigen::Matrix<int, 8, 3> mat = (Eigen::Matrix<int, 8, 3>() <<
		1, 1, 2,
		1, 2, 3,
		1, 5, 5,
		2, 3, 2,
		2, 5, 4,
		3, 1, 2,
		3, 4, 1,
		4, 1, 5).finished();
	
	model::LatentFactorModel* m = new model::LatentFactorModel;
	m = m->build(mat, 2);
	m = m->initialize();
	int n = 100;
	int i = 0;
	while (i < n) {
		m = m->iterate(0.1, 0.5);
		++i;
	}
	

	cout << m->latent_user_matrix.format(CleanFmt);
	cout << m->latent_item_matrix.format(CleanFmt);
	
	cout << m->bias_user_vector.format(CleanFmt);
	cout << m->bias_item_vector.format(CleanFmt);
	return 0;
}
