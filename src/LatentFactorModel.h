#include<vector>
#include <Eigen/Dense>
#include <Eigen/Core>


namespace model {
class LatentFactorModel {
private:
	int n;
	int m;
	int factors;
	long data_rows;
	float global_mean;
	Eigen::Matrix<int, Eigen::Dynamic, 3> data;
public:
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> latent_user_matrix;
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> latent_item_matrix;

	Eigen::VectorXf bias_user_vector;
	Eigen::VectorXf bias_item_vector;

	//Expects the data in user, item, rating format 
	LatentFactorModel* build(const Eigen::Ref<Eigen::MatrixXi>&  data, int factors);
	LatentFactorModel* initialize();
	LatentFactorModel* iterate(float learning_rate, float reg_term);
	float predict(int user, int item);
};
}
