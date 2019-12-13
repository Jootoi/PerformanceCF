#pragma once
#include<vector>
#include <Eigen/Dense>
#include <Eigen/Core>
#include<vector>
#include<algorithm>
#include"Utilities.h"

namespace model {
template<typename T>
class LatentFactorModel {
private:
	int n;
	int m;
	int factors;
	long data_rows;
	float global_mean;
	std::vector<int>& users;
	std::vector<int>& items;
	std::vector<T>& ratings;
public:
	LatentFactorModel(std::vector<int>& users, std::vector<int>& items, std::vector<T>& ratings);
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> latent_user_matrix;
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> latent_item_matrix;

	Eigen::VectorXf bias_user_vector;
	Eigen::VectorXf bias_item_vector;

	LatentFactorModel* build(int factors);
	LatentFactorModel* iterate(float learning_rate, float reg_term);
	float predict(int user, int item);
};
}
