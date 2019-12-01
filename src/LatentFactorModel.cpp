#include"LatentFactorModel.h"

model::LatentFactorModel* model::LatentFactorModel::build(const Eigen::Ref<Eigen::MatrixXi>& data, int factors) {
	this->factors = factors;
	this->m = data.col(0).maxCoeff()+1;
	this->n = data.col(1).maxCoeff()+1;
	this->data = data;
	this->data_rows = data.rows();
	this->global_mean = data.col(2).sum() / this->data_rows;

	return(this);
}

model::LatentFactorModel* model::LatentFactorModel::initialize() {
	this->latent_user_matrix = Eigen::MatrixXf::Ones(this->m, this->factors);
	this->latent_item_matrix = Eigen::MatrixXf::Ones(this->n, this->factors);

	this->bias_user_vector = Eigen::VectorXf::Zero(this->m);
	this->bias_item_vector = Eigen::VectorXf::Zero(this->n);

	return(this);
}

model::LatentFactorModel* model::LatentFactorModel::iterate(float learning_rate, float reg_term)
{
	for (long i = 0; i < this->data_rows; i++) {
		int user = this->data.coeff(i, 0);
		int item = this->data.coeff(i, 1);
		int rating = this->data.coeff(i, 2);

		float pred = this->global_mean + this->bias_user_vector(user) + bias_item_vector(item);
		pred += this->latent_user_matrix.row(user).dot(this->latent_item_matrix.row(item));
		float err = rating - pred;

		this->bias_user_vector(user) += learning_rate * (err - reg_term * this->bias_user_vector(user));
		this->bias_item_vector(item) += learning_rate * (err - reg_term * this->bias_item_vector(item));

		this->latent_user_matrix.row(user) += learning_rate * (err * this->latent_item_matrix.row(item) - reg_term * this->latent_user_matrix.row(user));
		this->latent_item_matrix.row(item) += learning_rate * (err * this->latent_user_matrix.row(user) - reg_term * this->latent_item_matrix.row(item));
	}
	return this;
}
