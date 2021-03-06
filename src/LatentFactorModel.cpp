#include"LatentFactorModel.h"
template<typename T>
model::LatentFactorModel<T>::LatentFactorModel(std::vector<int>& users, std::vector<int>& items, std::vector<T>& ratings) :users(users), items(items), ratings(ratings)
{
	if (!(users.size() == items.size() && users.size() == ratings.size())) {
		throw;
	}
	this->m = *std::max_element(this->users.begin(), this->users.end()) + 1;
	this->n = *std::max_element(this->items.begin(), this->items.end()) + 1;

	this->data_rows = users.size();
	this->global_mean = Utilities::sumVectorElements(ratings) / this->data_rows;
}
template<typename T>
model::LatentFactorModel<T>* model::LatentFactorModel<T>::build(int factors) {
	this->factors = factors;
	this->latent_user_matrix = Eigen::MatrixXf::Random(this->m, this->factors);
	this->latent_item_matrix = Eigen::MatrixXf::Random(this->n, this->factors);

	this->bias_user_vector = Eigen::VectorXf::Zero(this->m);
	this->bias_item_vector = Eigen::VectorXf::Zero(this->n);

	return(this);
}

template<typename T>
model::LatentFactorModel<T>* model::LatentFactorModel<T>::iterate(float learning_rate, float reg_term)
{

	for (long i = 0; i < this->data_rows; i++) {
		int user = this->users[i];
		int item = this->items[i];
		T rating = this->ratings[i];

		float pred = this->global_mean + this->bias_user_vector(user) + bias_item_vector(item) + this->latent_user_matrix.row(user) * this->latent_item_matrix.row(item).transpose();
		float err = rating - pred;

		this->bias_user_vector(user) += learning_rate * (err - reg_term * this->bias_user_vector(user));
		this->bias_item_vector(item) += learning_rate * (err - reg_term * this->bias_item_vector(item));

		this->latent_user_matrix.row(user) += learning_rate * (err * this->latent_item_matrix.row(item) - reg_term * this->latent_user_matrix.row(user));
		this->latent_item_matrix.row(item) += learning_rate * (err * this->latent_user_matrix.row(user) - reg_term * this->latent_item_matrix.row(item));
	}
	return this;
}

template<typename T>
model::LatentFactorModel<T>* model::LatentFactorModel<T>::batchIterate(int epochs, float learning_rate, float reg_term)
{
	for (int e = 0; e < epochs; e++) {
		int mini_batch_size = 64;
		int num_batches = this->data_rows / mini_batch_size;


#pragma omp parallel for default(none) shared(mini_batch_size, num_batches, learning_rate, reg_term)
		for (int k = 0; k < num_batches; k++) {
			std::vector<float> user_bias(mini_batch_size);
			std::vector<float> item_bias(mini_batch_size);
			std::vector<float> err(mini_batch_size);
			int begin = k * mini_batch_size;
			for (int i = 0; i < mini_batch_size; i++) {
				int user = this->users[i + begin];
				int item = this->items[i + begin];
				T rating = this->ratings[i + begin];

				float pred = this->global_mean + this->bias_user_vector(user) + bias_item_vector(item) + this->latent_user_matrix.row(user) * this->latent_item_matrix.row(item).transpose();
				err[i] = rating - pred;

				user_bias[i] = learning_rate * (err[i] - reg_term * this->bias_user_vector(user));
				item_bias[i] = learning_rate * (err[i] - reg_term * this->bias_item_vector(item));

			}
#pragma omp critical
			for (int j = 0; j < mini_batch_size; j++) {
				int user = this->users[j + begin];
				int item = this->items[j + begin];

				this->bias_user_vector(user) += user_bias[j];
				this->bias_item_vector(item) += item_bias[j];

				this->latent_user_matrix.row(user) += learning_rate * (err[j] * this->latent_item_matrix.row(item) - reg_term * this->latent_user_matrix.row(user));
				this->latent_item_matrix.row(item) += learning_rate * (err[j] * this->latent_user_matrix.row(user) - reg_term * this->latent_item_matrix.row(item));

			}
			//this->bias_user_vector(user) += learning_rate * (err[i] - reg_term * this->bias_user_vector(user));
			//this->bias_item_vector(item) += learning_rate * (err[i] - reg_term * this->bias_item_vector(item));

		}
	}



	return this;
}

template<typename T>
float model::LatentFactorModel<T>::predict(int user, int item)
{
	float pred = this->global_mean;
	if (user <= this->m) {
		pred += this->bias_user_vector(user);
	}
	if (item <= this->n) {
		pred += bias_item_vector(item);
	}
	if (user <= this->m && item <= this->n) {
		pred += this->latent_user_matrix.row(user).dot(this->latent_item_matrix.row(item));
	}

	return pred;
}



template class model::LatentFactorModel<float>;
template class model::LatentFactorModel<int>;
template class model::LatentFactorModel<char>;
template class model::LatentFactorModel<bool>;