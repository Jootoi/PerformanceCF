#include"../dependencies/fast-cpp-csv-parser/csv.h"
#include <iostream>
#include <vector>
#include <chrono>
#include<cmath>
#include"../src/LatentFactorModel.h"

template<typename T>
void readMovieLensToMatrix(std::string path, int rows, std::vector<int> &users, std::vector<int> &movies, std::vector<T> &ratings) {
	io::CSVReader<3> in(path);
	in.read_header(io::ignore_extra_column, "userId", "movieId", "rating");
	int user; int movie; double rating;
	for (int i = 0; i < rows; i++) {
		in.read_row(user, movie, rating);
		users[i] = user;
		movies[i] = movie;
		ratings[i] = static_cast<T>(rating);
		++i;
	}
}

int main() {
	std::cout << "Hello world"<<std::endl;
	std::vector<int> users(25000095);
	std::vector<int> items(25000095);
	std::vector<float> ratings(25000095);

	readMovieLensToMatrix<float>("ratings.csv", 25000095, users, items,ratings);
	auto start = std::chrono::system_clock::now();
	model::LatentFactorModel<float>* m = new model::LatentFactorModel<float>(users, items, ratings);
	m = m->build(10);
	int n = 10;
	int i = 0;

	while (i < n) {
		m = m->iterate(0.01, 0.05);
		++i;

	}
	auto end = std::chrono::system_clock::now();
	auto diff = std::chrono::duration_cast<std::chrono::seconds>(end - start);

	//float result = m->predict(162540, 3869);
	std::cout << diff.count() << std::endl;

	//std::cout << result << std::endl;

	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
	//std::cout << mat_train.block<10, 3>(0, 0).format(CleanFmt);
	std::cout << m->latent_user_matrix.block<10,10>(1,1).format(CleanFmt);
	std::cout << m->latent_item_matrix.block<10, 10>(1, 1).format(CleanFmt);

	std::cout << m->bias_user_vector.head(10).format(CleanFmt);
	std::cout << m->bias_item_vector.head(10).format(CleanFmt);

	//float error = 0;
	//for (int j = 0; j < 5000000; ++j) {
	//	int user = mat_train.coeff(j, 0);
	//	int item = mat_train.coeff(j, 1);
	//	int rating = mat_train.coeff(j, 2);

	//	error += std::abs(m->predict(user, item) - rating);
	//}
	//std::cout << error / 5000000 << std::endl;
	std::cout << "end" << std::endl;
}

