#include"../dependencies/fast-cpp-csv-parser/csv.h"
#include <iostream>
#include <chrono>
#include<cmath>
#include"../src/LatentFactorModel.h"

int main() {
	std::cout << "Hello world"<<std::endl;
	Eigen::setNbThreads(1);
	auto cores = Eigen::nbThreads();
	Eigen::MatrixXi mat(25000095, 3);
	io::CSVReader<3> in("path to csv goes here");
	in.read_header(io::ignore_extra_column, "userId", "movieId", "rating");
	int user; int movie; float rating;
	int i = 0;
	while (in.read_row(user, movie, rating)) {
		mat(i, 0) = user;
		mat(i, 1) = movie;
		mat(i, 2) = static_cast<int>(rating);
		++i;
	}
	auto mat_train = mat.topRows(20000000);
	auto mat_test = mat.bottomRows(5000000);
	auto start = std::chrono::system_clock::now();
	model::LatentFactorModel* m = new model::LatentFactorModel;
	m = m->build(mat_train, 40);
	m = m->initialize();
	int n = 100;
	i = 0;

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
	std::cout << mat.block<10, 3>(0, 0).format(CleanFmt);
	std::cout << m->latent_user_matrix.block<10,10>(1,1).format(CleanFmt);
	std::cout << m->latent_item_matrix.block<10, 10>(1, 1).format(CleanFmt);

	std::cout << m->bias_user_vector.head(10).format(CleanFmt);
	std::cout << m->bias_item_vector.head(10).format(CleanFmt);

	float error = 0;
	for (int j = 0; j < 5000000; ++j) {
		int user = mat_train.coeff(j, 0);
		int item = mat_train.coeff(j, 1);
		int rating = mat_train.coeff(j, 2);

		error += std::abs(m->predict(user, item) - rating);
	}
	std::cout << error / 5000000 << std::endl;
	std::cout << "end" << std::endl;
}