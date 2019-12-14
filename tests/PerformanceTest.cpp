#include"../dependencies/fast-cpp-csv-parser/csv.h"
#include <iostream>
#include <vector>
#include <chrono>
#include<cmath>
#include"../src/Utilities.h"
#include"../src/LatentFactorModel.h"

template<typename T>
void readMovieLensToMatrix(std::string path, int rows, std::vector<int> &users, std::vector<int> &movies, std::vector<T> &ratings) {
	io::CSVReader<4> in(path);
	in.read_header(io::ignore_extra_column, "userId", "movieId", "rating", "timestamp");
	int user; int movie; double rating; long timestamp;
	int i;
	for (i = 0; i < rows; i++) {
		in.read_row(user, movie, rating, timestamp);
		users[i] = user;
		movies[i] = movie;
		ratings[i] = static_cast<T>(rating);
	}
}

template<typename T>
void testLatentFactorModel(std::vector<int>& users, std::vector<int>& items, std::vector<T>& ratings, int factors, int iterations, float learning_rate, float reg_term) {
	auto start = std::chrono::system_clock::now();
	model::LatentFactorModel<T>* m = new model::LatentFactorModel<T>(users, items, ratings);
	m = m->build(factors);
	int i = 0;
	auto start_iteration = std::chrono::system_clock::now();
	while (i < iterations) {
		m = m->iterate(learning_rate, reg_term);
		++i;

	}
	auto end = std::chrono::system_clock::now();
	auto diff_init = std::chrono::duration_cast<std::chrono::seconds>(start_iteration - start);
	auto diff_iter = std::chrono::duration_cast<std::chrono::seconds>(end - start_iteration);
	std::cout << "Second to init: " << diff_init.count() <<" Seconds to iterate: "<<diff_iter.count() <<"  factors: " << factors << " iterations: " << iterations << std::endl;
	std::cout << "items rows: " << m->n << " users rows: " << m->m << std::endl;
	//Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
	//std::cout << m->latent_user_matrix.block<10,10>(0,0).format(CleanFmt);
	//std::cout << m->latent_item_matrix.block<10, 10>(0, 0).format(CleanFmt);

	//std::cout << m->bias_user_vector.head(10).format(CleanFmt);
	//std::cout << m->bias_item_vector.head(10).format(CleanFmt);
}

template<typename T>
void testSparseMatrixCreation(std::vector<int>& users, std::vector<int>& items, std::vector<T>& ratings) {
	Eigen::SparseMatrix<T> mat = Utilities::createSparseMatrix<T>(users, items, ratings);
	std::cout << "Non zeros: " << mat.nonZeros() << " rows: " << mat.rows() << " cols: " << mat.cols() << std::endl;
	Eigen::Matrix<float, Eigen::Dynamic, 1> rowSums = Utilities::rowSums(mat);
	Eigen::Matrix<int, Eigen::Dynamic, 1> nonzeros = Utilities::rowNonZeros(mat);
	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
	std::cout << rowSums.head(10).format(CleanFmt) << std::endl<<std::endl;
	std::cout << nonzeros.head(10).format(CleanFmt) << std::endl;
}

int main() {
	std::cout << "Hello world"<<std::endl;
	std::vector<int> users(25000095);
	std::vector<int> items(25000095);
	std::vector<float> ratings(25000095);
	int n = Eigen::nbThreads();
	std::cout << "Eigen threads: " << n << std::endl;

	readMovieLensToMatrix<float>("/mnt/c/Users/Joonas/Nextcloud/Shared/Source/Projects/PerformanceCF/tests/ratings.csv", 25000095, users, items,ratings);
	
	testLatentFactorModel<float>(users, items, ratings, 1, 1, 0.01, 0.05);
	std::cout << users.size() << std::endl;
	testSparseMatrixCreation<float>(users, items, ratings);



	std::cout << "end" << std::endl;
}

