#include"../dependencies/fast-cpp-csv-parser/csv.h"
#include <iostream>
#include <vector>
#include <chrono>
#include<cmath>
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
void splitToTrainTest(std::vector<int>& users, std::vector<int>& items, std::vector<T>& ratings, std::vector<int>& users_train, std::vector<int>& items_train, std::vector<T>& ratings_train, std::vector<int>& users_test, std::vector<int>& items_test, std::vector<T>& ratings_test) {
	int i = 0;
	int j = 0;
	int k = 0;
	while (i < users.size()) {
		if (!(i % 10)) {
			users_test[j] = users[i];
			items_test[j] = items[i];
			ratings_test[j] = ratings[i];
			j++;
		}
		else {
			users_train[k] = users[i];
			items_train[k] = items[i];
			ratings_train[k] = ratings[i];
			k++;
		}
		i++;
	}
	users_test.erase(users_test.begin() + j, users_test.end());
	items_test.erase(items_test.begin() + j, items_test.end());
	ratings_test.erase(ratings_test.begin() + j, ratings_test.end());

	users_train.erase(users_train.begin() + k, users_train.end());
	items_train.erase(items_train.begin() + k, items_train.end());
	ratings_train.erase(ratings_train.begin() + k, ratings_train.end());
}

template<typename T>
void testPredicting(std::vector<int>& users, std::vector<int>& items, std::vector<T>& ratings, model::LatentFactorModel<T> *m) {
	double sum = 0;
	for (int i = 1; i < users.size(); i++) {
		float pred = m->predict(users[i], items[i]);
		sum += std::abs(pred - ratings[i]);
	}
	std::cout << "MAE: " << sum / users.size() << std::endl;
}

void printFirstLast(std::vector<int> vec) {
	std::cout << *vec.begin() << " " << *vec.end() << std::endl;
}

template<typename T>
void testLatentFactorModel(std::vector<int>& users, std::vector<int>& items, std::vector<T>& ratings, int factors, int iterations, float learning_rate, float reg_term) {
	std::vector<int> users_train(24000000); std::vector<int> items_train(24000000); std::vector<T> ratings_train(24000000);
	std::vector<int> users_test(3000000); std::vector<int> items_test(3000000); std::vector<T> ratings_test(3000000);
	splitToTrainTest(users, items, ratings, users_train, items_train, ratings_train, users_test, items_test, ratings_test);
	printFirstLast(users_train); printFirstLast(items_train); printFirstLast(users_test); printFirstLast(items_test);
	auto start = std::chrono::system_clock::now();
	model::LatentFactorModel<T>* m = new model::LatentFactorModel<T>(users_train, items_train, ratings_train);
	m = m->build(factors);
	int i = 0;
	auto start_iteration = std::chrono::system_clock::now();
	while (i < iterations) {
		//testPredicting<T>(users_test, items_test, ratings_test, m);
		m = m->iterate(learning_rate, reg_term);
		//m = m->batchIterate(1, learning_rate, reg_term);
		++i;
		
	}
	
	auto end = std::chrono::system_clock::now();

	testPredicting<T>(users_test, items_test, ratings_test, m);
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
void testLatentFactorModelBatchIteration(std::vector<int>& users, std::vector<int>& items, std::vector<T>& ratings, int factors, int iterations, float learning_rate, float reg_term) {
	std::vector<int> users_train(24000000); std::vector<int> items_train(24000000); std::vector<T> ratings_train(24000000);
	std::vector<int> users_test(3000000); std::vector<int> items_test(3000000); std::vector<T> ratings_test(3000000);
	splitToTrainTest(users, items, ratings, users_train, items_train, ratings_train, users_test, items_test, ratings_test);
	printFirstLast(users_train); printFirstLast(items_train); printFirstLast(users_test); printFirstLast(items_test);
	auto start = std::chrono::system_clock::now();
	model::LatentFactorModel<T>* m = new model::LatentFactorModel<T>(users_train, items_train, ratings_train);
	m = m->build(factors);
	int i = 0;
	auto start_iteration = std::chrono::system_clock::now();
	m = m->batchIterate(iterations, learning_rate, reg_term);

	auto end = std::chrono::system_clock::now();

	testPredicting<T>(users_test, items_test, ratings_test, m);
	auto diff_init = std::chrono::duration_cast<std::chrono::seconds>(start_iteration - start);
	auto diff_iter = std::chrono::duration_cast<std::chrono::seconds>(end - start_iteration);
	std::cout << "Second to init: " << diff_init.count() << " Seconds to iterate: " << diff_iter.count() << "  factors: " << factors << " iterations: " << iterations << std::endl;
	std::cout << "items rows: " << m->n << " users rows: " << m->m << std::endl;


	//Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
	//std::cout << m->latent_user_matrix.block<10,10>(0,0).format(CleanFmt);
	//std::cout << m->latent_item_matrix.block<10, 10>(0, 0).format(CleanFmt);

	//std::cout << m->bias_user_vector.head(10).format(CleanFmt);
	//std::cout << m->bias_item_vector.head(10).format(CleanFmt);
}




template<typename T>
void testSparseMatrixCreation(std::vector<int>& users, std::vector<int>& items, std::vector<T>& ratings) {
	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
	Eigen::SparseMatrix<T> mat = Utilities::createSparseMatrix<T>(users, items, ratings);
	std::cout << "Non zeros: " << mat.nonZeros() << " rows: " << mat.rows() << " cols: " << mat.cols() << std::endl;

	Eigen::Matrix<float, Eigen::Dynamic, 1> colSums = Utilities::colSums(mat);
	std::cout << colSums.head(10).format(CleanFmt) << std::endl<<std::endl;

	Eigen::Matrix<float, Eigen::Dynamic, 1> nonzeros = Utilities::colNonZeros(mat);
	std::cout << nonzeros.head(10).format(CleanFmt) << std::endl;

	Eigen::Matrix<float, Eigen::Dynamic, 1> means = Utilities::colMeans(mat);
	std::cout << means.head(10).format(CleanFmt) << std::endl;

	Eigen::SparseMatrix<bool> binMat = Utilities::BinarizeByCol(mat);
	std::cout << "Non zeros: " << binMat.nonZeros() << " rows: " << binMat.rows() << " cols: " << binMat.cols() << std::endl;

	auto start = std::chrono::system_clock::now();
	std::vector<float> sim = Utilities::BatchJaccardSimilarity(1, binMat);
	auto end = std::chrono::system_clock::now();
	auto diff = std::chrono::duration_cast<std::chrono::seconds>(end - start);
	std::cout << "Seconds: " << diff.count() << std::endl;
	auto it = max_element(std::begin(sim), std::end(sim));
	std::cout<<"max: "<<*it<< std::endl;
}

int main(int argc, char* argv[]) {
	std::vector<int> users(25000095);
	std::vector<int> items(25000095);
	std::vector<char> ratings(25000095);
	int n = Eigen::nbThreads();
	std::cout << "Eigen threads: " << n << std::endl;

	readMovieLensToMatrix<char>(argv[1], 25000095, users, items,ratings);
	
	testLatentFactorModel<char>(users, items, ratings, 15, 100, 0.01, 0.05);
	testLatentFactorModelBatchIteration<char>(users, items, ratings, 15, 100, 0.01, 0.05);
	std::cout << users.size() << std::endl;
	
	//testSparseMatrixCreation(users, items, ratings);



	std::cout << "end" << std::endl;}

