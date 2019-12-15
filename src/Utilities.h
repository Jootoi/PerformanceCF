#pragma once
#include<vector>
#include<algorithm>
#include <Eigen/SparseCore>
namespace Utilities {

	void reduceToUniques(void* vec);


	template<typename T>
	T sumVectorElements(std::vector<T>& a) {
		T sum = 0;
		#pragma omp parallel for reduction (+:sum)
		for (int i = 0; i < a.size(); i++) {
			sum = sum + a[i];
		}
		return sum;
	}

	template<typename T>
	T maxVectorElement(std::vector<T>& a) {
		
		return *std::max_element(a.begin(), a.end());
	}


//Assumes that the user supplies vector that is large enough to hold all the triplets
	template<typename T>
	void createTripletVector(std::vector<int> &users, std::vector<int> &items, std::vector<T> &ratings, std::vector<Eigen::Triplet<T>> &tripletVec) {
		if (users.size() == items.size() && users.size() == ratings.size()) {
			for (int i = 0; i < users.size(); i++) {
				tripletVec[i] = Eigen::Triplet<T>(items[i], users[i], ratings[i]);
			}
		}
		else {
			throw;
		}
	}

	template<typename T>
	Eigen::Matrix<float, Eigen::Dynamic, 1> colSums(Eigen::SparseMatrix<T> mat) {
		Eigen::VectorXf oneVec = Eigen::VectorXf::Ones(mat.rows());
		Eigen::Matrix<float, Eigen::Dynamic, 1> sums = oneVec.transpose()*mat;
		return sums;
	}

	template<typename T>
	Eigen::Matrix<float, Eigen::Dynamic, 1> colNonZeros(Eigen::SparseMatrix<T> mat) {
		Eigen::VectorXf nonzeros(mat.cols());
		for (int i = 0; i < mat.cols(); i++) {
			nonzeros[i] = mat.innerVector(i).nonZeros();
		}
		return nonzeros;
	}

	template<typename T>
	Eigen::Matrix<float, Eigen::Dynamic, 1> colMeans(Eigen::SparseMatrix<T> mat) {
		Eigen::VectorXf means(mat.cols());
		means = colSums(mat).array() * colNonZeros(mat).array().inverse();
		return means;
	}
	template<typename T>
	Eigen::SparseMatrix<T> BinarizeByCol(Eigen::SparseMatrix<T> mat) {
		Eigen::Matrix<float, Eigen::Dynamic, 1> means = colMeans(mat);
		Eigen::SparseMatrix<T> binMat = mat;
		for (int k = 0; k < binMat.outerSize(); ++k)
			for (typename Eigen::SparseMatrix<T>::InnerIterator it(binMat, k); it; ++it)
				it.valueRef() -= means(k);
		binMat.prune(1,0.5);
		return binMat;
	}

	template<typename T>
	Eigen::SparseMatrix<T> createSparseMatrix(std::vector<int>& users, std::vector<int>& items, std::vector<T>& ratings) {
		int i_max = maxVectorElement<int>(users)+1;
		int j_max = maxVectorElement<int>(items)+1;
		Eigen::SparseMatrix<T, Eigen::RowMajor> mat(j_max, i_max);
		std::vector<Eigen::Triplet<T>> tripletVec(users.size());
		createTripletVector(users, items, ratings, tripletVec);
		mat.setFromTriplets(tripletVec.begin(), tripletVec.end());
		return mat;
	}


}

