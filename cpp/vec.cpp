#include "vec.h"

namespace Seaside {
	Vec::Vec(std::vector<float> data){
		this->data = data;
		this->precision = 2;

		srand((unsigned int)time(NULL));
	}

	int Vec::len(){
		return this->data.size();
	}

	float& Vec::operator[] (const int index){
		return this->data[index];
	}

	Vec Vec::operator+ (Vec other){
		int this_len = this->len();
		int other_len = other.len();

		if (this_len != other_len){
			std::cout << "ADDITION: The lengths of the two Vectors must match!" << std::endl;
			std::cout << "\tLength of first Vector:  " << this_len << std::endl;
			std::cout << "\tLength of second Vector: " << other_len << std::endl;
			exit(EXIT_FAILURE);
		} else {
			Vec temp = *this;

			for (int i = 0; i < this_len; i++){
				temp[i] += other[i];
			}

			return temp;
		}
	}

	Vec Vec::operator- (Vec other){
		int this_len = this->len();
		int other_len = other.len();

		if (this_len != other_len){
			std::cout << "SUBTRACTION: The lengths of the two Vectors must match!" << std::endl;
			std::cout << "\tLength of the first Vector:  " << this_len << std::endl;
			std::cout << "\tLength of the second Vector: " << other_len << std::endl;
			exit(EXIT_FAILURE);
		} else {
			Vec temp = *this;

			for (int i = 0; i < this_len; i++){
				temp[i] -= other[i];
			}

			return temp;
		}
	}

	Vec Vec::operator* (Vec other){
		int this_len = this->len();
		int other_len = other.len();

		if (this_len != other_len){
			std::cout << "MULTIPLICATION: The lengths of the two Vectors must match!" << std::endl;
			std::cout << "\tLength of the first Vector:  " << this_len << std::endl;
			std::cout << "\tLength of the second Vector: " << other_len << std::endl;
			exit(EXIT_FAILURE);
		} else {
			Vec temp = *this;

			for (int i = 0; i < this_len; i++){
				temp[i] *= other[i];
			}

			return temp;
		}
	}

	Vec Vec::operator* (float constant){
		int this_len = this->len();

		Vec temp = *this;

		for (int i = 0; i < this_len; i++){
			temp[i] *= constant;
		}

		return temp;
	}

	Vec Vec::operator/ (Vec other){
		int this_len = this->len();
		int other_len = other.len();

		if (this_len != other_len){
			std::cout << "DIVISION: The lengths of the two Vectors must match!" << std::endl;
			std::cout << "\tLength of the first Vector:  " << this_len << std::endl;
			std::cout << "\tLength of the second Vector: " << other_len << std::endl;
			exit(EXIT_FAILURE);
		} else {
			Vec temp = *this;

			for (int i = 0; i < this_len; i++){
				temp[i] /= other[i];
			}

			return temp;
		}
	}

	Vec Vec::operator/ (float constant){
		int this_len = this->len();

		Vec temp = *this;

		for (int i = 0; i < this_len; i++){
			temp[i] /= constant;
		}

		return temp;
	}

	int Vec::num_chars(float num){
		int chars = 0;
		if (num < 0)
			chars++;

		float abs_num = std::abs(num);
		int abs_floor_num = std::floor(abs_num);

		if (abs_floor_num == 0)
			chars += 2;
		else
			chars += std::log10(abs_floor_num) + 2;

		chars += this->precision;

		return chars;
	}

	std::string Vec::spaces(int num){
		std::string out;

		for (int i = 0; i < num; i++)
			out += " ";

		return out;
	}

	void Vec::print(){
		Vec temp = *this;

		int this_len = temp.len();
		int max_chars = -1;

		for (int i = 0; i < this_len; i++){
			int num_characters = num_chars(temp[i]);

			if (num_characters > max_chars)
				max_chars = num_characters;
		}

		std::string out;
		out += "(";
		out += temp.spaces(2 + max_chars);
		out += ")";

		std::cout << out << std::endl;

		for (int i = 0; i < this_len; i++){
			out = "| ";

			float num = temp[i];
			float abs_num = std::abs(num);
			int abs_floor_num = std::floor(abs_num);

			out += temp.spaces(max_chars - temp.num_chars(num));

			if (num < 0)
				out += "-";

			out += std::to_string(abs_floor_num);
			out += ".";

			std::string frac = std::to_string(abs_num);
			std::string abs_frac = std::to_string(abs_floor_num);
			int j = abs_frac.size() + 1;
			while (j < (int)frac.size() && j < this->precision + (int)abs_frac.size() + 1){
				out += frac[j];

				j++;
			}

			out += " |";
			std::cout << out << std::endl;
		}

		out = "(";
		out += temp.spaces(2 + max_chars);
		out += ")";

		std::cout << out << std::endl;
	}

	float Vec::dot(Vec other){
		int this_len = this->len();
		int other_len = other.len();
		if (this_len != other_len){
			std::cout << "DOT PRODUCT: The lengths of the two Vectors must match!" << std::endl;
			std::cout << "\tLength of first Vector:  " << this_len << std::endl;
			std::cout << "\tLength of second Vector: " << other_len << std::endl;
			exit(EXIT_FAILURE);
		} else {
			Vec temp = *this;

			float sum = 0.0f;
			for (int i = 0; i < this_len; i++){
				sum += temp[i] * other[i];
			}

			return sum;
		}
	}

	Vec Vec::map(Vec(*f)(Vec v)){
		Vec temp = *this;
		return f(temp);
	}

	Vec Vec::rand(float max, float min){
		int this_len = this->len();

		Vec temp = *this;

		for (int i = 0; i < this_len; i++){
			float r = (float)std::rand() / RAND_MAX;

			temp[i] = min + (r * (max - min));
		}

		return temp;
	}

	Vec Vec::clone(){
		return *this;
	}

	void Vec::set(int n, float v){
		std::vector<float> data;

		for (int i = 0; i < n; i++)
			data.push_back(v);

		this->data = data;
	}
}
