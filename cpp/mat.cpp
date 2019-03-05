#include "mat.h"

namespace Seaside {
	Mat::Mat(int m, int n){
		this->precision = 2;

		for (int i = 0; i < n; i++){
			Vec temp({});
			temp.set(m, 0);

			this->data.push_back(temp);
		}
	}

	void Mat::set_columns(std::vector<Vec> cols){
		this->data = cols;
	}

	std::pair<int, int> Mat::dim(){
		if (this->data.size() == 0)
			return std::pair<int, int>(0, 0);
		else if (this->data[0].len() == 0)
			return std::pair<int, int>(0, 0);
		else
			return std::pair<int, int>(this->data[0].len(), this->data.size());
	}

	Vec Mat::row(int row_num){
		auto this_dim = this->dim();

		if (row_num < 0 || row_num >= this_dim.first){
			std::cout << "ROW ACCESS: The row you are accessing must be within the dimensions of the Matrix!" << std::endl;
			std::cout << "\tDimension of the Matrix: " << this_dim.first << " x " << this_dim.second << std::endl;
			exit(EXIT_FAILURE);
		} else {
			Vec out({});
			out.set(this_dim.second, 0);

			for (int x = 0; x < this_dim.second; x++){
				out[x] = (this->operator[](x))[row_num];
			}

			return out;
		}
	}

	Vec& Mat::operator[] (const int index){
		return this->data[index];
	}

	Mat Mat::operator+ (Mat other){
		auto this_dim = this->dim();
		auto other_dim = other.dim();

		if (this_dim.first != other_dim.first || this_dim.second != other_dim.second){
			std::cout << "ADDITION: The dimensions of the two Matricies must match!" << std::endl;
			std::cout << "\tDimension of first Matrix:  " << this_dim.first << " x " << this_dim.second << std::endl;
			std::cout << "\tDimension of second Matrix: " << other_dim.first << " x " << other_dim.second << std::endl;
			exit(EXIT_FAILURE);
		} else {
			Mat temp = *this;

			for (int i = 0; i < this_dim.second; i++){
				temp[i] = temp[i] + other[i];
			}

			return temp;
		}
	}

	Mat Mat::operator- (Mat other){
		auto this_dim = this->dim();
		auto other_dim = other.dim();

		if (this_dim.first != other_dim.first || this_dim.second != other_dim.second){
			std::cout << "SUBTRACTION: The dimensions of the two Matricies must match!" << std::endl;
			std::cout << "\tDimension of first Matrix:  " << this_dim.first << " x " << this_dim.second << std::endl;
			std::cout << "\tDimension of second Matrix: " << other_dim.first << " x " << other_dim.second << std::endl;
			exit(EXIT_FAILURE);
		} else {
			Mat temp = *this;

			for (int i = 0; i < this_dim.second; i++){
				temp[i] = temp[i] - other[i];
			}

			return temp;
		}
	}

	Mat Mat::hadamard (Mat other){
		auto this_dim = this->dim();
		auto other_dim = other.dim();

		if (this_dim.first != other_dim.first || this_dim.second != other_dim.second){
			std::cout << "HADAMARD MULTIPLICATION: The dimensions of the two Matricies must match!" << std::endl;
			std::cout << "\tDimension of first Matrix:  " << this_dim.first << " x " << this_dim.second << std::endl;
			std::cout << "\tDimension of second Matrix: " << other_dim.first << " x " << other_dim.second << std::endl;
			exit(EXIT_FAILURE);
		} else {
			Mat temp = *this;

			for (int i = 0; i < this_dim.second; i++){
				temp[i] = temp[i] * other[i];
			}

			return temp;
		}
	}

	Mat Mat::operator* (Mat other){
		auto this_dim = this->dim();
		auto other_dim = other.dim();

		if (this_dim.second != other_dim.first){
			std::cout << "MULTIPLICATION: The width of the first Matrix must match the height of the second Matrix!" << std::endl;
			std::cout << "\tDimension of first Matrix:  " << this_dim.first << " x " << this_dim.second << std::endl;
			std::cout << "\tDimension of second Matrix: " << other_dim.first << " x " << other_dim.second << std::endl;
			exit(EXIT_FAILURE);
		} else {
			Mat prod(this_dim.first, other_dim.second);

			for (int y = 0; y < this_dim.first; y++){
				for (int x = 0; x < other_dim.second; x++){
					Vec this_row = this->row(y);

					prod[x][y] = this_row.dot(other[x]);
				}
			}

			return prod;
		}
	}

	Mat Mat::operator* (float constant){
		auto this_dim = this->dim();

		Mat temp = *this;

		for (int i = 0; i < this_dim.second; i++){
			temp[i] = temp[i] * constant;
		}

		return temp;
	}

	Mat Mat::operator/ (Mat other){
		auto this_dim = this->dim();
		auto other_dim = other.dim();

		if (this_dim.first != other_dim.first || this_dim.second != other_dim.second){
			std::cout << "DIVISION: The dimensions of the two Matricies must match!" << std::endl;
			std::cout << "\tDimension of first Matrix:  " << this_dim.first << " x " << this_dim.second << std::endl;
			std::cout << "\tDimension of second Matrix: " << other_dim.first << " x " << other_dim.second << std::endl;
			exit(EXIT_FAILURE);
		} else {
			Mat temp = *this;

			for (int i = 0; i < this_dim.second; i++){
				temp[i] = temp[i] / other[i];
			}

			return temp;
		}
	}

	Mat Mat::operator/ (float constant){
		auto this_dim = this->dim();

		Mat temp = *this;

		for (int i = 0; i < this_dim.second; i++){
			temp[i] = temp[i] / constant;
		}

		return temp;
	}

	int Mat::num_chars(float num){
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

	std::string Mat::spaces(int num){
		std::string out;

		for (int i = 0; i < num; i++)
			out += " ";

		return out;
	}

	void Mat::print(){
		Mat temp = *this;
		auto this_dim = this->dim();

		std::vector<int> maxes;

		for (int i = 0; i < this_dim.second; i++)
			maxes.push_back(-1);

		for (int i = 0; i < this_dim.second; i++){
			for (int j = 0; j < this_dim.first; j++){
				int num_characters = this->num_chars(temp[i][j]);

				if (num_characters > maxes[i])
					maxes[i] = num_characters;
			}
		}

		int total_spaces = 1;
		for (int i = 0; i < this_dim.second; i++)
			total_spaces += maxes[i] + 1;

		std::string out;
		out += "(";
		out += this->spaces(total_spaces);
		out += ")";

		std::cout << out << std::endl;

		for (int y = 0; y < this_dim.first; y++){
			out = "| ";

			for (int x = 0; x < this_dim.second; x++){
				float num = temp[x][y];
				float abs_num = std::abs(num);

				int abs_floor_num = std::floor(abs_num);
				out += this->spaces(maxes[x] - this->num_chars(num));

				if (num < 0)
					out += "-";

				out += std::to_string(abs_floor_num);
				out += ".";

				std::string frac = std::to_string(abs_num);
				std::string abs_frac = std::to_string(abs_floor_num);

				int i = abs_frac.size() + 1;
				while (i < (int)frac.size() && i < this->precision + (int)abs_frac.size() + 1){
					out += frac[i];
					i++;
				}

				out += " ";
			}

			out += "|";
			std::cout << out << std::endl;
		}

		out = "(";
		out += this->spaces(total_spaces);
		out += ")";

		std::cout << out << std::endl;
	}

	Mat Mat::rand(float max, float min){
		Mat temp = *this;

		auto this_dim = this->dim();
		for (int i = 0; i < this_dim.second; i++)
			temp[i] = temp[i].rand(max, min);

		return temp;
	}

	Mat Mat::append_column(Vec col){
		auto this_dim = this->dim();

		if (col.len() != this_dim.first){
			std::cout << "COLUMN APPEND: The length of the column Vector must match the height of the Matrix!" << std::endl;
			std::cout << "\tDimension of Matrix: " << this_dim.first << " x " << this_dim.second << std::endl;
			std::cout << "\tLength of Vector:    " << col.len() << std::endl;
			exit(EXIT_FAILURE);
		} else {
			this->data.push_back(col);

			Mat temp = *this;
			this->data.erase(this->data.end());

			return temp;
		}
	}

	Mat Mat::append_row(Vec row){
		auto this_dim = this->dim();

		if (row.len() != this_dim.second){
			std::cout << "ROW APPEND: The length of the row Vector must match the width of the Matrix!" << std::endl;
			std::cout << "\tDimension of Matrix: " << this_dim.first << " x " << this_dim.second << std::endl;
			std::cout << "\tLength of Vector:    " << row.len() << std::endl;
			exit(EXIT_FAILURE);
		} else {
			Mat temp = *this;

			for (int i = 0; i < this_dim.second; i++){
				temp[i].data.push_back(row[i]);
			}

			return temp;
		}
	}

	Mat Mat::remove_column(int col){
		auto this_dim = this->dim();

		if (col < 0 || col >= this_dim.second){
			std::cout << "REMOVE COLUMN: The column index must be within bounds!" << std::endl;
			std::cout << "\tDimension of Matrix: " << this_dim.first << " x " << this_dim.second << std::endl;
			exit(EXIT_FAILURE);
		} else {
			Mat temp = *this;

			temp.data.erase(temp.data.begin() + col);

			return temp;
		}
	}

	Mat Mat::remove_row(int row){
		auto this_dim = this->dim();

		if (row < 0 || row >= this_dim.first){
			std::cout << "REMOVE ROW: The row index must be within bounds!" << std::endl;
			std::cout << "\tDimension of Matrix: " << this_dim.first << " x " << this_dim.second << std::endl;
			exit(EXIT_FAILURE);
		} else {
			Mat temp = *this;
			for (int i = 0; i < this_dim.second; i++){
				temp[i].data.erase(temp[i].data.begin() + row);
			}

			return temp;
		}
	}

	Mat Mat::transpose(){
		auto this_dim = this->dim();

		std::vector<Vec> cols;
		for (int i = 0; i < this_dim.first; i++){
			cols.push_back(this->row(i));
		}

		Mat temp(0, 0);
		temp.set_columns(cols);

		return temp;
	}

	Mat Mat::identity(){
		auto this_dim = this->dim();

		if (this_dim.first != this_dim.second){
			std::cout << "IDENTITY: Set up the vector so it has square dimensions!" << std::endl;
			std::cout << "\tDimensions of Matrix: " << this_dim.first << " x " << this_dim.second << std::endl;
			exit(EXIT_FAILURE);
		} else {
			Mat temp = *this;

			for (int y = 0; y < this_dim.first; y++){
				for (int x = 0; x < this_dim.second; x++){
					if (x == y)
						temp[x][y] = 1.0;
					else
						temp[x][y] = 0.0;
				}
			}

			return temp;
		}
	}

	Mat Mat::map(Vec(*f)(Vec v)){
		Mat temp = *this;

		auto this_dim = this->dim();

		for (int i = 0; i < this_dim.second; i++){
			temp[i] = temp[i].map(f);
		}

		return temp;
	}
}
