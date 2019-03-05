#include "vec.h"

namespace Seaside {
	class Mat {
		private:
			int precision;

			int num_chars(float num);
			std::string spaces(int num);
		public:
			std::vector<Vec> data;

			Mat(int m, int n);
			void set_columns(std::vector<Vec> cols);
			std::pair<int, int> dim();
			Vec row(int row_num);

			Vec& operator[] (const int index);
			Mat operator+ (Mat other);
			Mat operator- (Mat other);
			Mat operator* (Mat other);
			Mat operator* (float constant);
			Mat operator/ (Mat other);
			Mat operator/ (float constant);

			Mat hadamard (Mat other);
			Mat rand(float max, float min);
			Mat append_column(Vec col);
			Mat append_row(Vec row);
			Mat remove_column(int col);
			Mat remove_row(int row);

			Mat transpose();
			Mat identity();

			Mat map(Vec(*f)(Vec v));

			void print();
	};
}
