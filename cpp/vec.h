#pragma once

#include <vector>
#include <time.h>
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <cmath>

namespace Seaside {
	class Vec {
		private:
			int precision;

			int num_chars(double num);
			std::string spaces(int num);
		public:
			std::vector<double> data;

			Vec(std::vector<double> data);

			int len();

			double& operator[] (const int index);
			Vec operator+ (Vec other);
			Vec operator- (Vec other);
			Vec operator* (Vec other);
			Vec operator* (double constant);
			Vec operator/ (Vec other);
			Vec operator/ (double constant);

			void print();
			double dot(Vec other);

			Vec map(Vec(*f)(Vec v));
			Vec rand(double max, double min);
			Vec clone();
		    void set(int n, double v);
	};
}
