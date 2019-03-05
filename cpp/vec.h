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

			int num_chars(float num);
			std::string spaces(int num);
		public:
			std::vector<float> data;

			Vec(std::vector<float> data);

			int len();

			float& operator[] (const int index);
			Vec operator+ (Vec other);
			Vec operator- (Vec other);
			Vec operator* (Vec other);
			Vec operator* (float constant);
			Vec operator/ (Vec other);
			Vec operator/ (float constant);

			void print();
			float dot(Vec other);

			Vec map(Vec(*f)(Vec v));
			Vec rand(float max, float min);
			Vec clone();
		    void set(int n, float v);
	};
}
