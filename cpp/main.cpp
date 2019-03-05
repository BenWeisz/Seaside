#include <iostream>

#include "net.h"

using namespace Seaside;

int main(){
	Net net({2, 3, 2}, {"sigmoid", "sigmoid"});
	
	Mat input_data(2, 8);
	input_data.set_columns({Vec({3, 1.5}), Vec({2, 1}), 
						   Vec({4, 1.5}), Vec({3, 1}),
	     				   Vec({3.5, 0.5}), Vec({2, 0.5}), 
	 					   Vec({5.5, 1}), Vec({1, 1})});

	Mat target_data(2, 8);
	target_data.set_columns({Vec({1, 0}), Vec({0, 1}),
	 						 Vec({1, 0}), Vec({0, 1}),
	  						 Vec({1, 0}), Vec({0, 1}),
	   						 Vec({1, 0}), Vec({0, 1})});

	net.learn("mse", input_data, target_data, 0.1, 10000);

	Mat test_data(2, 1);
	test_data.set_columns({Vec({3, 1.5}), Vec({2, 1}), 
						   Vec({4, 1.5}), Vec({3, 1}),
	     				   Vec({3.5, 0.5}), Vec({2, 0.5}), 
	 					   Vec({5.5, 1}), Vec({1, 1})});
	
	Mat test_output = net.query(test_data);
	test_output.print();

	return 0;
}
