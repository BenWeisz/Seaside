#pragma once

#include <map>

#include "mat.h"

namespace Seaside {
    class Net {
        private:
            std::vector<Mat> layers;
            std::vector<int> schematic;
            std::vector<std::string> active_funcs;
            std::map<std::string, Vec (*)(Vec)> maps; 
        public:
            Net(std::vector<int> schematic, std::vector<std::string> active_funcs);
            
            Mat query(Mat input_data);
            std::vector<Mat> feed_forward(Mat input_data);
        
            void train_mse(Mat input_data, Mat target_data, float eta);
            void train_xent(Mat input_data, Mat target_data, float eta);

            void learn(std::string optimizer, Mat input_data, Mat target_data, float eta, int epochs);

            static Vec sigmoid(Vec v){
                int len = v.len();

                for (int i = 0; i < len; i++)
                    v[i] = 1.0 / (1.0 + std::exp(-v[i]));

                return v;
            }

            static Vec sigmoid_prime(Vec v){
                int len = v.len();

                for (int i = 0; i < len; i++)
                    v[i] = std::exp(-v[i]) / ((1.0 + std::exp(-v[i])) * (1.0 + std::exp(-v[i])));

                return v;
            }

            static Vec soft_max(Vec v){
                float max_in = -999999;
                
                for (int i = 0; i < v.len(); i++){
                    if (v[i] > max_in)
                        max_in = v[i];
                }

                float exponential_sum = 0;
                for (int i = 0; i < v.len(); i++)
                    exponential_sum += std::exp(v[i] - max_in);

                for (int i = 0; i < v.len(); i++)
                    v[i] = std::exp(v[i] - max_in) / exponential_sum;

                return v;
            }

            static Vec soft_max_prime(Vec v){
                Vec soft_max_output = soft_max(v);

                for (int i = 0; i < v.len(); i++)
                    v[i] = soft_max_output[i] * (1.0 - soft_max_output[i]);

                return v;
            }

            static Vec relu(Vec v){
                for (int i = 0; i < v.len(); i++)
                    v[i] = std::log(1.0 + std::exp(v[i]));
            
                return v;
            }

            static Vec relu_prime(Vec v){
                for (int i = 0; i < v.len(); i++)
                    v[i] = std::exp(v[i]) / (1.0 + std::exp(v[i]));

                return v;
            }
    };
}