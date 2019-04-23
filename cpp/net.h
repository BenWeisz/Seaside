#pragma once

#include <map>

#include "mat.h"

#define CONST_SEASIDE_MODEL 98426124
#define METRICS_NONE 0
#define METRICS_MIN  1
#define METRICS_ALL  2

namespace Seaside {
    class Net {
        private:
            std::vector<Mat> layers;
            std::vector<int> schematic;
            std::vector<std::string> active_funcs;
            std::map<std::string, Vec (*)(Vec)> maps; 
            void Fwrite(const void *ptr, size_t size, size_t count, FILE *stream);
            void Fread(void *ptr, size_t size, size_t count, FILE *stream);
            void metrics(double percent);
        public:
            int metrics_type;
        public:
            Net(std::vector<int> schematic, std::vector<std::string> active_funcs);
            
            Mat query(Mat input_data);
            std::vector<Mat> feed_forward(Mat input_data);
        
            void train_mse(Mat input_data, Mat target_data, double eta);
            void train_xent(Mat input_data, Mat target_data, double eta);

            void learn(std::string optimizer, Mat input_data, Mat target_data, double eta, int epochs, int batch_size);

            void save(const char *file_name);
            void load(const char *file_name);
            
            std::vector<Mat> mini_batch(Mat input_data, Mat target_data, int size);

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
                double max_in = -999999;
                
                for (int i = 0; i < v.len(); i++){
                    if (v[i] > max_in)
                        max_in = v[i];
                }

                double exponential_sum = 0;
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

            static Vec htan(Vec v){
                for (int i = 0; i < v.len(); i++)
                    v[i] = (std::exp(2.0 * v[i]) - 1.0) / (std::exp(2.0 * v[i]) + 1.0);

                return v;
            }

            static Vec tanh_prime(Vec v){
                for (int i = 0; i < v.len(); i++)
                    v[i] = (4 * std::exp(2.0 * v[i])) / ((std::exp(2.0 * v[i]) + 1) * (std::exp(2.0 * v[i]) + 1));

                return v;
            }
    };
}