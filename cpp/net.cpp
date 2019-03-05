#include "net.h"

namespace Seaside {
    Net::Net(std::vector<int> schematic, std::vector<std::string> active_funcs){
        this->schematic = schematic;
        this->active_funcs = active_funcs;
        
        // Add all activation functions to a map.
        this->maps.insert(std::make_pair<std::string, Vec (*)(Vec)>("sigmoid", Net::sigmoid));
        this->maps.insert(std::make_pair<std::string, Vec (*)(Vec)>("sigmoid_prime", Net::sigmoid_prime));
        this->maps.insert(std::make_pair<std::string, Vec (*)(Vec)>("soft_max", Net::soft_max));
        this->maps.insert(std::make_pair<std::string, Vec (*)(Vec)>("soft_max_prime", Net::soft_max_prime));
        this->maps.insert(std::make_pair<std::string, Vec (*)(Vec)>("relu", Net::relu));
        this->maps.insert(std::make_pair<std::string, Vec (*)(Vec)>("relu_prime", Net::relu_prime));

        // Create the weight layers for the neural network.
        for (int i = 0; i < (int)(this->schematic.size() - 1); i++){
            Mat layer(this->schematic[i + 1], this->schematic[i] + 1);
            layer = layer.rand(-1.0, 1.0);

            this->layers.push_back(layer);
        }
    }

    Mat Net::query(Mat input_data){
        // Query the neural network.
        Mat cur_layer = input_data;

        for (int i = 0; i < (int)(this->layers.size()); i++){
            Vec bias({});
            bias.set(cur_layer.dim().second, 1.0);

            cur_layer = cur_layer.append_row(bias);
            cur_layer = this->layers[i] * cur_layer;
            cur_layer = cur_layer.map(this->maps[this->active_funcs[i]]);
        }

        return cur_layer;
    }

    std::vector<Mat> Net::feed_forward(Mat input_data){
        std::vector<Mat> layer_data;

        Mat cur_layer = input_data;
        layer_data.push_back(cur_layer);

        for (int i = 0; i < (int)(this->layers.size()); i++){
            Vec bias({});
            bias.set(cur_layer.dim().second, 1.0);

            cur_layer = cur_layer.append_row(bias);
            cur_layer = this->layers[i] * cur_layer;

            layer_data.push_back(cur_layer);

            cur_layer = cur_layer.map(this->maps[this->active_funcs[i]]);
        }

        return layer_data;
    }

    void Net::train_mse(Mat input_data, Mat target_data, float eta){
        std::vector<Mat> layer_data = this->feed_forward(input_data);

        Mat layer_input = layer_data[layer_data.size() - 1];
        Mat layer_output = layer_input.map(this->maps[this->active_funcs[this->active_funcs.size() - 1]]);
        Mat layer_primed = layer_input.map(this->maps[this->active_funcs[this->active_funcs.size() - 1] + "_prime"]);

        Mat delta_error = layer_primed.hadamard(layer_output - target_data);

        for (int i = this->layers.size() - 1; i >= 0; i--){
            // Maintain layer inputs
            layer_input = layer_data[i];
            layer_primed = layer_input.map(this->maps[this->active_funcs[i] + "_prime"]);

            // Calculate layer delta errors
            Mat next_delta_error = this->layers[i].transpose();
            next_delta_error = next_delta_error.remove_row(next_delta_error.dim().first - 1);
            next_delta_error = next_delta_error * delta_error;
            next_delta_error = next_delta_error.hadamard(layer_primed);

            // Calculate layer weight deltas
            Vec bias({});
            bias.set(layer_input.dim().second, 1.0);

            Mat cur_weight_delta = layer_input;
            cur_weight_delta = cur_weight_delta.transpose();
            cur_weight_delta = cur_weight_delta.map(this->maps[this->active_funcs[i]]);
            cur_weight_delta = cur_weight_delta.append_column(bias);
            cur_weight_delta = cur_weight_delta * -eta;
            cur_weight_delta = delta_error * cur_weight_delta;

            delta_error = next_delta_error;

            // Update Weights
            this->layers[i] = this->layers[i] + cur_weight_delta;
        }
    }

    void Net::train_xent(Mat input_data, Mat target_data, float eta){
        std::vector<Mat> layer_data = this->feed_forward(input_data);

        Mat layer_output = layer_data[layer_data.size() - 1];
        for (int example_num = 0; example_num < (int)(layer_output.dim().second); example_num++){
            // Generate Softmax Jacobian
            Vec example_output_in = layer_output[example_num];
            Vec example_target = target_data[example_num];

            Vec example_output_out = Net::soft_max(example_output_in);

            Mat soft_max_jacobian(0, 0);
            std::vector<Vec> jacobian_cols;
            for (int i = 0; i < example_output_out.len(); i++)
                jacobian_cols.push_back(example_output_out.clone());

            soft_max_jacobian.set_columns(jacobian_cols);

            Mat soft_max_jacobian_copy = soft_max_jacobian.clone();

            soft_max_jacobian = soft_max_jacobian * -1.0;
            soft_max_jacobian = soft_max_jacobian.identity() + soft_max_jacobian;
            soft_max_jacobian = soft_max_jacobian_copy.transpose().hadamard(soft_max_jacobian);

            // Generate Cost Function Gradient
            Vec cost_function_gradient = example_target * -1.0;
            for (int i = 0; i < example_output_out.len(); i++)
                cost_function_gradient[i] = cost_function_gradient[i] / example_output_out[i];

            Mat gradient = Mat::to_matrix(cost_function_gradient).transpose();

            Mat delta_error = (gradient * soft_max_jacobian).transpose();

            for (int i = this->layers.size() - 1; i >= 1; i--){
                // Maintain layer inputs
                Mat this_layer_input = Mat::to_matrix(layer_data[i + 1][example_num]);
                Mat next_layer_input = Mat::to_matrix(layer_data[i][example_num]);

                Mat next_layer_output = next_layer_input.map(this->maps[this->active_funcs[i - 1]]);
                Mat next_layer_input_primed = next_layer_input.map(this->maps[this->active_funcs[i - 1] + "_prime"]);

                // Calculate layer delta errors
                Mat next_delta_error = this->layers[i].transpose();
                next_delta_error = next_delta_error.remove_row(next_delta_error.dim().first - 1);
                next_delta_error = next_delta_error * delta_error;
                next_delta_error = next_delta_error.hadamard(next_layer_input_primed);

                // Calculate layer weight deltas
                Vec bias({});
                bias.set(next_layer_output.dim().second, 1.0);

                Mat cur_weight_delta = next_layer_output;
                cur_weight_delta = cur_weight_delta.transpose();
                cur_weight_delta = cur_weight_delta.append_column(bias);
                cur_weight_delta = cur_weight_delta * -eta;
                cur_weight_delta = delta_error * cur_weight_delta;

                delta_error = next_delta_error;

                // Update Weights
                this->layers[i] = this->layers[i] + cur_weight_delta;
            }
        }
    }

    void Net::learn(std::string optimizer, Mat input_data, Mat target_data, float eta, int epochs){
        std::cout << "Starting Learning..." << std::endl;
        
        if (optimizer.compare("mse") == 0){
            for (int i = 0; i < epochs; i++){
                train_mse(input_data, target_data, eta);

                if (i % (int)(epochs / 10) == 0 && i != 0)
                    std::cout << ((i / (epochs + 0.0)) * 100) << "% Complete!" << std::endl;
            }
        }
        else if (optimizer.compare("xent") == 0){
            for (int i = 0; i < epochs; i++){
                train_xent(input_data, target_data, eta);

                if (i % (int)(epochs / 10) == 0 && i != 0)
                    std::cout << ((i / (epochs + 0.0)) * 100) << "% Complete!" << std::endl;
            }
        }
        else {
            std::cout << optimizer << " is not a valid optimizer!" << std::endl;
            exit(EXIT_FAILURE);
        }

        std::cout << "Learning Complete!" << std::endl;
    }
}