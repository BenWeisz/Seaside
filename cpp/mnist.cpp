#include <iostream>
#include <ctime>
#include <cstdlib>

#include "net.h"

using namespace Seaside;

std::vector<Mat> prepare_dataset(const char *images, const char *labels);
void Fread(void *ptr, size_t size, size_t count, FILE *stream);
int swap_endian(int num);
double check_accuracy(Mat output, Mat correct_output);

Vec normalize_relu(Vec v);
Vec normalize_tanh(Vec v);

int main(){
    // Randomize
    std::srand(std::time(NULL));

    // Prepare the dataset
    auto data_set = prepare_dataset("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
    
    // Normalize the input values
    Mat input_data = data_set[0];
    input_data = input_data.map(normalize_relu);

    Mat target_data = data_set[1];

    Net mnist_model({784, 16, 10}, {"relu", "soft_max"});

    // Train the model
    mnist_model.learn("xent", input_data, target_data, 0.01, 5);

    // Test the model's preformance
    auto test_set = prepare_dataset("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
    Mat test_input = test_set[0];
    test_input = test_input.map(normalize_relu);

    Mat test_output = mnist_model.query(test_input);

    std::cout << "Accuracy: " << check_accuracy(test_output, test_set[1]) << "%" << std::endl;

    mnist_model.save("mnist_model.b");

    return 0;
}

double check_accuracy(Mat output, Mat correct_output){
    int correct = 0;

    auto dim = output.dim();
    for (int i = 0; i < dim.second; i++){
        int max_index = 0;
        double max_val = -10;

        for (int j = 0; j < 10; j++){
            if (output[i][j] > max_val){
                max_index = j;
                max_val = output[i][j];
            }
        }

        int correct_index = 0;
        for (int j = 0; j < 10; j++){
            if (correct_output[i][j] == 1.0){
                correct_index = j;
                break;
            }
        }

        if (max_index == correct_index)
            correct++;
    }

    return (correct + 0.0) / (100.0);
}

std::vector<Mat> prepare_dataset(const char *images_file_name, const char *labels_file_name){
    std::cout << "Started loading data set:" << std::endl;
    std::cout << "\tImages: " << images_file_name << std::endl;
    std::cout << "\tLabels: " << labels_file_name << std::endl;

    std::vector<Mat> data_set;

    FILE *images_file = fopen(images_file_name, "rb");
    if (images_file == nullptr){
        std::cout << "FOPEN: The images file could not be opened!" << std::endl;
        exit(1);
    }

    int magic_number;
    Fread(&magic_number, sizeof(int), 1, images_file);

    magic_number = swap_endian(magic_number);

    if (magic_number != 2051){
        std::cout << "The images file is not a proper MNIST images file!" << std::endl;
        exit(1);
    }

    int num_images;
    Fread(&num_images, 4, 1, images_file);
    num_images = swap_endian(num_images);

    int width, height;
    Fread(&height, 4, 1, images_file);
    Fread(&width, 4, 1, images_file);

    width = swap_endian(width);
    height = swap_endian(height);

    std::vector<Vec> image_cols;
    for (int i = 0; i < num_images; i++){
        std::vector<double> col_values;
        
        for (int xy = 0; xy < width * height; xy++){
            unsigned char byte_value;
            Fread(&byte_value, 1, 1, images_file);

            col_values.push_back((double)byte_value);
        }

        Vec col(col_values);
        image_cols.push_back(col);
    }

    Mat images(0, 0);
    images.set_columns(image_cols);

    data_set.push_back(images);

    if (fclose(images_file) != 0){
        std::cout << "FCLOSE: Images file could not be closed!" << std::endl;
        exit(1);
    }

    // Image Reading is done, Notify user.
    std::cout << "Read " << num_images << " images from: " << images_file_name << std::endl; 

    FILE *labels_file = fopen(labels_file_name, "rb");
    if (labels_file == nullptr){
        std::cout << "FOPEN: The labels file could not be opened!" << std::endl;
        exit(1);
    }

    Fread(&magic_number, sizeof(int), 1, labels_file);
    magic_number = swap_endian(magic_number);

    if (magic_number != 2049){
        std::cout << "The labels file is not a proper MNIST labels file!" << std::endl;
        exit(1);
    }

    int num_labels;
    Fread(&num_labels, sizeof(int), 1, labels_file);
    num_labels = swap_endian(num_labels);

    std::vector<Vec> label_cols;
    for (int i = 0; i < num_labels; i++){
        unsigned char val;
        Fread(&val, 1, 1, labels_file);

        Vec col({});
        col.set(10, 0.0);
        
        col[(int)val] = 1.0;

        label_cols.push_back(col);
    }

    Mat labels(0, 0);
    labels.set_columns(label_cols);

    data_set.push_back(labels);

    if (fclose(labels_file) != 0){
        std::cout << "FCLOSE: Labels file could not be closed!" << std::endl;
        exit(1);
    }

    // Labels Reading is done, Notify user.
    std::cout << "Read " << num_labels << " labels from: " << labels_file_name << std::endl; 

    return data_set; 
 }

void Fread(void *ptr, size_t size, size_t count, FILE *stream){
    if (fread(ptr, size, count, stream) == 0){
        std::cout << "FREAD: Data could not be read!" << std::endl;
        exit(1); 
    }
}

int swap_endian(int num){
    int swapped_num = ((num >> 24) &0xff)     |
                      ((num << 8)  &0xff0000) |
                      ((num >> 8)  &0xff00)   |
                      ((num << 24) &0xff000000);

    return swapped_num;
}

Vec normalize_relu(Vec v){
    for (int i = 0; i < v.len(); i++)
        v[i] = v[i] / 255.0;

    return v;
}

Vec normalize_tanh(Vec v){
    for (int i = 0; i < v.len(); i++)
        v[i] = ((v[i] / 255.0) * 2.0) - 1.0;

    return v;
}