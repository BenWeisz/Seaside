from seaside import Vec, Mat, Net

if __name__ == '__main__':
    neural_net = Net([2, 3, 2], ['relu', 'softmax'])

    input_data = Mat(2, 8)
    input_data.data = [Vec([3, 1.5]), Vec([2, 1]), Vec([4, 1.5]), Vec([3, 1]), Vec([3.5, 0.5]), Vec([2, 0.5]), Vec([5.5, 1]), Vec([1, 1])]

    target_data = Mat(2, 8)
    target_data.data = [Vec([1, 0]), Vec([0, 1]), Vec([1, 0]), Vec([0, 1]), Vec([1, 0]), Vec([0, 1]), Vec([1, 0]), Vec([0, 1])]

    # neural_net.learn('xent', input_data, target_data, 0.1, 10000)

    # neural_net.save('flowers_xent_relu.txt')
    neural_net.load('flowers_xent_relu.txt')

    test_input = Vec([4.5, 1]).to_mat()
    test_output = neural_net.query(input_data)
    test_output.precision = 5
    print(test_output)