# Names: Prerak Patel, Hsinghui Ku, Zining Ou NetID: pbp73, hk795, zo19
# RUID: 185005168, 203009502, 187007137
from io import StringIO
import math
from os import remove
import random
from abc import abstractproperty
from types import coroutine
from typing import final
import numpy as np
import time
import statistics   
from texttable import Texttable

from numpy.lib.function_base import iterable

max_train_digits = 5000
max_train_face = 451
max_test_face = 150
max_test_digit = 1000
range_max = 10


class Read():
    def read_digits(file_path):
        image = []
        images = []
        counter = 0
        image_string = ""
        file = open(file_path, 'r')
        for line in file:
            count = line.count(' ') + line.count('\n')
            if count == len(line) and len(image_string) > 0:
                image.append(image_string)
                image_string = ""
            elif line.count('+') > 0 or line.count('#') > 0:
                image_string += line
        for i in range(len(image)):
            if image[i].count('\n') <= 5:
                continue
            else:
                images.append(image[i])
        del image
        return images

    def read_faces(file_path):
        image = []
        image_string = ""
        file = open(file_path, 'r')
        i = 1
        for line in file:
            if i % 70 == 0:
                image.append(image_string)
                image_string = ""
            image_string += line
            i += 1
            # print(f"Digits :   Height: {len(image)} Width: {len(image[0])}")
        return image

    def read_labels(file_path):
        try:
            digits = []
            file = open(file_path, 'r')
            for line in file:
                digits.append(line.rstrip())
            return digits
        except IOError:
            print("incorrect file_path provided for training labels")

class Features():
    def percentage_filled(string):
        full = 0
        total = 1
        for i in range(len(string)):
            if string[i] == '+':
                full += .5
            if string[i] == '#':
                full += 1
            total += 1
        percentage = full/total
        return percentage

    def string_to_matrix(image):
        matrix = []
        split_string = image.split('\n')
        for i in range(len(split_string)):
            matrix.append([])
            matrix[i] = list(split_string[i])
        
        while(len(matrix) < 71):
            matrix.append([])
            matrix[len(split_string)-1].append(''.ljust(60))
        return matrix


    def string_to_matrix2(image):
        matrix = np.zeros((70, 60))
        split_string = image.split('\n')
        
        for str in split_string:
            i = 0
            for j in range(len(str)):
                matrix[i][j] = 0 if str[j] == '#' else 1
            i += 1
        
        return matrix

    def string_to_matrix3(image):
        matrix = []
        split_string = image.split('\n')
        for i in range(len(split_string)):
            matrix.append([])
            matrix[i] = list(split_string[i])
            if len(matrix[i]) < 28:
                matrix[i].append(''.ljust(28 - len(matrix[i])))
            
        
        while(len(matrix) < 21):
            matrix.append([])
            matrix[len(split_string)-1].append(''.ljust(28))
        return matrix

    def quadrant_one(image):
        mat = Features.string_to_matrix(image)
        quadrant_one = ""
        for i in range(len(mat)//2):
            for j in range(len(mat[0])//2):
                quadrant_one = quadrant_one+mat[i][j]
        return Features.percentage_filled(quadrant_one)

    def quadrant_two(image):
        mat = Features.string_to_matrix(image)
        quadrant_two = ""
        for i in range(len(mat)//2):
            for j in range(len(mat[0])//2, len(mat[0])):
                quadrant_two = quadrant_two+mat[i][j]
        return Features.percentage_filled(quadrant_two)

    def quadrant_three(image):
        mat = Features.string_to_matrix(image)
        quadrant_three = ""
        for i in range(len(mat)//2, len(mat)-1):
            for j in range(len(mat[0])//2):
                quadrant_three = quadrant_three+mat[i][j]
        return Features.percentage_filled(quadrant_three)

    def quadrant_four(image):
        mat = Features.string_to_matrix(image)
        quadrant_four = ""
        for i in range(len(mat)//2, len(mat)-1):
            for j in range(len(mat[0])//2, len(mat[0])):
                quadrant_four = quadrant_four+mat[i][j]
        return Features.percentage_filled(quadrant_four)

    def top_heavy(image):
        mat = Features.string_to_matrix3(image)
        top_half = ""
        for i in range(len(mat)//2):
            for j in range(len(mat[0])):
                top_half += mat[i][j]
        retval = 1 if Features.percentage_filled(top_half) > .7 else 0
        return retval

    def get_features(image):
        mat = Features.string_to_matrix3(image)
        feat = [1]
        for i in range(len(mat)-1):
            for j in range(len(mat[i])):
                # print(f"{i}  {j}")
                if mat[i][j] == '#' or mat[i][j] == '+':
                    feat.append(1)
                else:
                    feat.append(0)
        top_hvy = Features.top_heavy(image)
        feat.append(top_hvy)
        while (len(feat) < 588):
            feat.append(0)
        return feat[:588]

    def get_features_nb(image):
        mat = Features.string_to_matrix3(image)
        feat = []
        for i in range(len(mat)-1):
            for j in range(len(mat[i])):
                if mat[i][j] == '#' or mat[i][j] == '+':
                    feat.append(1)
                else:
                    feat.append(0)

        top_hvy = Features.top_heavy(image)
        feat.append(top_hvy)
        while (len(feat) < 588):
            feat.append(0)
        return feat[:588]

    def get_face_features(image):
        mat = Features.string_to_matrix(image)
        feat = [1]
        for i in range(len(mat)-1):
            for j in range(len(mat[i])):
                if mat[i][j] == '#':
                    feat.append(1)
                else:
                    feat.append(0)
        while (len(feat) < 4200):
            feat.append(0)
        return feat[:4200]

    def get_face_features_nb(image):
        mat = Features.string_to_matrix(image)
        feat = []
        for i in range(len(mat)-1):
            for j in range(len(mat[i])):
                if mat[i][j] == '#':
                    feat.append(1)
                else:
                    feat.append(0)
        while (len(feat) < 4200):
            feat.append(0)
        return feat[:4200]

    def get_features_knn(image):
        mat = Features.string_to_matrix3(image)
        feat = []
        for i in range(len(mat)-1):
            for j in range(len(mat[i])):
                if mat[i][j] == '#' or mat[i][j] == '+':
                    feat.append(1)
                else:
                    feat.append(0)
        while (len(feat) < 588):
            feat.append(0)
        return feat

    def get_face_features_knn(image):
        mat = Features.string_to_matrix(image)



        feat = []
        for i in range(len(mat)-1):
            for j in range(len(mat[0])):
                if mat[i][j] == '#':
                    feat.append(1)
                    # print(mat[i][j],end="")
                else:
                    feat.append(0)
                    # print(mat[i][j],end="")
        #     print()

        # print(f"{len(mat)}     {len(mat[0])}")
        while (len(feat) < 4200):
            feat.append(0)
        return feat

    def get_face_features_knn2(image):
        mat = Features.string_to_matrix2(image)
        split_test = Features.split(mat, 14, 12)
        perc_fill = []

        for i in range(len(split_test)):
            count = 0
            for j in split_test[i]:
                count += j
                if j != 0 :
                    count += 1
            perc_fill.append(count / 182.0)

        return perc_fill
        # for i in range(25):
        #     print(perc_fill[i])   



    def split(array, nrows, ncols):
        """Split a matrix into sub-matrices."""
        sub_mat = [[] for i in range(25)]
        for i in range(len(array)):
            for j in range(len(array[0])):
                # print(f"{i} {j}   {int(i / 5)+int(j / 5)}")
                sub_mat[int(i / 5)+int(j / 5)].append(array[i][j])

        return sub_mat

class Perceptron():
    def initialize_weights(w_map, digits, num_features):
        for i in range(digits):
            w_map[str(i)] = [random.random() for _ in range(num_features)]
            # w_map[str(i)].append(1)
        return w_map
    def weight_adjustments(multi_classes, predicted_value, real_value, features):
        if predicted_value != real_value:
            for i in range(len(features)):
                multi_classes[predicted_value][i] = multi_classes[predicted_value][i]-features[i]
            for i in range(len(features)):
                multi_classes[real_value][i] = multi_classes[real_value][i]+features[i]
    def face_weight_adjustments(weights_list, predicted_value, real_value, features):
        if predicted_value > 0 and real_value == '0':
            for i in range(len(weights_list)):
                weights_list[i] = weights_list[i]-features[i]
        elif predicted_value < 0 and real_value == '1':
            for i in range(len(weights_list)):
                weights_list[i] = weights_list[i]+features[i]

    
    def train(data_set, models):
        if data_set == 'digits':
            digit_weights = {'0': [], '1': [], '2': [], '3': [],
                            '4': [], '5': [], '6': [], '7': [], '8': [], '9': []}
            len_features = len(Features.get_features(models[0].image))
            digit_weights = Perceptron.initialize_weights(
                digit_weights, 10, len_features)
            function_map = {'0': 0, '1': 0, '2': 0, '3': 0,
                            '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
            for i in range(len(models)):
                features = Features.get_features(models[i].image)
                for digit, weights in digit_weights.items():
                    func_val = 0
                    for w in range(len(features)):
                        func_val += features[w]*weights[w]
                    function_map[digit] = func_val
                prediction = max(function_map, key=function_map.get)
                Perceptron.weight_adjustments(
                    digit_weights, prediction, models[i].label, features)
            return digit_weights
        else:
            len_features = len(Features.get_face_features(models[0].image))
            face_weights = [random.random() for _ in range(len_features)]
            for i in range(len(models)):
                func_val = 0
                features = Features.get_face_features(models[i].image)
                for w in range(len(face_weights)):
                    func_val += face_weights[w]*features[w]
                Perceptron.face_weight_adjustments(
                    face_weights, func_val, models[i].label, features)
            return face_weights
    def test(data_set, train, test):
        weights_dict = Perceptron.train(data_set, train)
        if data_set == 'digits' or data_set == 'digit':
            correct = 0
            function_map = {'0': 0, '1': 0, '2': 0, '3': 0,
                            '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
            for i in range(len(test)):
                features = Features.get_features(test[i].image)
                for digit, weights in weights_dict.items():
                    func_val = 0
                    for w in range(len(features)):
                        func_val += features[w]*weights[w]
                    func_val += weights[-1]
                    function_map[digit] = func_val
                prediction = max(function_map, key=function_map.get)
                if prediction == test[i].label:
                    # print('Correct: ', correct, ' Prediction:',
                    #       prediction, " Actual:", labels[i])
                    correct += 1
        else:
            correct = 0
            for i in range(len(test)):
                func_val = 0
                prediction = '0'
                features = Features.get_face_features(test[i].image)
                for w in range(len(weights_dict)):
                    func_val += weights_dict[w]*(features[w])
                if func_val > 0:
                    prediction = '1'
                elif func_val <= 0:
                    prediction = '0'
                if str(prediction) == test[i].label:
                    correct += 1
        
        return correct/len(test)


class Bayes():
    def clean_changes(zero, one, two, three, four, five, six, seven, eight, nine, prior_dict):
        for key, val in zero.items():
            for i in range(len(val)):
                if val[i] == 0:
                    val[i] = .01
                else:
                    val[i] = val[i]/prior_dict['0']
        for key, val in one.items():
            for i in range(len(val)):
                if val[i] == 0:
                    val[i] = .01
                else:
                    val[i] = val[i]/prior_dict['1']
        for key, val in two.items():
            for i in range(len(val)):
                if val[i] == 0:
                    val[i] = .01
                else:
                    val[i] = val[i]/prior_dict['2']
        for key, val in three.items():
            for i in range(len(val)):
                if val[i] == 0:
                    val[i] = .01
                else:
                    val[i] = val[i]/prior_dict['3']
        for key, val in four.items():
            for i in range(len(val)):
                if val[i] == 0:
                    val[i] = .01
                else:
                    val[i] = val[i]/prior_dict['4']
        for key, val in five.items():
            for i in range(len(val)):
                if val[i] == 0:
                    val[i] = .01
                else:
                    val[i] = val[i]/prior_dict['5']
        for key, val in six.items():
            for i in range(len(val)):
                if val[i] == 0:
                    val[i] = .01
                else:
                    val[i] = val[i]/prior_dict['6']
        for key, val in seven.items():
            for i in range(len(val)):
                if val[i] == 0:
                    val[i] = .01
                else:
                    val[i] = val[i]/prior_dict['7']
        for key, val in eight.items():
            for i in range(len(val)):
                if val[i] == 0:
                    val[i] = .01
                else:
                    val[i] = val[i]/prior_dict['8']
        for key, val in nine.items():
            for i in range(len(val)):
                if val[i] == 0:
                    val[i] = .01
                else:
                    val[i] = val[i]/prior_dict['9']

    def clean_prediction(p_zero, p_one, p_two, p_three, p_four, p_five, p_six, p_seven, p_eight, p_nine):
        max_p = max([p_zero, p_one, p_two, p_three, p_four,
                    p_five, p_six, p_seven, p_eight, p_nine])
        prediction = ''
        if max_p == p_zero:
            prediction = '0'
        elif max_p == p_one:
            prediction = '1'
        elif max_p == p_two:
            prediction = '2'
        elif max_p == p_three:
            prediction = '3'
        elif max_p == p_four:
            prediction = '4'
        elif max_p == p_five:
            prediction = '5'
        elif max_p == p_six:
            prediction = '6'
        elif max_p == p_seven:
            prediction = '7'
        elif max_p == p_eight:
            prediction = '8'
        elif max_p == p_nine:
            prediction = '9'

        return prediction


    def train_digits(models):
        prior_dict = {}
        for i in range(10):
            prior_dict[str(i)] = 1

        for i in range(len(models)):
            prior_dict[models[i].label] += 1

        length = len(Features.get_features_nb(models[0].image))



        zero = {i: [0, 0] for i in range(length)}
        one = {i: [0, 0] for i in range(length)}
        two = {i: [0, 0] for i in range(length)}
        three = {i: [0, 0] for i in range(length)}
        four = {i: [0, 0] for i in range(length)}
        five = {i: [0, 0] for i in range(length)}
        six = {i: [0, 0] for i in range(length)}
        seven = {i: [0, 0] for i in range(length)}
        eight = {i: [0, 0] for i in range(length)}
        nine = {i: [0, 0] for i in range(length)}

        for i in range(len(models)):
            features = Features.get_features_nb(models[i].image)
            if models[i].label == '0':
                for j in range(len(features)):
                    if features[j] == 1:
                        zero[j][1] += 1
                    else:
                        zero[j][0] += 1
            elif models[i].label == '1':
                for j in range(len(features)):
                    if features[j] == 1:
                        one[j][1] += 1
                    else:
                        one[j][0] += 1
            elif models[i].label == '2':
                for j in range(len(features)):
                    if features[j] == 1:
                        two[j][1] += 1
                    else:
                        two[j][0] += 1
            elif models[i].label == '3':
                for j in range(len(features)):
                    if features[j] == 1:
                        three[j][1] += 1
                    else:
                        three[j][0] += 1
            elif models[i].label == '4':
                for j in range(len(features)):
                    if features[j] == 1:
                        four[j][1] += 1
                    else:
                        four[j][0] += 1
            elif models[i].label == '5':
                for j in range(len(features)):
                    if features[j] == 1:
                        five[j][1] += 1
                    else:
                        five[j][0] += 1
            elif models[i].label == '6':
                for j in range(len(features)):
                    if features[j] == 1:
                        six[j][1] += 1
                    else:
                        six[j][0] += 1
            elif models[i].label == '7':
                for j in range(len(features)):
                    if features[j] == 1:
                        seven[j][1] += 1
                    else:
                        seven[j][0] += 1
            elif models[i].label == '8':
                for j in range(len(features)):
                    if features[j] == 1:
                        eight[j][1] += 1
                    else:
                        eight[j][0] += 1
            elif models[i].label == '9':
                for j in range(len(features)):
                    if features[j] == 1:
                        nine[j][1] += 1
                    else:
                        nine[j][0] += 1
        Bayes.clean_changes(zero, one, two, three, four, five,
                            six, seven, eight, nine, prior_dict)

        return prior_dict, zero, one, two, three, four, five, six, seven, eight, nine


    def train_faces(models):
        prior_dict = {}
        for i in range(len(models)):
            if models[i].label in prior_dict:
                prior_dict[models[i].label] += 1
            else:
                prior_dict[models[i].label] = 1
        face_training_dict = {i: [0, 0] for i in range(60*70)}
        not_face_training_dict = {i: [0, 0] for i in range(60*70)}
        for i in range(len(models)):
            if models[i].label == '1':
                features = Features.get_face_features_nb(models[i].image)
                for i in range(len(features)):
                    if features[i] == 1:
                        face_training_dict[i][1] += 1
                    else:
                        face_training_dict[i][0] += 1
            elif models[i].label == '0':
                features = Features.get_face_features_nb(models[i].image)
                for i in range(len(features)):
                    if features[i] == 1:
                        not_face_training_dict[i][1] += 1
                    else:
                        not_face_training_dict[i][0] += 1
        for key, val in face_training_dict.items():
            for i in range(len(val)):
                if val[i] == 0:
                    val[i] = .01
                else:
                    val[i] = val[i]/prior_dict['1']
        for key, val in not_face_training_dict.items():
            for i in range(len(val)):
                if val[i] == 0:
                    val[i] = .01
                else:
                    val[i] = val[i]/prior_dict['0']

        return prior_dict, face_training_dict, not_face_training_dict

    def test_digits(train, test):
        prior, zero,one,two,three,four,five,six,seven,eight,nine= Bayes.train_digits(train)
        correct = 0
        n = len(train)
        for j in range(len(test)):
            p_zero = math.log(prior['0']/n)
            p_one = math.log(prior['1']/n)
            p_two = math.log(prior['2']/n)
            p_three = math.log(prior['3']/n)
            p_four = math.log(prior['4']/n)
            p_five = math.log(prior['5']/n)
            p_six = math.log(prior['6']/n)
            p_seven = math.log(prior['7']/n)
            p_eight = math.log(prior['8']/n)
            p_nine = math.log(prior['9']/n)
            features = Features.get_features_nb(test[j].image)
            for i in range(len(features)):
                p_zero += math.log(zero[i][features[i]])
                p_one += math.log(one[i][features[i]])
                p_two += math.log(two[i][features[i]])
                p_three += math.log(three[i][features[i]])
                p_four += math.log(four[i][features[i]])
                p_five += math.log(five[i][features[i]])
                p_six += math.log(six[i][features[i]])
                p_seven += math.log(seven[i][features[i]])
                p_eight += math.log(eight[i][features[i]])
                p_nine += math.log(nine[i][features[i]])
            prediction = Bayes.clean_prediction(
                p_zero, p_one, p_two, p_three, p_four, p_five, p_six, p_seven, p_eight, p_nine)
            if prediction == test[j].label:
                # print(prediction, labels[j], 'CORRECT', correct, test_amount)
                correct += 1
        return correct/len(test)

    def test_faces(train, test):
        n = len(train)
        prior, face_probs, not_face_probs=Bayes.train_faces(train)
        correct = 0
        for j in range(len(test)):
            p_face = math.log(prior['1']/n)
            p_not_face = math.log(prior['0']/n)
            features = Features.get_face_features_nb(test[j].image)
            for i in range(len(features)):
                p_face += math.log(face_probs[i][features[i]])
                p_not_face += math.log(not_face_probs[i][features[i]])
            prediction = ''
            if p_face > p_not_face:
                prediction = '1'
            else:
                prediction = '0'
            if prediction == test[j].label:
                correct += 1
                # print(prediction, labels[j], 'CORRECT', correct, test_amount)
            # else:
            #     print(prediction, labels[j], 'Incorrect', correct, test_amount)
        return correct/ len(test)


class KNN():
    def classify(test, data, labels, k=5):
        diff = test-data
        dist = (diff ** 2).sum(axis=1) ** 0.5
        sort_dist = dist.argsort()
        knn = {}

        for i in range(k):
            label2 = int(labels[sort_dist[i]])
            knn[label2] = knn.get(label2, 0) + 1

        max_num = 0
        res = 0
        for i in knn:
            if knn[i] > max_num:
                max_num = knn[i]
                res = i
        return res

    def test(data_set, train, test):
        correct = 0

        labels = []

        for l in train:
            labels.append(l.label)

        # TEST DIGITS
        if data_set == 'digits' or data_set == 'digit':
            train_feat = np.zeros((len(train), 588))
            for i in range(len(train)):
                train_feat[i] = Features.get_features_knn(train[i].image)

            for i in range(len(test)):
                test_features = Features.get_features_knn(test[i].image)
                prediction = KNN.classify(test_features, train_feat, labels)
                # print(' Prediction:', prediction, " Actual:", test_labels[i])
                if prediction == int(test[i].label):
                    correct += 1
        # TEST FACE DATA SET
        else:
            train_feat = np.zeros((len(train), 25))
            for i in range(len(train)):
                train_feat[i] = Features.get_face_features_knn2(train[i].image)

            for i in range(len(test)):
                test_features = Features.get_face_features_knn2(test[i].image)
                prediction = KNN.classify(test_features, train_feat, labels)
                if prediction == int(test[i].label):
                    correct += 1
            
        return correct/len(test)

















max_train_digits = 5000
max_train_face = 451
max_test_face = 150
max_test_digit = 1000
range_max = 10
iteration = 5




# Features.get_face_features_knn2(images[0])



class Model:
    def __init__(self, image, label) -> None:
        self.image = image
        self.label = label
        self.height = len(self.image)
        self.width = 0
        for i in range(len(self.image)):
            self.width = len(self.image[i]) if len(self.image[i]) > self.width else self.width

                


faces = []
labels = Read.read_labels('facedata/facedatatrainlabels')
images = Read.read_faces('facedata/facedatatrain')

for i in range(max_train_face):
    faces.append(Model(images[i],labels[i]))

    
digits = []
images = Read.read_digits('digitdata/trainingimages')
labels = Read.read_labels('digitdata/traininglabels') 
for i in range(max_train_digits):
    digits.append(Model(images[i],labels[i]))


test_faces = []
images = Read.read_faces('facedata/facedatatest')
labels = Read.read_labels('facedata/facedatatestlabels')
for i in range(max_test_face):
    test_faces.append(Model(images[i],labels[i]))


test_digits = []
images = Read.read_digits('digitdata/testimages')
labels = Read.read_labels('digitdata/testlabels')
for i in range(max_test_digit):
    test_digits.append(Model(images[i],labels[i]))

# train_digit = random.choices(digits, k = 5)
# train_face = random.choices(faces, k = 5)
# Perceptron.test('digits', train_digit, test_d)
# Perceptron.test('face', train_face, test_f)
# KNN.test('digits', train_digit, test_d)
# KNN.test('face', train_face, test_f)
# Bayes.test_digits(train_digit, test_d)
# Bayes.test_faces(train_face, test_f)

def report_with_train_percent(percent):

    
        
    perception_face = []
    perception_digits = []
    bayes_face = []
    bayes_digits = []
    knn_face = []
    knn_digits = []

    p_time = [0,0]
    b_time = [0,0]
    k_time = [0,0]
    
    # rand_digits(label, image)
    print(f"\n\nTest on {percent * 100} % of data points:")
    for i in range(5):
        train_digit = random.choices(digits, k = int(percent * max_train_digits))
        train_face = random.choices(faces, k = int(percent * max_train_face))
        
        # Perception
        start_time = time.time()
        res_digit = Perceptron.test('digits', train_digit, test_digits)
        p_time[0] += time.time()-start_time

        start_time = time.time()
        res_face = Perceptron.test('face', train_face, test_faces)
        p_time[1] += time.time()-start_time
        perception_face.append(res_face)
        perception_digits.append(res_digit)
        

        # Bayes
        start_time = time.time()
        res_digit = Bayes.test_digits(train_digit, test_digits)
        b_time[0] += time.time()-start_time

        start_time = time.time()
        res_face = Bayes.test_faces(train_face, test_faces)
        b_time[1] += time.time()-start_time

        bayes_face.append(res_face)
        bayes_digits.append(res_digit)
        


        #KNN
        start_time = time.time()
        res_digit = KNN.test('digits', train_digit, test_digits)
        k_time[0] += time.time()-start_time

        start_time = time.time()
        res_face = KNN.test('face', train_face, test_faces)
        k_time[1] += time.time()-start_time

        knn_face.append(res_face)
        knn_digits.append(res_digit)

    print(f"{'Algorithm'.ljust(25)}|{'Data Type'.ljust(15)}|   1   |   2   |   3   |   4   |   5   | {'Mean'.ljust(6)}| {'Stdv'.ljust(5)}|Time |")
    print('-------------------------+---------------+-------+-------+-------+-------+-------+-------+------+-----+')
    print(format_report('Perception','Faces', perception_face,p_time[1]))
    print(format_report('Perception','Digits', perception_digits,p_time[0]))        
    print(format_report('Naive Bayes','Face', bayes_face,b_time[1]))
    print(format_report('Naive Bayes','Digits', bayes_digits,b_time[0]))
    print(format_report('K-Nearest Neighbor','Face', knn_face,k_time[1]))
    print(format_report('K-Nearest Neighbor','Digits', knn_digits,k_time[0])) 

    p = (statistics.mean(perception_face), statistics.stdev(perception_face),statistics.mean(perception_digits), statistics.stdev(perception_digits))
    b = (statistics.mean(bayes_face), statistics.stdev(bayes_face),statistics.mean(bayes_digits), statistics.stdev(bayes_digits))
    k = (statistics.mean(knn_face), statistics.stdev(knn_face),statistics.mean(knn_digits), statistics.stdev(knn_digits))   
    return p,b,k

def format_report(alg, data, res, time):
    report = f"{alg.ljust(25)}|{data.ljust(15)}|"
    for i in range(iteration):
        report += f" {res[i]*100:.2f}%|"
    report += f" {statistics.mean(res)*100:.2f}%| {statistics.stdev(res):.2f} |{time:5.2f}|"
    return report

def report():
    
    for i in range(range_max):
        percent = (i+1) / range_max
        p,b,k = report_with_train_percent(percent)
report()
