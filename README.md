# mnist_cnn
图像分类hello world实践

# input\_mnist.py
读取mnist数据集

x_train, y_train = load\_data(train_data_file, train_label_file)

x_train: (60000, 28, 28, 1) # 已经reshape过了，可以直接输入给模型

y_train: (60000,)

# mnist\_cnn.py
cnn分类器
