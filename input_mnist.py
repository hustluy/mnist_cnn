
import struct
import sys
import numpy as np

def load_data(data_file):
    file_handler = open(data_file, 'rb')
    file_info_buffer = file_handler.read(4 * 4)
    magic, image_n, pixel_row_n, pixel_col_n = struct.unpack_from('>IIII', file_info_buffer)
    data = np.empty((image_n, pixel_row_n, pixel_col_n, 1), dtype = 'uint8')
    for i in range(image_n):
        pixels_buffer = file_handler.read(pixel_row_n * pixel_col_n)
        pixels = struct.unpack_from('>' + pixel_row_n * pixel_col_n * 'B', pixels_buffer)
        data[i] = np.asarray(pixels).reshape(pixel_row_n, pixel_col_n, 1)
    return data

def load_label(label_file):
    file_handler = open(label_file, 'rb')
    file_info_buffer = file_handler.read(2 * 4)
    magic, image_n = struct.unpack_from('>II', file_info_buffer)
    labels = np.empty(image_n, dtype = 'uint8')
    for i in range(image_n):
        labels[i] = struct.unpack_from('>B', file_handler.read(1))[0]
    return labels

def load(data_file, label_file):
    data = load_data(data_file)
    labels = load_label(label_file)
    return data, labels
