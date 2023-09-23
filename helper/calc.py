from math import floor

def output_dim(h_in, kernel_size, stride = 1, dilation = 1, padding = 0):
    return floor(((h_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)

h_in1 = output_dim(224, 3)
mp_1 = output_dim(h_in1, 2, stride=2)
h_in2 = output_dim(mp_1, 3)
mp_2 = output_dim(h_in2, 2, stride=2)
h_in3 = output_dim(mp_2, 3)
mp_3 = output_dim(h_in3, 2, stride=2)
print(mp_3)