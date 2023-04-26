import sys
import re
import numpy as np

# print("-----------------------------------------------")
# for i in sys.path:
#     print(i)
from src.Helpers import pickleOpen, open_file, encode

data = open_file("dfts-v1.csv")
encoders = pickleOpen("encoders")

i = "Hotel_Country"
x = encode(data[[i]], encoders["Hotel_Country"], testing=True, label=False)

print(x)
