import os

Root = "../input/multilabel_images/cmb_6_7/"
i = 6000
for f in sorted(os.listdir(Root)):
    os.rename(f"{Root}" + str(f), f"{Root}" + f"image_{i}.png")
    i = i + 1
