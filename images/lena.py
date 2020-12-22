def lena():
    import pickle
    from os.path import join, dirname
    from numpy import array, float32
    fname = join(dirname(__file__), "lena.dat")
    with open(fname, "rb") as f:
        lena = array(pickle.load(f))
    return lena.astype(float32)


def show_img(img):
    import matplotlib.pyplot as plt
    plt.gray()
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    show_img(lena())
