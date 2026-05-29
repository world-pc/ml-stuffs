from keras.datasets import mnist
import curses
import time

#load in the training and testing data from MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

stdscr = curses.initscr()

counter = 1
for image in x_train:
    stdscr.clear()

    for row in range(0, len(image)):
        for pixel in range(0, len(image[row])):
            if image[row][pixel] > 0:
                stdscr.addstr(row, pixel, '@')

    stdscr.addstr(40, 0, f'image {counter}/{len(x_train)}')

    stdscr.refresh()

    counter += 1
    time.sleep(.025)

stdscr.getch()
