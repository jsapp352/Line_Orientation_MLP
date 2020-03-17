#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 21:49:12 2018

@author: kaustabh
"""

from tkinter import *
from PIL import Image, ImageDraw

import emnist_mlp
from emnist_mlp import NeuralNetwork, NeuronLayer

import numpy as np

_saved_image_file = 'saved_drawing.gif'
_processed_image_file = 'processed_drawing.gif'

_neural_network_file = 'emnist_mlp_UCF_2020_03_14_16_03_12.pickle'

class TacocatUI(object):
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()

        self.clear_button = Button(self.root, text='Clear', command=self.clear)
        self.clear_button.grid(row=1, column=0)

        self.ok_button = Button(self.root, text='OK', command=self.process_image)
        self.ok_button.grid(row=1, column=1)

        self.width = 300
        self.height = 300
        self.drawing_canvas = Canvas(self.root, bg='white', width=self.width, height=self.height)
        self.drawing_canvas.grid(row=0, column=0, columnspan=2)

        self.tacocat_canvas = Canvas(self.root, bg='white', width=self.width, height=self.height)
        self.tacocat_canvas.grid(row=0, column=2)

        self.prediction_text = StringVar()
        self.prediction_label = Label(self.root, textvariable=self.prediction_text)
        self.prediction_label.grid(row=1, column=2)

        self.placeholder_label = None

        self.image = Image.new("RGB", (self.width, self.height), 'white')
        self.draw = ImageDraw.Draw(self.image)

        self.neural_network = self.initialize_network()


        self.setup()
        self.root.mainloop()

    def initialize_network(self):
        return emnist_mlp.deserialize_neural_network(_neural_network_file)

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = 12
        self.color = self.DEFAULT_COLOR
        self.active_button = self.ok_button
        self.drawing_canvas.bind('<B1-Motion>', self.paint)
        self.drawing_canvas.bind('<ButtonRelease-1>', self.reset)

    def clear(self):
        self.drawing_canvas.delete('all')
        self.tacocat_canvas.delete('all')
        self.prediction_text.set('')

        self.image = Image.new("RGB", (self.width, self.height), 'white')
        self.draw = ImageDraw.Draw(self.image)

    def activate_button(self, some_button):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button

    def paint(self, event):
        if self.old_x and self.old_y:
            self.drawing_canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=self.color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)

            self.draw.line([(self.old_x, self.old_y), (event.x, event.y)], fill=self.color, width=self.line_width, joint='curve')

        self.old_x = event.x
        self.old_y = event.y
    
    def process_image(self):
        filename = _saved_image_file
        self.image.save(filename)

        resized_image = self.image.resize((10, 10))
        img_arr = ((255 - np.array(resized_image.convert('L'))) > 40) * 255

        print(img_arr)

        prediction_result = self.neural_network.think([img_arr.T.flatten()])[-1]

        self.prediction_text.set(f'TACOCAT prediction: {self.neural_network.data_char_set[np.argmax(prediction_result)]}')

        resized_image = Image.fromarray(img_arr).resize((300, 300))
        resized_image.save(_processed_image_file)

        tacocat_result = PhotoImage(file=_processed_image_file)
        self.placeholder_label = Label(image=tacocat_result)
        self.placeholder_label.image = tacocat_result

        self.tacocat_canvas.create_image(0, 0, anchor=NW, image=tacocat_result)

    def reset(self, event):
        self.old_x, self.old_y = None, None

if __name__ == '__main__':
    TacocatUI()