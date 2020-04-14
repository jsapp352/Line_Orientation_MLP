from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageDraw

import emnist_mlp
from emnist_mlp import NeuralNetwork, NeuronLayer

import numpy as np

_saved_image_file = 'saved_drawing.gif'
_processed_image_file = 'processed_drawing.gif'

_neural_network_file = 'emnist_mlp_UCF_2020_04_09_00_53_01_95p17.pickle'

class TacocatUI(object):
    def __init__(self):
        self.root = Tk()
        # w, h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w,h = 800, 480
        self.root.geometry(f'{w}x{h}+0+0')
        self.root.title("TACOCAT")
        # self.root.config(cursor='none')

        self.network_button = Button(self.root, text='Change Network', command=self.open_network_file, padx=15, pady=12)
        self.network_button.grid(row=1, column=0, rowspan=2, padx=10, pady=5)

        self.clear_button = Button(self.root, text='Clear', command=self.clear, padx=15, pady=12)
        self.clear_button.grid(row=1, column=1, rowspan=2, padx=10, pady=5)

        self.ok_button = Button(self.root, text='OK', command=self.process_image, padx=15, pady=12)
        self.ok_button.grid(row=1, column=2, rowspan=2, padx=10, pady=5)

        self.width = 350
        self.height = 350
        self.input_canvas = Canvas(self.root, bg='white', width=self.width, height=self.height)
        self.input_canvas.grid(row=0, column=0, columnspan=3, padx=20, pady=10)

        self.tacocat_canvas = Canvas(self.root, bg='white', width=self.width, height=self.height)
        self.tacocat_canvas.grid(row=0, column=3, columnspan=5, padx=20, pady=10)

        label_font = ('Courier', 14)

        self.char_text0 = StringVar()
        self.char_label0 = Label(self.root, textvariable=self.char_text0, font=label_font)
        self.char_label0.grid(row=1, column=4, padx=5, pady=0)

        self.char_text1 = StringVar()
        self.char_label1 = Label(self.root, textvariable=self.char_text1, font=label_font)
        self.char_label1.grid(row=1, column=5, padx=5, pady=0)

        self.char_text2 = StringVar()
        self.char_label2 = Label(self.root, textvariable=self.char_text2, font=label_font)
        self.char_label2.grid(row=1, column=6, padx=5, pady=0)

        self.prediction_text0 = StringVar()
        self.prediction_label0 = Label(self.root, textvariable=self.prediction_text0, font=label_font)
        self.prediction_label0.grid(row=2, column=4, padx=5, pady=0)

        self.prediction_text1 = StringVar()
        self.prediction_label1 = Label(self.root, textvariable=self.prediction_text1, font=label_font)
        self.prediction_label1.grid(row=2, column=5, padx=5, pady=0)

        self.prediction_text2 = StringVar()
        self.prediction_label2 = Label(self.root, textvariable=self.prediction_text2, font=label_font)
        self.prediction_label2.grid(row=2, column=6, padx=5, pady=0)

        self.prediction_labels = [self.prediction_label0, self.prediction_label1, self.prediction_label2]
        self.prediction_texts = [self.prediction_text0, self.prediction_text1, self.prediction_text2]

        self.char_labels = [self.char_label0, self.char_label1, self.char_label2]
        self.char_texts = [self.char_text0, self.char_text1, self.char_text2]

        self.default_label_bg = self.char_label0.cget('bg')

        self.placeholder_label = None

        self.image = Image.new("RGB", (self.width, self.height), 'white')
        self.image_draw = ImageDraw.Draw(self.image)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.clear_on_next_paint = False
        self.line_width = 12
        self.color = 'black'
        
        self.input_canvas.bind('<B1-Motion>', self.draw)
        self.input_canvas.bind('<ButtonRelease-1>', self.clear_xy)

        self.previous_x = None
        self.previous_y = None

        self.neural_network_file = _neural_network_file        
        self.initialize_network()
        self.clear()

    def initialize_network(self):
        self.neural_network = emnist_mlp.deserialize_neural_network(self.neural_network_file)

        for idx in range(len(self.neural_network.data_char_set)):
            self.char_texts[idx].set(f'  {self.neural_network.data_char_set[idx]}   ')

    def open_network_file(self):
        self.clear()
        self.neural_network_file = filedialog.askopenfilename(initialdir = "./demo_networks/",title = "Select file")
        self.initialize_network()
    
    def draw_guide_lines(self):
        margin = self.width * 0.15

        lines = []

        lines.append((margin, 0, margin, self.height))
        lines.append((self.width-margin, 0, self.width-margin, self.height))
        lines.append((0, margin, self.width, margin))
        lines.append((0, self.height-margin, self.width, self.height-margin))

        for (x1,y1,x2,y2) in lines:
            self.input_canvas.create_line(x1, y1, x2, y2, width=1, fill='gray')

        
        
    def clear(self):
        self.input_canvas.delete('all')
        self.tacocat_canvas.delete('all')
        self.draw_guide_lines()
        
        for idx, _ in enumerate(self.char_texts):
            # self.char_texts[idx].set('')
            self.prediction_texts[idx].set('')

            self.char_labels[idx].configure(bg = self.default_label_bg)
            self.prediction_labels[idx].configure(bg = self.default_label_bg)

        self.image = Image.new("RGB", (self.width, self.height), 'white')
        self.image_draw = ImageDraw.Draw(self.image)

        # Reset network inputs to reduce power consumption.
        self.neural_network.think([np.zeros((100,), dtype=int)])

    def draw(self, event):
        if self.clear_on_next_paint:
            self.clear()
            self.clear_on_next_paint = False
            return

        if self.previous_x and self.previous_y:
            self.input_canvas.create_line(self.previous_x, self.previous_y, event.x, event.y,
                               width=self.line_width, fill=self.color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
            
            line_points = [(self.previous_x, self.previous_y), (event.x, event.y)]

            self.image_draw.line(line_points, fill=self.color, width=self.line_width, joint='curve')

        self.previous_x = event.x
        self.previous_y = event.y
    
    def process_image(self):
        filename = _saved_image_file
        self.image.save(filename)

        resized_image = self.image.resize((10, 10))
        img_arr = ((255 - np.array(resized_image.convert('L'))) > 40) * 255

        # print(img_arr)

        prediction_result = self.neural_network.think([img_arr.T.flatten()])[-1][0]

        prediction_idx = np.argmax(prediction_result)

        # print(prediction_result)

        prediction_result = (prediction_result + 1) / 2 * 3.3

        for idx, _ in enumerate(self.char_texts):
            self.char_texts[idx].set(f'  {self.neural_network.data_char_set[idx]}   ')
            self.prediction_texts[idx].set(f'{prediction_result[idx]:0.3} V')

            if idx == prediction_idx:
                self.char_labels[idx].config(bg = 'yellow')
                self.prediction_labels[idx].config(bg = 'yellow')
            else:
                self.char_labels[idx].config(bg = self.default_label_bg)
                self.prediction_labels[idx].config(bg = self.default_label_bg)


        resized_image = Image.fromarray(img_arr).resize((self.width, self.height))
        resized_image.save(_processed_image_file)

        tacocat_result = PhotoImage(file=_processed_image_file)
        self.placeholder_label = Label(image=tacocat_result)
        self.placeholder_label.image = tacocat_result

        self.tacocat_canvas.create_image(0, 0, anchor=NW, image=tacocat_result)

        self.clear_on_next_paint = True

    def clear_xy(self, event):
        self.previous_x = None
        self.previous_y = None

if __name__ == '__main__':
    TacocatUI()
