"""
Script for going through every image of a directory and
moving images to a makred_dir to remove them from the set. 
"""

import os
import shutil
import tkinter as tk
from PIL import Image, ImageTk

class ManualClassifier:
    def __init__(self, shot_dir, class_dir):
        self.shot_dir = shot_dir
        self.class_dir = class_dir
        # get generator object to loop through files of shot dir
        self.images = [f for f in os.listdir(self.shot_dir) if (f.endswith(".jpg") or f.endswith(".png")) and f[0] != '.']
        self.images.sort()
        self.window = tk.Tk()
        self.curr_img = None
        self.label = None
        self.curr_img_obj = None

        self.window.bind("<Up>", self.move_file_on_event)
        self.window.bind("<Down>", self.move_file_on_event)
        self.window.bind("<Right>", self.move_file_on_event)
        self.window.bind("<Left>", self.move_file_on_event)
        self.window.bind("<Left>", self.move_file_on_event)
        self.window.bind("<BackSpace>", self.move_file_on_event)

        self.class_dict = {
            "Up" : os.path.join(self.class_dir, "distress"),
            "Down" : os.path.join(self.class_dir, "slight_distress"),
            "Left" : os.path.join(self.class_dir, "unknown"),
            "Right" : os.path.join(self.class_dir, "no_distress"),
            "BackSpace" : os.path.join(self.class_dir, "trash")
        }

    def start_classifier(self):
        try:
            self.curr_img = self.images.pop()
            if self.curr_img is None:
                raise IndexError
        except IndexError:
            pass
        
        curr_img_path = os.path.join(self.shot_dir, self.curr_img)
        img = Image.open(curr_img_path)
        self.curr_img_obj = ImageTk.PhotoImage(img)
        print("Curr img dimensions: ", img.size)
        self.label = tk.Label(self.window, image=self.curr_img_obj)
        self.label.pack()
        self.window.mainloop()

    def move_file_on_event(self, event):
        try:
            # path of shot you were looking at
            if event.keysym == "Up":
                curr_img_path = os.path.join(self.shot_dir, self.curr_img)

                to_path = os.path.join(self.class_dir, self.curr_img)

                print(f"Moving img from {curr_img_path} to {to_path}")
                shutil.move(curr_img_path, to_path)
            
            new_img = self.images.pop()
            new_shot_img_path = os.path.join(self.shot_dir, new_img)
            img = Image.open(new_shot_img_path)
            self.curr_img_obj = ImageTk.PhotoImage(img)
            print("Curr img dimensions: ", img.size)
            self.label.configure(image=self.curr_img_obj)

            # don't entirely understand why, but you need
            # to create a reference to the curr_img_obj
            # otherwise it gets garbage collected
            self.label.image = self.curr_img_obj
            self.curr_img = new_img
        except IndexError:
            print("No more screenshots! Stopping classifier loop")
            self.window.destroy()

if __name__ == "__main__":
    shot_dir = "/media/colet/Images/detroit3/no_distress"
    class_dir = "/home/colet/programming/projects/ai_model/marked_data"
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)

    mc = ManualClassifier(shot_dir, class_dir)
    mc.start_classifier()