#
# import tkinter as tk
# from tkinter import filedialog
# import pandas as pd
# import pickle
#
# from sklearn import metrics
#
# from config.constants import TARGET_COLUMN
#
# class CsvModelTester:
#     def __init__(self, master):
#         self.master = master
#         master.title("CSV Model Tester")
#
#         # Create UI elements
#         self.file_label = tk.Label(master, text="No file selected")
#         self.file_label.pack()
#
#         self.select_file_button = tk.Button(master, text="Select File", command=self.select_file)
#         self.select_file_button.pack()
#
#         self.model_label = tk.Label(master, text="Select a model:")
#         self.model_label.pack()
#
#         self.model_var = tk.StringVar()
#         self.model_var.set("RandomForest")
#
#         self.model_1_radio = tk.Radiobutton(master, text="RandomForest", variable=self.model_var, value="RandomForest")
#         self.model_1_radio.pack()
#
#         self.model_2_radio = tk.Radiobutton(master, text="LinearModel", variable=self.model_var, value="LinearModel")
#         self.model_2_radio.pack()
#
#         self.mse_label = tk.Label(master, text="")
#         self.mse_label.pack()
#
#         self.accuracy_label = tk.Label(master, text="")
#         self.accuracy_label.pack()
#
#         self.test_button = tk.Button(master, text="Test Model", command=self.test_model)
#         self.test_button.pack()
#
#     def select_file(self):
#         file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
#         self.file_label.config(text=file_path)
#         self.data = pd.read_csv(file_path)
#
#     def evaluate_model(self, model, X_test, y_test):
#         y_pred = model.predict(X_test)
#         mse = metrics.mean_squared_error(y_test, y_pred)
#         accuracy = model.score(X_test, y_test)
#         return mse, accuracy
#
#     def test_model(self):
#         # Load the selected model
#         if self.model_var.get() == "RandomForest":
#             with open("C:/Users/GH/Documents/HotelRenting/models/RandomForest.pkl", "rb") as f:
#                 model = pickle.load(f)
#         else:
#             with open("C:/Users/GH/Documents/HotelRenting/models/LinearM.pkl", "rb") as f:
#                 model = pickle.load(f)
#
#         X_test = self.data.loc[:, self.data.columns != TARGET_COLUMN]
#         y_test = self.data[TARGET_COLUMN]
#         mse, accuracy = self.evaluate_model(model, X_test, y_test)
#         self.mse_label.config(text=f"MSE: {mse:.4f}")
#         self.accuracy_label.config(text=f"Accuracy: {accuracy:.4f}")
#
#
#
# root = tk.Tk()
# app = CsvModelTester(root)
# root.mainloop()

import tkinter as tk
from tkinter import filedialog
import pandas as pd
import pickle

from sklearn import metrics

from config.constants import TARGET_COLUMN

class CsvModelTester:
    def __init__(self, master):
        self.master = master
        master.title("CSV Model Tester")

        # Create UI elements with colorful styles
        self.file_label = tk.Label(master, text="No file selected", fg="white", bg="#3c3f41", font=("Arial", 12))
        self.file_label.pack(pady=10)

        self.select_file_button = tk.Button(master, text="Select File", command=self.select_file, bg="#0072C6", fg="white", font=("Arial", 12))
        self.select_file_button.pack(padx=10, pady=10)

        self.model_label = tk.Label(master, text="Select a model:", fg="white", bg="#3c3f41", font=("Arial", 12))
        self.model_label.pack(pady=10)

        self.model_var = tk.StringVar()
        self.model_var.set("RandomForest")

        self.model_1_radio = tk.Radiobutton(master, text="RandomForest", variable=self.model_var, value="RandomForest", fg="white", bg="#3c3f41", font=("Arial", 12), selectcolor="#0072C6")
        self.model_1_radio.pack()

        self.model_2_radio = tk.Radiobutton(master, text="LinearModel", variable=self.model_var, value="LinearModel", fg="white", bg="#3c3f41", font=("Arial", 12), selectcolor="#0072C6")
        self.model_2_radio.pack()

        self.mse_label = tk.Label(master, text="", fg="white", bg="#3c3f41", font=("Arial", 12))
        self.mse_label.pack(pady=10)

        self.accuracy_label = tk.Label(master, text="", fg="white", bg="#3c3f41", font=("Arial", 12))
        self.accuracy_label.pack(pady=10)

        self.test_button = tk.Button(master, text="Test Model", command=self.test_model, bg="#0072C6", fg="white", font=("Arial", 12))
        self.test_button.pack(pady=10)

        # Set the background color of the window
        master.config(bg="#3c3f41")

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        self.file_label.config(text=file_path)
        self.data = pd.read_csv(file_path)

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        mse = metrics.mean_squared_error(y_test, y_pred)
        accuracy = model.score(X_test, y_test)
        return mse, accuracy

    def test_model(self):
        # Load the selected model
        if self.model_var.get() == "RandomForest":
            with open("C:/Users/GH/Documents/HotelRenting/models/RandomForest.pkl", "rb") as f:
                model = pickle.load(f)
        else:
            with open("C:/Users/GH/Documents/HotelRenting/models/LinearM.pkl", "rb") as f:
                model = pickle.load(f)

        X_test = self.data.loc[:, self.data.columns != TARGET_COLUMN]
        y_test = self.data[TARGET_COLUMN]
        mse, accuracy = self.evaluate_model(model, X_test, y_test)
        self.mse_label.config(text=f"MSE: {mse:.4f}")
        self.accuracy_label.config(text=f"Accuracy: {accuracy:.4f}")



root = tk.Tk()
app = CsvModelTester(root)
root.mainloop()