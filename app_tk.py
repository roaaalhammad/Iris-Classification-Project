import tkinter as tk
from tkinter import messagebox
import joblib


class IrisGUI:
    def __init__(self, model):
        self.model = model
        self.window = tk.Tk()
        self.window.title("Iris Flower Classification")
        self.window.geometry("450x350")
        self.window.configure(bg="#f0f8ff")

        self.entries = {}
        self.create_widgets()

    def create_widgets(self):
        title_label = tk.Label(
            self.window,
            text="Iris Flower Predictor",
            font=("Arial", 16, "bold"),
            bg="#f0f8ff"
        )
        title_label.pack(pady=15)

        form_frame = tk.Frame(self.window, bg="#d9d9d9", padx=20, pady=15)
        form_frame.pack(pady=15)

        labels = [
            ("Sepal Length (cm):", "sl"),
            ("Sepal Width (cm):", "sw"),
            ("Petal Length (cm):", "pl"),
            ("Petal Width (cm):", "pw")
        ]

        for i, (label_text, key) in enumerate(labels):
            lbl = tk.Label(form_frame, text=label_text, bg="#e6f3ff", width=18)
            lbl.grid(row=i, column=0, padx=5, pady=8)

            entry = tk.Entry(form_frame, width=12)
            entry.grid(row=i, column=1, padx=5, pady=8)

            self.entries[key] = entry

        predict_button = tk.Button(
            self.window,
            text="Predict Species",
            width=15,
            height=2,
            command=self.on_predict
        )
        predict_button.pack(pady=15)

        self.result_label = tk.Label(
            self.window, text="", font=("Arial", 12), bg="#f0f8ff"
        )
        self.result_label.pack()

    def on_predict(self):
        try:
            sl = float(self.entries["sl"].get())
            sw = float(self.entries["sw"].get())
            pl = float(self.entries["pl"].get())
            pw = float(self.entries["pw"].get())

            features = [[sl, sw, pl, pw]]

            pred = self.model.predict(features)[0]

            species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
            result = species_map[pred]

            self.result_label.config(text=f"Predicted Species: {result}")

        except:
            messagebox.showerror("Error", "Please enter valid numbers.")

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    model = joblib.load("src/iris_model.pkl")
    app = IrisGUI(model)
    app.run()
