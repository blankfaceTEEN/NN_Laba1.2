from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image


def openfile():
    filepath = filedialog.askopenfilename(initialdir="test",
                                          title="Тест",
                                          filetypes=(("png файлы", "*.png"),
                                                     ("png файлы", "*.png")))
    print(filepath)


window = Tk()
window.title("Хисамов Искандер Лабораторная работа №1.2")
window.geometry("550x350+700+400")
button = Button(text="Загрузить картинку", command=openfile)
button.pack(fill=BOTH, expand=0)
frame = Frame(window, relief=RAISED, borderwidth=1)
frame.pack(fill=BOTH, expand=True)
img = ImageTk.PhotoImage(Image.open("test/test.png"))
panel = Label(frame, image=img)
panel.pack(side="bottom", fill="both", expand="yes")
label = Label(text="На картинке: ")
label.pack(side=LEFT, padx=5, pady=5)
label = Label(text="нолик")
label.pack(side=LEFT)
window.mainloop()
