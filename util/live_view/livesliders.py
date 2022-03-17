import tkinter
from typing import Optional

class SliderValue:
    def __init__(self, label: str, range: tuple[float, float], resolution: float, value: float):
        self.label = label
        self.range = range
        self.resolution = resolution
        self.value = value
        self.scale: Optional[tkinter.Scale] = None


class LiveSliders:
    def __init__(self, values: list[SliderValue]):
        self.master = tkinter.Tk()

        for value in values:
            frame = tkinter.Frame(self.master)
            frame.pack()
            label = tkinter.Label(frame, text=value.label, width=10)
            label.pack(side='left')
            value.scale = tkinter.Scale(frame, from_=0, to=63, orient=tkinter.HORIZONTAL, length=400)
            value.scale.pack()

if __name__ == '__main__':
    values = [
        SliderValue('dac.vn1', (0, 63), 1, 10),
        SliderValue('dac.vn2', (0, 63), 1, 0),
        SliderValue('vssa', (1.0, 2.1), 0.005, 1.25),
    ]
    livesliders = LiveSliders(values)
    tkinter.mainloop()
