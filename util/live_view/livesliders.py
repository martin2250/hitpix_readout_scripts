from dataclasses import dataclass
import tkinter
from typing import Callable, Optional, cast


@dataclass
class SliderValue:
    label: str
    extent: tuple[float, float]
    value: int | float
    resolution: Optional[float] = None  # None == integer
    scale: Optional[tkinter.Scale] = None


CallbackType = Callable[[SliderValue], None]


class LiveSliders:
    def __init__(self, slider_values: list[SliderValue], callback: CallbackType):
        self.master = tkinter.Tk()
        self.callback = callback

        for slider_value in slider_values:
            frame = tkinter.Frame(self.master)
            frame.pack()
            label = tkinter.Label(frame, text=slider_value.label, width=10)
            label.pack(side='left')
            vfrom, vto = slider_value.extent
            slider_value.scale = tkinter.Scale(
                frame,
                from_=vfrom, to=vto,
                orient=tkinter.HORIZONTAL,
                length=400,
                command=lambda val, sli=slider_value: self.__scale_command(
                    val, sli),
                resolution=cast(float, slider_value.resolution),  # None is ok
            )
            slider_value.scale.pack()

    def __scale_command(self, val_new: str, slider_value: SliderValue):
        if slider_value.resolution is None:
            slider_value.value = int(val_new)
        else:
            slider_value.value = float(val_new)
        self.callback(slider_value)


if __name__ == '__main__':
    values = [
        SliderValue('dac.vn1', (0, 63), 10),
        SliderValue('dac.vn2', (0, 63), 0),
        SliderValue('vssa', (1.0, 2.1), 1.25, 0.005),
    ]

    def cb(x):
        print(x)
    livesliders = LiveSliders(values, cb)
    tkinter.mainloop()
