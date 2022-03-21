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
            scale = slider_value.scale = tkinter.Scale(
                frame,
                from_=vfrom, to=vto,
                orient=tkinter.HORIZONTAL,
                length=400,
                command=lambda val, sli=slider_value: self.__scale_command(
                    val, sli),
                resolution=cast(float, slider_value.resolution),  # None is ok
            )
            scale.bind("<Button-4>", lambda _, sli=slider_value: self.__on_mousewheel(+1, sli))
            scale.bind("<Button-5>", lambda _, sli=slider_value: self.__on_mousewheel(-1, sli))
            scale.set(slider_value.value)
            scale.pack()
        
    def __on_mousewheel(self, direction: int, slider_value: SliderValue):
        scale = slider_value.scale
        assert scale is not None
        # update value
        if slider_value.resolution is None:
            slider_value.value += direction
        else:
            slider_value.value += slider_value.resolution * direction
        # check range
        if slider_value.value < slider_value.extent[0]:
            slider_value.value = slider_value.extent[0]
        if slider_value.value > slider_value.extent[1]:
            slider_value.value = slider_value.extent[1]
        scale.set(slider_value.value)
        self.callback(slider_value)

    def __scale_command(self, val_new: str, slider_value: SliderValue):
        if slider_value.resolution is None:
            value = int(val_new)
        else:
            value = float(val_new)
        if slider_value.value == value:
            return
        slider_value.value = value
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
