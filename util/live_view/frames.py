from typing import Any, cast
import numpy as np
from multiprocessing import Process, Queue, Event
from queue import Empty

def format_time(us: float) -> str:
    if us < 1000.0:
        return f'{us:0.1f} us'
    us /= 1000
    if us < 1000.0:
        return f'{us:0.1f} ms'
    us /= 1000
    return f'{us:0.1f} s'

class LiveViewFrames:
    def __init__(self, sensor_size: tuple[int, int]) -> None:
        self.q_hits: Queue[tuple[np.ndarray, float]] = Queue()
        self.sensor_size = sensor_size
        self.closed_evt = Event()
        self.proc_plot = Process(
            target=self._proc_plot_target,
            name='LiveViewFramesProcess',
            daemon=True,
        )
        self.proc_plot.start()
    
    @property
    def closed(self) -> bool:
        return self.closed_evt.is_set()

    def _proc_plot_target(self) -> None:
        hits_total = np.ones(self.sensor_size, np.int64)
        plot_total_us = 0
        
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        fig, (ax_frame, ax_total) = cast(Any, plt.subplots(1, 2))
        image_frame = ax_frame.imshow(hits_total)
        image_total = ax_total.imshow(hits_total)
        plt.colorbar(image_frame, ax=ax_frame)
        plt.colorbar(image_total, ax=ax_total)
        plt.ion()
        plt.show()
        running = True

        def on_close(_):
            nonlocal running
            running = False
            self.closed_evt.set()

        fig.canvas.mpl_connect('close_event', on_close)
    
        while running:
            plt.pause(cast(int, 0.05)) # plt.pause accepts floats
            # read all available frames from queue
            hits_new = None
            plot_new_us = 0.0
            try:
                while True:
                    frame_hits, frame_us = self.q_hits.get_nowait()
                    if hits_new is None:
                        hits_new = frame_hits + 1
                    else:
                        hits_new += frame_hits
                    plot_new_us += frame_us
            except Empty:
                pass
            # no frames read?
            if hits_new is None:
                continue
            # update total state 
            hits_total += hits_new
            plot_total_us += plot_new_us
            # update plots
            image_frame.set_data(np.flip(hits_new, axis=1))
            image_frame.set_norm(LogNorm(vmin=1, vmax=np.max(hits_new)))
            image_total.set_data(np.flip(hits_total, axis=1))
            image_total.set_norm(LogNorm(vmin=1, vmax=np.max(hits_total)))
            ax_frame.set_title(f'hits / {format_time(plot_new_us)}')
            ax_total.set_title(f'hits / {format_time(plot_total_us)}')

    def show_frame(self, frame: np.ndarray, frame_us: float):
        self.q_hits.put((frame, frame_us))
