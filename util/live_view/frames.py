from abc import ABC, abstractmethod
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


class LiveView(ABC):
    def __init__(self) -> None:
        self.q_hits: Queue[tuple[np.ndarray, float]] = Queue()
        self.closed_evt = Event()
        self.proc_plot = Process(
            target=self._proc_plot_target,
            name='LiveViewFramesProcess',
            daemon=True,
        )
        self.proc_plot.start()

    @abstractmethod
    def _proc_plot_target(self) -> None:
        raise NotImplementedError()

    @property
    def closed(self) -> bool:
        return self.closed_evt.is_set()

    def show_frame(self, frame: np.ndarray, frame_us: float):
        try:
            if self.q_hits.qsize() < 20:
                self.q_hits.put((frame, frame_us))
        except BrokenPipeError:
            pass


class LiveViewFrames(LiveView):
    def __init__(self, sensor_size: tuple[int, int]) -> None:
        self.sensor_size = sensor_size
        super().__init__()

    def _proc_plot_target(self) -> None:
        hits_total = np.ones(self.sensor_size, np.int64)
        plot_total_us = 0
        # only import matplotlib in child process
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
            fig.canvas.start_event_loop(0.05)
            # plt.pause(cast(int, 0.05))  # plt.pause accepts floats
            # read all available frames from queue
            hits_new = None
            plot_new_us = 0.0
            try:
                while True:
                    frame_hits, frame_us = self.q_hits.get_nowait()
                    if hits_new is None:
                        hits_new = frame_hits
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
            # plot ranges
            vmax_new = max(2, np.max(hits_new))
            vmax_total = max(2, np.max(hits_total))
            # update plots
            image_frame.set_data(np.flip(hits_new, axis=1))
            image_frame.set_norm(LogNorm(vmin=1, vmax=vmax_new))
            image_total.set_data(np.flip(hits_total, axis=1))
            image_total.set_norm(LogNorm(vmin=1, vmax=vmax_total))
            ax_frame.set_title(f'hits / {format_time(plot_new_us)}')
            ax_total.set_title(f'hits / {format_time(plot_total_us)}')


class LiveViewAdders(LiveView):
    def __init__(self, numcolumns: int, scroll_rows: int) -> None:
        self.numcolumns = numcolumns
        self.scroll_rows = scroll_rows
        super().__init__()

    def _proc_plot_target(self) -> None:
        total_us = 0 
        frame_buffer = np.zeros(self.numcolumns, np.int64)
        total_buffer = np.zeros(self.numcolumns, np.int64)
        scroll_buffer = np.zeros((self.scroll_rows, self.numcolumns), np.int64)
        # only import matplotlib in child process
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import Axes
        from matplotlib.figure import Figure
        from matplotlib.colors import LogNorm
        # create plots
        fig, (ax_frame, ax_total, ax_scroll) = cast(
            tuple[Figure, tuple[Axes,...]],
            plt.subplots(1, 3),
        )
        # ax_frame.semilogy()
        # ax_total.semilogy()
        frame_plot, = ax_frame.plot(total_buffer)
        total_plot, = ax_total.plot(total_buffer)
        scroll_img = ax_scroll.imshow(scroll_buffer)
        plt.colorbar(scroll_img, ax=ax_scroll)
        plt.ion()
        plt.show()
        running = True

        # catch close event
        def on_close(_):
            nonlocal running
            running = False
            self.closed_evt.set()

        fig.canvas.mpl_connect('close_event', on_close)

        # main loop
        while running:
            # plt.pause accepts floats
            plt.pause(cast(int, 0.05))
            # read all available frames from queues
            frame_any = False
            frame_us = 0.0
            frame_buffer.fill(0)
            try:
                while True:
                    frame_hits, frame_us = self.q_hits.get_nowait()
                    frame_buffer += frame_hits
                    frame_us += frame_us
                    frame_any = True
            except Empty:
                pass
            # no frames read?
            if not frame_any:
                continue
            assert frame_buffer.ndim == 1
            # flip axis
            frame_buffer[:] = frame_buffer[::-1]
            # update total state
            total_buffer += frame_buffer
            total_us += frame_us
            scroll_buffer[1:-1] = scroll_buffer[0:-2]
            scroll_buffer[0] = frame_buffer
            # plot ranges
            ax_frame.set_ylim(0, max(2, 1.1*np.max(frame_buffer)))
            ax_total.set_ylim(0, max(2, 1.1*np.max(total_buffer)))
            vmax_scroll = max(2, np.max(scroll_buffer))
            # update plots
            frame_plot.set_ydata(frame_buffer)
            total_plot.set_ydata(total_buffer)
            scroll_img.set_data(scroll_buffer)
            # update ranges
            scroll_img.set_norm(LogNorm(vmin=1, vmax=vmax_scroll))
            # image_total.set_data(np.flip(hits_total, axis=1))
            # image_total.set_norm(LogNorm(vmin=1, vmax=vmax_total))
            # ax_frame.set_title(f'hits / {format_time(plot_new_us)}')
            # ax_total.set_title(f'hits / {format_time(plot_total_us)}')
