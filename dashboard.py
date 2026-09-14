import tkinter as tk
import queue

BG = "#faf9f6"
INK = "#1a1a1a"
INK_MUTED = "#6b6862"
INK_SUBTLE = "#a19d94"
ACCENT = "#0F6E56"
DIVIDER = "#1a1a1a"
DOTTED = "#d3d1c7"
GRAPH_BG = "#f2f0ea"
GRAPH_LINE = "#0F6E56"
GRAPH_AXIS = "#d3d1c7"

MONO = "Consolas"


class Dashboard:
    def __init__(self, update_queue, speed_control=None):
        self.queue = update_queue
        self.speed_control = speed_control

        self.scores = []
        self.mean_scores = []
        self.record = 0
        self.n_games = 0
        self.epsilon = 0
        self.total_score = 0

        self.root = tk.Tk()
        self.root.title("snake_dqn // training")
        self.root.configure(bg=BG)
        self.root.geometry("420x620")

        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=20, pady=(18, 10))

        tk.Label(header, text="SNAKE_DQN // TRAINING", font=(MONO, 13, "bold"),
                 bg=BG, fg=INK, anchor="w").pack(side="left")
        self.status_var = tk.StringVar(value=">> running")
        tk.Label(header, textvariable=self.status_var, font=(MONO, 12),
                 bg=BG, fg=ACCENT, anchor="e").pack(side="right")

        divider = tk.Frame(self.root, bg=DIVIDER, height=1)
        divider.pack(fill="x", padx=20)

        stats_frame = tk.Frame(self.root, bg=BG)
        stats_frame.pack(fill="x", padx=20, pady=(12, 10))

        self.row_vars = {}
        rows = [("games", "0000"), ("record", "0000"), ("last", "0000"),
                ("mean", "0.00"), ("epsilon", "0.00"), ("memory", "0000")]
        for label, default in rows:
            self._make_row(stats_frame, label, default)

        # --- speed slider ---
        speed_frame = tk.Frame(self.root, bg=BG)
        speed_frame.pack(fill="x", padx=20, pady=(4, 10))

        tk.Label(speed_frame, text="render speed", font=(MONO, 11),
                 bg=BG, fg=INK_MUTED, anchor="w").pack(side="left")

        initial_speed = self.speed_control.value if self.speed_control else 60
        self.speed_value_var = tk.StringVar(value=f"{initial_speed} fps")
        tk.Label(speed_frame, textvariable=self.speed_value_var, font=(MONO, 11, "bold"),
                 bg=BG, fg=INK, anchor="e").pack(side="right")

        self.speed_slider = tk.Scale(
            self.root, from_=5, to=200, orient="horizontal",
            bg=BG, fg=INK, troughcolor=GRAPH_BG, highlightthickness=0,
            bd=0, showvalue=False, command=self._on_speed_change
        )
        self.speed_slider.set(initial_speed)
        self.speed_slider.pack(fill="x", padx=20, pady=(0, 6))
        # --- end speed slider ---

        graph_label = tk.Label(self.root, text="score_history[-200:]", font=(MONO, 11),
                                bg=BG, fg=INK_MUTED, anchor="w")
        graph_label.pack(fill="x", padx=20, pady=(6, 6))

        self.canvas = tk.Canvas(self.root, bg=GRAPH_BG, height=180, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=20, pady=(0, 12))

        self.range_var = tk.StringVar(value="min 0   max 0   avg 0.0")
        tk.Label(self.root, textvariable=self.range_var, font=(MONO, 10),
                 bg=BG, fg=INK_SUBTLE, anchor="w").pack(fill="x", padx=20, pady=(0, 18))

        self._poll_queue()

    def _make_row(self, parent, label, default):
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", pady=1)

        lbl = tk.Label(row, text=label, font=(MONO, 12), bg=BG, fg=INK_MUTED, anchor="w", width=10)
        lbl.pack(side="left")

        value_var = tk.StringVar(value=default)
        val_color = ACCENT if label == "record" else INK
        val = tk.Label(row, textvariable=value_var, font=(MONO, 12, "bold"), bg=BG, fg=val_color, anchor="e")
        val.pack(side="right")

        underline = tk.Frame(row, bg=DOTTED, height=1)
        underline.pack(fill="x", side="bottom", pady=(2, 0))

        self.row_vars[label] = value_var

    def _on_speed_change(self, value):
        fps = int(float(value))
        self.speed_value_var.set(f"{fps} fps")
        if self.speed_control is not None:
            self.speed_control.value = fps

    def _poll_queue(self):
        try:
            while True:
                score, n_games, epsilon, record, memory_size = self.queue.get_nowait()
                self._apply_update(score, n_games, epsilon, record, memory_size)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _apply_update(self, score, n_games, epsilon, record, memory_size):
        self.scores.append(score)
        self.total_score += score
        self.mean_scores.append(self.total_score / n_games)
        self.n_games = n_games
        self.epsilon = max(epsilon, 0)
        self.record = record

        if len(self.scores) > 200:
            self.scores = self.scores[-200:]
            self.mean_scores = self.mean_scores[-200:]

        self.row_vars["games"].set(str(self.n_games).zfill(4))
        self.row_vars["record"].set(str(self.record).zfill(4))
        self.row_vars["last"].set(str(score).zfill(4))
        self.row_vars["mean"].set(f"{self.mean_scores[-1]:.2f}")
        self.row_vars["epsilon"].set(f"{self.epsilon / 80:.2f}" if self.epsilon else "0.00")
        self.row_vars["memory"].set(str(min(memory_size, 9999)).zfill(4))

        min_s = min(self.scores)
        max_s = max(self.scores)
        avg_s = sum(self.scores) / len(self.scores)
        self.range_var.set(f"min {min_s}   max {max_s}   avg {avg_s:.1f}")

        self._draw_graph()

    def _draw_graph(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10 or len(self.scores) < 2:
            return

        max_val = max(max(self.scores), 1)
        inset = 12

        axis_y = h - inset
        self.canvas.create_line(inset, axis_y, w - inset, axis_y, fill=GRAPH_AXIS)

        def to_points(data):
            n = len(data)
            pts = []
            for i, v in enumerate(data):
                px = inset + (i / (n - 1)) * (w - 2 * inset)
                py = h - inset - (v / max_val) * (h - 2 * inset)
                pts.append((px, py))
            return pts

        pts = to_points(self.scores)
        self.canvas.create_line(*[c for pt in pts for c in pt], fill=GRAPH_LINE, width=2)
        last_x, last_y = pts[-1]
        self.canvas.create_oval(last_x - 3, last_y - 3, last_x + 3, last_y + 3, fill=GRAPH_LINE, outline="")

    def run_forever(self):
        self.root.mainloop()