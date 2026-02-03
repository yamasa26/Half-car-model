import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button

# パラメータ設定
car_params = {
    "GR86": {"L1": 1.28, "L2": 1.29},
    "LexusLS": {"L1": 1.55, "L2": 1.57},
    "Samber": {"L1": 0.95, "L2": 0.95}
}

sim_dt = 0.001  
fps = 30        
step_size = int((1/fps) / sim_dt)  
interval_ms = 1000 / fps

class VehicleDynamicsAnimator:
    def __init__(self):
        # レイアウトの設定 (画面とグラフ)
        self.fig = plt.figure(figsize=(12, 9))
        self.gs = self.fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
        self.ax = self.fig.add_subplot(self.gs[0])
        self.ax_telemetry = self.fig.add_subplot(self.gs[1])
        
        plt.subplots_adjust(bottom=0.15)
        
        # 画面の設定
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.body, = self.ax.plot([], [], 'k-', lw=6, label='Body')
        self.susp_f, = self.ax.plot([], [], 'r-', lw=3)
        self.susp_r, = self.ax.plot([], [], 'b-', lw=3)
        self.wheel_f = plt.Circle((0, 0), 0.25, color='gray', fill=False, lw=2)
        self.wheel_r = plt.Circle((0, 0), 0.25, color='gray', fill=False, lw=2)
        self.ax.add_patch(self.wheel_f)
        self.ax.add_patch(self.wheel_r)
        self.ground, = self.ax.plot([-100, 10000], [0, 0], 'g-', lw=1)
        
        # テキスト情報
        self.info_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, 
                                     fontsize=12, fontweight='bold', verticalalignment='top',
                                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        self.brake_text = self.ax.text(0.5, 0.85, '', transform=self.ax.transAxes,
                                      fontsize=20, fontweight='bold', color='red', 
                                      ha='center', va='center')

        # グラフの設定
        self.ax_telemetry.set_title("Pitch Angle Graph", fontsize=10)
        self.ax_telemetry.set_ylabel("Pitch [deg]")
        self.ax_telemetry.set_xlabel("Time [s]")
        self.ax_telemetry.grid(True, linestyle=':', alpha=0.6)
        self.pitch_line, = self.ax_telemetry.plot([], [], 'b-', lw=1.5)
        self.pitch_pointer, = self.ax_telemetry.plot([], [], 'ro')
        
        self.df = None
        self.current_car = "GR86"
        self.load_data("GR86")

        # UIボタンの設定
        self.buttons = []
        for i, car_name in enumerate(car_params.keys()):
            ax_btn = plt.axes([0.2 + i*0.2, 0.03, 0.15, 0.05])
            btn = Button(ax_btn, car_name)
            btn.on_clicked(lambda event, name=car_name: self.change_car(name))
            self.buttons.append(btn)

        self.ani = animation.FuncAnimation( 
            self.fig, 
            self.update, 
            frames=len(self.df) // step_size, 
            interval=interval_ms, 
            blit=False 
        )

    def load_data(self, car_name):
        try:
            filename = f"csv/{car_name}_sim.csv"
            self.df = pd.read_csv(filename)
            self.current_car = car_name
            
            # 加速度
            dt = self.df['time'].diff().mean()
            self.df['accel'] = self.df['v_abs'].diff() / dt
            
            self.ax_telemetry.set_xlim(self.df['time'].min(), self.df['time'].max())
            pitch_deg = np.degrees(self.df['theta'])
            self.ax_telemetry.set_ylim(pitch_deg.min() - 1, pitch_deg.max() + 1)
            self.pitch_line.set_data(self.df['time'], pitch_deg)
            
            print(f"Loaded: {filename}")
        except FileNotFoundError:
            print(f"Error: {filename} is not found.")

    def change_car(self, car_name):
        self.load_data(car_name)
        self.ani.frame_seq = self.ani.new_frame_seq() 

    def update(self, frame):
        if self.df is None: return self.body,
        
        idx = frame * step_size
        if idx >= len(self.df):
            idx = len(self.df) - 1 
        
        row = self.df.iloc[idx]
        x_curr = row['x_abs']
        ys, theta = row['ys'], row['theta']
        yu1, yu2 = row['yu1'], row['yu2']
        accel = row['accel']
        
        L1 = car_params[self.current_car]["L1"]
        L2 = car_params[self.current_car]["L2"]
        
        y_f = ys + L1 * np.sin(theta) + 0.75 
        y_r = ys - L2 * np.sin(theta) + 0.75
        
        # 車体パーツの更新
        self.body.set_data([x_curr - L2, x_curr + L1], [y_r, y_f])
        self.susp_f.set_data([x_curr + L1, x_curr + L1], [yu1, y_f])
        self.susp_r.set_data([x_curr - L2, x_curr - L2], [yu2, y_r])
        self.wheel_f.center = (x_curr + L1, yu1 + 0.25)
        self.wheel_r.center = (x_curr - L2, yu2 + 0.25)
        
        # カメラ追従
        self.ax.set_xlim(x_curr - 5, x_curr + 5)
        self.ax.set_ylim(-1, 3)
        
        # ポインターの更新
        pitch_val = np.degrees(theta)
        self.pitch_pointer.set_data([row['time']], [pitch_val])
        
        # ブレーキ判定とテキスト更新 (加速度が -1.0 m/s^2 以下ならブレーキと定義)
        is_braking = accel < -1.0 
        if is_braking:
            self.brake_text.set_text("BRAKING")
            self.info_text.set_color('red')
        else:
            self.brake_text.set_text("")
            self.info_text.set_color('black')
            
        self.info_text.set_text(
            f"Model: {self.current_car}\n"
            f"Time: {row['time']:.2f}s\n"
            f"Speed: {row['v_abs']*3.6:.1f} km/h\n"
            f"Pitch: {pitch_val:.2f} deg"
        )
        
        return (self.body, self.susp_f, self.susp_r, self.wheel_f, 
                self.wheel_r, self.info_text, self.brake_text, self.pitch_pointer)

if __name__ == "__main__":
    vis = VehicleDynamicsAnimator()
    plt.show()