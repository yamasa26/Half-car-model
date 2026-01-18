import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# データ読み込み
df = pd.read_csv('simulation_results.csv')

# パラメータ設定
L1, L2 = 1.2, 1.3
sim_dt = 0.001
fps = 60
step_size = int((1/fps) / sim_dt) 

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_aspect('equal')
ax.grid(True)

# 描画オブジェクト
body, = ax.plot([], [], 'k-', lw=6, label='Body')
susp_f, = ax.plot([], [], 'r-', lw=3)
susp_r, = ax.plot([], [], 'b-', lw=3)
wheel_f = plt.Circle((0, 0), 0.3, color='gray', fill=False, lw=2)
wheel_r = plt.Circle((0, 0), 0.3, color='gray', fill=False, lw=2)
ax.add_patch(wheel_f)
ax.add_patch(wheel_r)
ground, = ax.plot([-100, 1000], [0, 0], 'g-', lw=1)

def update(frame):
    idx = frame * step_size
    if idx >= len(df): return body,
    
    row = df.iloc[idx]
    x_curr = row['x_abs']
    ys, theta = row['ys'], row['theta']
    yu1, yu2 = row['yu1'], row['yu2']
    
    # 座標変換 (x軸方向に移動)
    y_f = ys + L1 * np.sin(theta) + 0.5
    y_r = ys - L2 * np.sin(theta) + 0.5
    
    body.set_data([x_curr - L2, x_curr + L1], [y_r, y_f])
    susp_f.set_data([x_curr + L1, x_curr + L1], [yu1, y_f])
    susp_r.set_data([x_curr - L2, x_curr - L2], [yu2, y_r])
    wheel_f.center = (x_curr + L1, yu1)
    wheel_r.center = (x_curr - L2, yu2)
    
    # 並走カメラ：車を画面中央に
    ax.set_xlim(x_curr - 5, x_curr + 5)
    ax.set_ylim(-1, 3)
    ax.set_title(f"Time: {row['time']:.2f}s | Accel: {row['v_abs']:.1f}m/s")
    
    return body, susp_f, susp_r, wheel_f, wheel_r

ani = animation.FuncAnimation(fig, update, frames=len(df)//step_size, 
                              interval=1000/fps, blit=False)
plt.legend()
plt.show()