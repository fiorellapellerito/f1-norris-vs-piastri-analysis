#Project's goal: Compare Norris' and Piastri's performance lap by lap

import fastf1
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.ensemble import RandomForestRegressor
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
import os

fastf1.Cache.enable_cache('cache')

#Load the session (Abu Dhabi, 2024)
session = fastf1.get_session(2024, "Abu Dhabi", "R")
session.load()

#Get Norris' and Piastri's quick laps
norris_quicklaps = session.laps.pick_drivers('NOR').pick_quicklaps()
piastri_quicklaps = session.laps.pick_drivers('PIA').pick_quicklaps()

#Convert lap times to seconds
norris_times = norris_quicklaps['LapTime'].dt.total_seconds()
piastri_times = piastri_quicklaps['LapTime'].dt.total_seconds()

#Plot lap time comparison
plt.figure(figsize=(12, 6))
plt.plot(norris_times.values, label='Norris', marker='o', color=(1.0, 0.53, 0.0))
plt.plot(piastri_times.values, label='Piastri', marker='o', color=(0.0, 0.75, 1.0))
plt.xlabel("Lap number")
plt.ylabel("Lap Time (seconds)")
plt.title('Norris vs. Piastri - Lap Time Comparison (Abu Dhabi 2024)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Print each driver's average lap time and consistency
print(f"Norris' average lap time is: {norris_times.mean():.3f}s")
print(f"Piastri's average lap time is: {piastri_times.mean():.3f}s")
print(f"Norris' lap consistency (std) is: {norris_times.std():.3f}s")
print(f"Piastri's lap consistency (std) is: {piastri_times.std():.3f}s")

#Select one specific lap from each driver
lap_norris = norris_quicklaps[norris_quicklaps['LapNumber'] == 28].iloc[0]
lap_piastri = piastri_quicklaps[piastri_quicklaps['LapNumber'] == 28].iloc[0]

#Get Telemetry Data for each driver on lap 28
tel_nor = lap_norris.get_telemetry().add_distance()
tel_pia = lap_piastri.get_telemetry().add_distance()

#Get throttle and brake average
throttle_avg_nor = tel_nor['Throttle'].mean()
brake_avg_nor = tel_nor['Brake'].mean()
speed_avg_nor = tel_nor['Speed'].mean()
throttle_avg_pia = tel_pia['Throttle'].mean()
brake_avg_pia = tel_pia['Brake'].mean()
speed_avg_pia = tel_pia['Speed'].mean()

#Plot throttle and brake (x-axis)
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(tel_nor['Distance'], tel_nor['Throttle'] * 100, label='Norris Throttle', color=(1.0, 0.53, 0.0))
ax1.plot(tel_pia['Distance'], tel_pia['Throttle'] * 100, label='Piastri Throttle', color=(0.0, 0.75, 1.0))

ax1.plot(tel_nor['Distance'], tel_nor['Brake'] * 100, label='Norris Brake', linestyle='--', color=(1.0, 0.53, 0.0))
ax1.plot(tel_pia['Distance'], tel_pia['Brake'] * 100, label='Piastri Brake', linestyle='--', color=(0.0, 0.75, 1.0))

ax1.set_xlabel('Distance around lap (m)')
ax1.set_ylabel('Pedal Input (%)')
ax1.legend(loc='upper left')
ax1.grid(True)

#Plot speed and gear (y-axis)
norris_avg_speed = tel_nor['Speed'].mean()
piastri_avg_speed = tel_pia['Speed'].mean()

ax2 = ax1.twinx()
ax2.plot(tel_nor['Distance'], tel_nor['Speed'], label='Norris Speed', linestyle=':', color=(1.0, 0.53, 0.0))
ax2.plot(tel_pia['Distance'], tel_pia['Speed'], label='Piastri Speed', linestyle=':', color=(0.0, 0.75, 1.0))
ax2.set_ylabel('Speed (km/h)')

plt.title("Throttle, Brake, Speed - Lap 28 (Abu Dhabi 2024)")
fig.tight_layout()
plt.show()

#Create a Machine Learning Model that predicts Lap Times Based on Driving Style
df_nor = tel_nor[['Throttle', 'Brake', 'Speed']].copy()
df_nor['Driver'] = 'Norris'
df_nor['LapTime'] = lap_norris['LapTime'].total_seconds()

df_pia = tel_pia[['Throttle', 'Brake', 'Speed']].copy()
df_pia['Driver'] = 'Piastri'
df_pia['LapTime'] = lap_piastri['LapTime'].total_seconds()

df = pd.concat([df_nor, df_pia])

X = df[["Throttle", "Brake", "Speed"]]
y = df['LapTime']
model = RandomForestRegressor()
model.fit(X, y)

#Predict using average inputs
throttle_avg_pia = tel_pia['Throttle'].mean()
brake_avg_pia = tel_pia['Brake'].mean()
speed_avg_pia = tel_pia['Speed'].mean()

#Print predictions
pred_nor = model.predict([[throttle_avg_nor, brake_avg_nor, speed_avg_nor]])
pred_pia = model.predict([[throttle_avg_pia, brake_avg_pia, speed_avg_pia]])

#Actual drivers' times
actual_nor = lap_norris['LapTime'].total_seconds()
actual_pia = lap_piastri['LapTime'].total_seconds()

#Differences between the ML predicted times and the actual times
delta_nor = pred_nor[0] - actual_nor
delta_pia = pred_pia[0] - actual_pia

print(f"Predicted Norris' Lap Time: {pred_nor[0]:.3f}s")
print(f"Predicted Piastri's Lap Time: {pred_pia[0]:.3f}s")

#Plot ML Model
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')
    
table_data = [
["Driver", "Avg Lap Time (s)", "Consistency (std)", "ML Prediction (s)", "Actual (s)", "Δ Prediction (s)"],
["Norris", f"{norris_times.mean():.3f}", f"{norris_times.std():.3f}", f"{pred_nor[0]:.3f}", f"{actual_nor:.3f}", f"{delta_nor:+.3f}"],
["Piastri", f"{piastri_times.mean():.3f}", f"{piastri_times.std():.3f}", f"{pred_pia[0]:.3f}", f"{actual_pia:.3f}", f"{delta_pia:+.3f}"]
]
 
table = ax.table(cellText=table_data, loc='center', cellLoc='center', colLoc='center', edges='horizontal')
table.scale(1, 2)
table.auto_set_font_size(False)
table.set_fontsize(12)
    
ax.text(0.2, 0.8, "Performance Summary Table - Lando Norris vs. Oscar Piastri", fontsize=16, weight='bold')

plt.title
fig.tight_layout()
plt.show()

#Save the project as a pdf
def add_logo(ax, path, zoom=0.1, x=0.95, y=1.05):
    img = mpimg.imread(path)
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (x, y), xycoords='axes fraction', frameon=False)
    ax.add_artist(ab)
page_num = 2

with PdfPages("norris_vs_piastri_report.pdf") as pdf:
    
    #Plot 1
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(norris_times.values, label='Norris', marker='o', color=(1.0, 0.53, 0.0))
    ax.plot(piastri_times.values, label='Piastri', marker='o', color=(0.0, 0.75, 1.0))
    ax.set_xlabel("Lap number")
    ax.set_ylabel("Lap Time (s)")
    ax.set_title('Lando Norris vs. Oscar Piastri - Lap Time Comparison (Abu Dhabi 2024)', fontsize=16, weight='bold')
    ax.legend()
    ax.grid(True)
    add_logo(ax, r'C:\Users\PC\Desktop\-\python\projects\F1\mclaren_logo.png')
    fig.text(0.95, 0.02, f"{page_num}", ha='right', fontsize=10, color='black')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    page_num += 1

    #Plot 2
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(tel_nor['Distance'], tel_nor['Throttle'] * 100, label='Norris Throttle', color=(1.0, 0.53, 0.0))
    ax1.plot(tel_pia['Distance'], tel_pia['Throttle'] * 100, label='Piastri Throttle', color=(0.0, 0.75, 1.0))
    
    ax1.plot(tel_nor['Distance'], tel_nor['Brake'] * 100,
         label='Norris Brake', linestyle='--', linewidth=1.5, color=(1.0, 0.53, 0.0), alpha=0.8)
    ax1.plot(tel_pia['Distance'], tel_pia['Brake'] * 100,
         label='Piastri Brake', linestyle='--', linewidth=1.5, color=(0.0, 0.75, 1.0), alpha=0.8)
    
    ax1.set_xlabel('Distance around lap (m)')
    ax1.set_ylabel('Pedal Input (%)')
    ax1.set_title('Throttle and Brake Usage (Lap 28)')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    add_logo(ax1, r'C:\Users\PC\Desktop\-\python\projects\F1\mclaren_logo.png')
    fig.text(0.95, 0.02, f"{page_num}", ha='right', fontsize=10, color='black')
    pdf.savefig(fig)
    plt.close(fig)
    page_num += 1
    
    #Plot 3
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)
    
    ax.text(0.1, 1.0, "ML-Predicted Lap Times (Lap 28)", fontsize=16, weight='bold', transform=ax.transAxes)
    ax.text(0.1, 0.9, f"Norris - Predicted: {pred_nor[0]:.3f}s | Actual: {actual_nor:.3f}s | Δ: {delta_nor:+.3f}s", fontsize=12, transform=ax.transAxes)
    ax.text(0.1, 0.83, f"Piastri - Predicted: {pred_pia[0]:.3f}s | Actual: {actual_pia:.3f}s | Δ: {delta_pia:+.3f}s", fontsize=12, transform=ax.transAxes)
    
    bar_labels = ['Norris', 'Piastri']
    predicted_times = [pred_nor[0], pred_pia[0]]
    actual_times = [actual_nor, actual_pia]
    x = [0, 1]
    bar_width = 0.5
    ax.bar(x[0] - bar_width/2, predicted_times[0], width=bar_width, color=(1.0, 0.53, 0.0), label='Predicted (Norris)')
    ax.bar(x[0] + bar_width/2, actual_times[0],    width=bar_width, color=(1.0, 0.53, 0.0), alpha=0.5, label='Actual (Norris)')
    
    ax.bar(x[1] - bar_width/2, predicted_times[1], width=bar_width, color=(0.0, 0.75, 1.0), label='Predicted (Piastri)')
    ax.bar(x[1] + bar_width/2, actual_times[1],    width=bar_width, color=(0.0, 0.75, 1.0), alpha=0.5, label='Actual (Piastri)')
    
    for i in range(len(x)):
        ax.text(x[i] - bar_width/2, predicted_times[i] + 0.2, f"{predicted_times[i]:.3f}s", ha='center', fontsize=8)
        ax.text(x[i] + bar_width/2, actual_times[i] + 0.2, f"{actual_times[i]:.3f}s", ha='center', fontsize=8)
        
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels)
    ax.set_ylim(min(actual_times + predicted_times) - 0.5, max(actual_times + predicted_times) + 1)
    ax.set_ylabel('Lap Time (s)')
    ax.legend(title="Predicted vs. Actual Times",loc='lower right')
    
    add_logo(ax, r'C:\Users\PC\Desktop\-\python\projects\F1\mclaren_logo.png')
    fig.text(0.95, 0.02, f"{page_num}", ha='right', fontsize=10, color='black')
    
    pdf.savefig(fig)
    plt.close(fig)
    page_num += 1
    
    #Plot 4
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    table_data = [
    ["Driver", "Avg Lap Time (s)", "Consistency (std)", "ML Prediction (s)", "Actual (s)", "Δ Prediction (s)"],
    ["Norris", f"{norris_times.mean():.3f}", f"{norris_times.std():.3f}", f"{pred_nor[0]:.3f}", f"{actual_nor:.3f}", f"{delta_nor:+.3f}"],
    ["Piastri", f"{piastri_times.mean():.3f}", f"{piastri_times.std():.3f}", f"{pred_pia[0]:.3f}", f"{actual_pia:.3f}", f"{delta_pia:+.3f}"]
    ]
 
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colLoc='center', edges='horizontal')
    table.scale(1, 2)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    
    #Plot 5
    ax.text(0.07, 0.8, "Performance Summary Table - Lando Norris vs. Oscar Piastri", fontsize=16, weight='bold')
    
    add_logo(ax, r'C:\Users\PC\Desktop\-\python\projects\F1\mclaren_logo.png')
    fig.text(0.95, 0.02, f"{page_num}", ha='right', fontsize=10, color='black')
    
    pdf.savefig(fig)
    plt.close(fig)
    page_num += 1

    #Summary (txt)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.text(-0.1, 0.8, "Lap-by-Lap Performance Comparison: Lando Norris vs. Oscar Piastri - Abu Dhabi 2024", fontsize=16, weight='bold')
    ax.text(0, 0.65, f"Norris Avg. lap: {norris_times.mean():.3f}s", fontsize=12)
    ax.text(0, 0.55, f"Piastri Avg. lap: {piastri_times.mean():.3f}s", fontsize=12)
    ax.text(0, 0.45, f"Norris Consistency (std): {norris_times.std():.3f}s", fontsize=12)
    ax.text(0, 0.35, f"Piastri Consistency (std): {piastri_times.std():.3f}s", fontsize=12)
    ax.text(0, 0.25, f"Lap analyzed in detail: 28", fontsize=12)
    add_logo(ax, r'C:\Users\PC\Desktop\-\python\projects\F1\mclaren_logo.png')
    fig.text(0.95, 0.02, f"{page_num}", ha='right', fontsize=10, color='black')
    pdf.savefig(fig)
    plt.close()
