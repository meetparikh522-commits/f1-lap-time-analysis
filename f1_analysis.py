import fastf1
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ============ SETUP ============
fastf1.Cache.enable_cache('cache')

session = fastf1.get_session(2026, 'Baku', 'R')
session.load()
laps = session.laps   # ALL drivers

# ============ DATA PREP (tracking how many laps survive each step) ============
raw = laps[['Driver', 'LapNumber', 'TyreLife', 'Compound', 'LapTime',
            'PitInTime', 'PitOutTime', 'TrackStatus']].copy()
raw['LapTimeSeconds'] = raw['LapTime'].dt.total_seconds()

funnel = {'All laps': len(raw)}

d = raw[raw['PitInTime'].isna() & raw['PitOutTime'].isna()]
funnel['No pit laps'] = len(d)

d = d.dropna(subset=['LapTimeSeconds', 'TyreLife', 'Compound'])
funnel['Complete data'] = len(d)

d = d[d['LapNumber'] > 1]
funnel['Not lap 1'] = len(d)

median_time = d['LapTimeSeconds'].median()
d = d[d['LapTimeSeconds'] < median_time * 1.15]
funnel['Not extremely slow'] = len(d)

d = d[d['TrackStatus'] == '1']
funnel['Green flag only'] = len(d)

clean = d.copy()

# Mark which raw laps survived cleaning (the index is preserved by the filters above)
raw['kept'] = raw.index.isin(clean.index)

# ============ MODEL ============
model_df = pd.get_dummies(clean, columns=['Compound', 'Driver'])

for col in ['Compound_SOFT', 'Compound_MEDIUM', 'Compound_HARD']:
    if col not in model_df.columns:
        model_df[col] = False

print("Correlation TyreLife vs LapNumber:",
      round(model_df['TyreLife'].corr(model_df['LapNumber']), 2))

feature_cols = ['TyreLife', 'LapNumber'] + [c for c in model_df.columns
                                            if c.startswith('Compound_') or c.startswith('Driver_')]
X = model_df[feature_cols].astype(float)
y = model_df['LapTimeSeconds']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
rmse = mean_squared_error(y_test, predictions) ** 0.5
baseline_rmse = ((y_test - y_train.mean()) ** 2).mean() ** 0.5

coefs = pd.Series(model.coef_, index=feature_cols)

print(f"Laps used for modeling: {len(clean)}")
print(f"Model RMSE: {rmse:.2f}s")
print(f"Baseline RMSE (always predict the mean): {baseline_rmse:.2f}s")
print(coefs.sort_values())

# Driver pace relative to the average driver (absolute values are arbitrary)
driver_coefs = coefs[[c for c in feature_cols if c.startswith('Driver_')]]
driver_coefs.index = driver_coefs.index.str.replace('Driver_', '')
driver_rel = (driver_coefs - driver_coefs.mean()).sort_values()

soft_vs_medium = coefs['Compound_SOFT'] - coefs['Compound_MEDIUM']

#*DISPLAY GRAPHS*
def plot_driver_stints(ax, driver_code, laps_data):
    driver_laps = laps_data.pick_drivers(driver_code).copy()
    driver_laps['LapTimeSeconds'] = driver_laps['LapTime'].dt.total_seconds()
    for stint in driver_laps['Stint'].unique():
        s = driver_laps[driver_laps['Stint'] == stint].sort_values('TyreLife')
        ax.plot(s['TyreLife'], s['LapTimeSeconds'], marker='o', markersize=3,
                label=f'{driver_code} S{int(stint)} ({s["Compound"].iloc[0]})')

fig = plt.figure(figsize=(15, 19))
gs = GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.25,
              top=0.93, bottom=0.04, left=0.07, right=0.97)
fig.suptitle('Baku 2026 Grand Prix - Lap Time Analysis', fontsize=18, y=0.975)

# eliminated laps vs laps kept
ax1 = fig.add_subplot(gs[0, 0])
rawp = raw.dropna(subset=['LapTimeSeconds'])
removed = rawp[~rawp['kept']]
kept = rawp[rawp['kept']]
ax1.scatter(removed['LapNumber'], removed['LapTimeSeconds'], s=12, c='red',
            alpha=0.6, label=f'Removed ({len(removed)})')
ax1.scatter(kept['LapNumber'], kept['LapTimeSeconds'], s=12, c='blue',
            alpha=0.6, label=f'Kept ({len(kept)})')
ax1.set_ylim(top=median_time * 1.6)
ax1.set_xlabel('Lap number')
ax1.set_ylabel('Lap time (s)')
ax1.set_title('eliminated laps vs laps kept')
ax1.legend()

# Cleaning Criterias
ax2 = fig.add_subplot(gs[0, 1])
labels = list(funnel.keys())[::-1]
values = list(funnel.values())[::-1]
ax2.barh(labels, values, color='tab:blue')
for i, v in enumerate(values):
    ax2.text(v + 5, i, str(v), va='center')
ax2.set_xlim(0, max(values) * 1.12)
ax2.set_title('Cleaning Criterias')
ax2.set_xlabel('Number of laps')

# Stint comparison
ax3 = fig.add_subplot(gs[1, 0])
for drv in ['RUS', 'VER', 'HAD']:
    plot_driver_stints(ax3, drv, laps)
ax3.set_ylim(median_time * 0.96, median_time * 1.12)
ax3.set_xlabel('Tyre life (laps)')
ax3.set_ylabel('Lap time (s)')
ax3.set_title('Lap time vs tyre age by stint\n(zoomed to racing pace, Safety Car laps off-screen)')
ax3.legend(fontsize=7, ncol=2)

# Effect sizes
ax4 = fig.add_subplot(gs[1, 1])
effects = {'TyreLife\n(s per lap of age)': coefs['TyreLife'],
           'LapNumber\n(s per lap, fuel proxy)': coefs['LapNumber'],
           'SOFT vs MEDIUM\n(s per lap)': soft_vs_medium}
bar_colors = ['tab:green' if v < 0 else 'tab:red' for v in effects.values()]
ax4.bar(list(effects.keys()), list(effects.values()), color=bar_colors)
for i, v in enumerate(effects.values()):
    ax4.text(i, v, f'{v:+.2f}', ha='center', va='bottom' if v >= 0 else 'top')
ax4.axhline(0, color='black', linewidth=0.8)
ax4.set_title('Model effects (negative = faster lap)')
ax4.set_ylabel('Seconds')

# Predicted vs actual
ax5 = fig.add_subplot(gs[2, 0])
ax5.scatter(y_test, predictions, s=14, alpha=0.6)
lims = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
ax5.plot(lims, lims, 'r--', label='Perfect prediction')
ax5.set_xlabel('Actual lap time (s)')
ax5.set_ylabel('Predicted lap time (s)')
ax5.set_title('Predicted vs actual (test laps)')
ax5.legend()

# Model vs baseline
ax6 = fig.add_subplot(gs[2, 1])
ax6.bar(['Baseline\n(always predict mean)', 'Linear Regression'],
        [baseline_rmse, rmse], color=['tab:gray', 'tab:green'])
for i, v in enumerate([baseline_rmse, rmse]):
    ax6.text(i, v, f'{v:.2f}s', ha='center', va='bottom')
ax6.set_ylabel('RMSE (s)')
ax6.set_title('Mean/Avg pace vs Model predicted pace')

# 7. Driver pace to avg pace
ax7 = fig.add_subplot(gs[3, :])
ax7.bar(driver_rel.index, driver_rel.values,
        color=['tab:green' if v < 0 else 'tab:red' for v in driver_rel.values])
ax7.axhline(0, color='black', linewidth=0.8)
ax7.set_ylabel('Seconds vs average driver')
ax7.set_title('7. Driver + car pace to average')

plt.savefig('f1_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()