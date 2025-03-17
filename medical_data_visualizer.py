# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import the data into a DataFrame
df = pd.read_csv('medical_examination.csv')  # Read the CSV file into pandas DataFrame

# 2. Add 'overweight' column based on BMI calculation
# Calculate BMI = weight (kg) / (height (m))^2
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)  # 1 if BMI > 25, else 0

# 3. Normalize 'cholesterol' and 'gluc' columns: 0 is good, 1 is bad
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)  # cholesterol >1 is bad (1), else good (0)
df['gluc'] = (df['gluc'] > 1).astype(int)  # gluc >1 is bad (1), else good (0)

# 4. Function to draw categorical plot
def draw_cat_plot():
    # 5. Reshape data with melt to long format for specific columns
    df_cat = pd.melt(df, 
                     id_vars=['cardio'], 
                     value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

    # 6. Group by cardio, variable, and value, then count occurrences
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7. Create categorical plot using seaborn's catplot
    cat_plot = sns.catplot(data=df_cat, kind='bar',
                           x='variable', y='total', hue='value',
                           col='cardio')

    # 8. Extract the figure object for saving
    fig = cat_plot.fig

    # 9. Save figure
    fig.savefig('catplot.png')
    return fig

# 10. Function to draw heatmap
def draw_heat_map():
    # 11. Clean the data:
    # Remove rows where:
    # - diastolic pressure > systolic pressure
    # - height outside 2.5th and 97.5th percentiles
    # - weight outside 2.5th and 97.5th percentiles
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # 12. Calculate correlation matrix
    corr = df_heat.corr()

    # 13. Generate mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Set up matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15. Draw heatmap with seaborn
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f',
                center=0, vmin=-0.1, vmax=0.3, square=True,
                linewidths=0.5, cbar_kws={'shrink': 0.5})

    # 16. Save figure
    fig.savefig('heatmap.png')
    return fig