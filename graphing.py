# graphing.py
# Author: Rigoberto Rodriguez-Anton, Romil Shah
# Course: Probabilistic Systems Analysis
# Final Project
# Purpose:
# This file contains the function to produce the graph of our observed data on
# the MBTA Green Line delays.
# We will be reading in the data from a csv file. 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def graph_data():
    data = pd.read_csv("final_delays_with_dates.csv")
    # This returns a 2D data structure with the columns as names and the rows as
    # data points. The names are trip_id, stop_id, precipitation, route_id, 
    # to_stop_arrival_datetime (ISO 8601 format), stop_name, delay_sec, 
    # sch_dur_sec, and act_dur_sec.
    # We will be plotting the data using matplotlib.
    # First just a histogram of the delay_sec column to see the distribution of
    # delays.
    # We need to mod the delay_sec column by 86400 to get the actual delay since
    # some of the delays overflow into the next day. There are negative delays
    # that shouldn't be moded, so we should only mod negative delays that are 
    # less than -43200 (half a day).

    data['delay_sec'] = ((data['delay_sec'] + 43200) % 86400) - 43200
    min_val = -2000 #data['delay_sec'].min()
    max_val = 2000 #data['delay_sec'].max()

    # Now that delay_sec is cleaned up, we have our raw data.
    # We'll plot multiple histograms of the data to see how the distribution
    # changes when conditioning on different variables. For example, we can plot
    # the histogram of delays for different precipitation levels, or for 
    # different routes. This will help us understand how these factors affect 
    # the delay distribution.

    # Delay distribution raw: DONE
    # Delay distribution by precipitation level:
    # We need to bin the precipitation levels (inches per hour) into categories:
    data['precip_bin'] = pd.cut(data['precipitation'], bins=[-0.01, 0, 0.05, 0.09, np.inf], labels=['none', 'light rain', 'moderate rain', 'heavy rain'])
    # Rush hour vs non rush hour:
    # We can define rush hour as 6:30-9am and 3:30-6:30pm
    # We will bin the scheduled duration (sch_dur_sec) into rush hour and non 
    # rush hour based on the time of day.
    data['rush_hour'] = data['sch_dur_sec'].apply(lambda x: 'rush hour' if (x % 86400 >= 15.5 * 3600 and x % 86400 <= 18.5 * 3600) or (x % 86400 >= 6.5 * 3600 and x % 86400 <= 9 * 3600) else 'non rush hour')
    # Combine both plots into subplots within the same figure
    fig, axs = plt.subplots(4, 1, figsize=(10, 24))

    # First sub-plot: Overall delay distribution
    mean_delay = data['delay_sec'].mean()
    axs[0].hist(data['delay_sec'], bins=3000, density=True)
    axs[0].axvline(mean_delay, color='blue', linestyle='--', label=f'Mean: {mean_delay:.1f} seconds')
    axs[0].set_xlabel("Delay (seconds)")
    axs[0].set_ylabel("Density")
    axs[0].set_title("Distribution of MBTA Green Line Delays")
    axs[0].set_xlim(min_val - 100, max_val + 100)

    # Second sub-plot: Conditional distributions by rain type
    rain_types = data['precip_bin'].unique()
    # Assign a unique color for each rain type using a color map
    color_map = plt.cm.get_cmap('tab10')
    for rain_type in rain_types:
        subset = data[data['precip_bin'] == rain_type]
        mean_delay = subset['delay_sec'].mean()
        color_index = list(rain_types).index(rain_type) % color_map.N
        color = color_map(color_index)
        axs[1].hist(subset['delay_sec'], bins=300, density=True, alpha=0.5, label=f'Rain: {rain_type}')
        axs[1].axvline(mean_delay, color=color, linestyle='--', label=f'Mean ({rain_type})')
        axs[1].text(mean_delay, 0.01, f'{mean_delay:.1f}', rotation=90, verticalalignment='bottom', color=color)

    axs[1].set_title('Conditional Distributions of Delay_sec by Rain Type')
    axs[1].set_xlabel('Delay (seconds)')
    axs[1].set_ylabel('Density')
    axs[1].set_xlim(min_val - 100, max_val + 100)
    axs[1].legend()

    # Third sub-plot: Conditional distributions by rush hour vs non rush hour
    rush_hour_types = data['rush_hour'].unique()
    for rush_hour_type in rush_hour_types:
        subset = data[data['rush_hour'] == rush_hour_type]
        mean_delay = subset['delay_sec'].mean()
        # Assign a unique color for each rush hour type using the same color map
        color_index = list(rush_hour_types).index(rush_hour_type) % color_map.N
        color = color_map(color_index)
        axs[2].hist(subset['delay_sec'], bins=900, density=True, alpha=0.5, label=f'{rush_hour_type}', color=color)
        axs[2].axvline(mean_delay, color=color, linestyle='--', label=f'Mean ({rush_hour_type})')
        axs[2].text(mean_delay, 0.01, f'{mean_delay:.1f}', rotation=90, verticalalignment='bottom', color=color)

    axs[2].set_title('Rush Hour vs Non-Rush Hour Delays')
    axs[2].set_xlabel('Delay (seconds)')
    axs[2].set_ylabel('Density')
    axs[2].set_xlim(min_val - 100, max_val + 100)
    axs[2].legend()

    # Fourth sub_plot: Conditional distributions by route_id
    route_ids = data['route_id'].unique()
    for route_id in route_ids:
        subset = data[data['route_id'] == route_id]
        mean_delay = subset['delay_sec'].mean()
        # Assign a unique color for each route_id using the same color map
        color_index = list(route_ids).index(route_id) % color_map.N
        color = color_map(color_index)
        axs[3].hist(subset['delay_sec'], bins=900, density=True, alpha=0.5, label=f'Route: {route_id}', color=color)
        axs[3].axvline(mean_delay, color=color, linestyle='--', label=f'Mean ({route_id})')
        # axs[3].text(mean_delay, 0.01, f'{mean_delay:.1f}', rotation=90, verticalalignment='bottom', color=color)

    axs[3].set_title('Conditional Distributions of Delay_sec by Route ID')
    axs[3].set_xlabel('Delay (seconds)')
    axs[3].set_ylabel('Density')
    axs[3].set_xlim(min_val - 100, max_val + 100)
    axs[3].legend()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.9)
    plt.show()

    # Figure 2: Delay by route_id
    n_routes = len(route_ids)
    fig2, axs2 = plt.subplots(n_routes, 2, figsize=(10, 5 * n_routes), sharex=True)

    if n_routes == 1:
        axs2 = np.array([axs2])  # Ensure axs2 is always a 2D array for consistency

    for ax, route_id in zip(axs2, route_ids):
        subset = data[data['route_id'] == route_id]
        if subset.empty:
            continue

        mean_delay = subset['delay_sec'].mean()
        ax[0].hist(subset['delay_sec'], bins=900, density=True, alpha=0.7)
        ax[0].axvline(mean_delay, color='red', linestyle='--',
                      label=f'Mean: {mean_delay:.1f}')
        ax[0].set_title(f'Conditional Distribution of Delay_sec | Route ID = {route_id}')
        ax[0].set_ylabel('Density')
        ax[0].set_xlim(min_val - 100, max_val + 100)
        ax[0].legend()

    # Delay by route_id with rain conditioning
    for ax, route_id in zip(axs2[:, 1], route_ids):
        subset = data[data['route_id'] == route_id]
        if subset.empty:
            continue

        rain_types = subset['precip_bin'].unique()
        for rain_type in rain_types:
            rain_subset = subset[subset['precip_bin'] == rain_type]
            mean_delay = rain_subset['delay_sec'].mean()
            color_index = list(rain_types).index(rain_type) % color_map.N
            color = color_map(color_index)
            ax.hist(rain_subset['delay_sec'], bins=200, density=True, alpha=0.5, label=f'Rain: {rain_type}', color=color)
            ax.axvline(mean_delay, color=color, linestyle='--', label=f'Mean ({rain_type})')

        ax.set_title(f'Conditional Distribution of Delay_sec | Route ID = {route_id} | Rain Type')
        ax.set_ylabel('Density')
        ax.set_xlim(min_val - 100, max_val + 100)
        ax.legend()

    axs2[0][-1].set_xlabel('Delay (seconds)')
    axs2[1][-1].set_xlabel('Delay (seconds)')
    fig2.tight_layout()
    plt.subplots_adjust(hspace=0.9)
    plt.show()

    # Figure 3: Delay by route_id with season conditioning

def main():
    graph_data()

if __name__ == "__main__":
    main()