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
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, gaussian_kde


def _normalize_component_weights(components):
    total_weight = sum(comp['global_weight'] for comp in components)
    if total_weight <= 0:
        return components

    for comp in components:
        comp['global_weight'] /= total_weight

    return components


def sample_from_mixture_model(model, n_samples=1, random_state=None):
    """
    Draw samples from a returned mixture model.

    model should be the dictionary returned by create_X_line_model() or
    create_X_line_modelK().
    """
    components = model.get('components', [])
    if not components:
        raise ValueError('Model has no components to sample from.')

    rng = np.random.default_rng(random_state)
    weights = np.array([comp['global_weight'] for comp in components], dtype=float)
    weights = weights / weights.sum()

    chosen_components = rng.choice(len(components), size=n_samples, p=weights)
    samples = np.empty(n_samples, dtype=float)

    for i, comp_idx in enumerate(chosen_components):
        comp = components[comp_idx]
        samples[i] = rng.normal(loc=comp['mu'], scale=comp['sigma'])

    return samples


def run_delay_simulation(model, n_steps=100, initial_delay=0.0, random_state=None):
    """
    Numerical simulation using a fitted mixture model.
    The sampled values are simulated delays, and their cumulative sum is also
    returned in case you want a running-delay simulation.
    """
    delay_samples = sample_from_mixture_model(
        model,
        n_samples=n_steps,
        random_state=random_state
    )

    cumulative_delays = initial_delay + np.cumsum(delay_samples)
    return {
        'samples': delay_samples,
        'increments': delay_samples,
        'cumulative_delays': cumulative_delays
    }


def plot_simulation_vs_real_data(model, simulation_result, bins=300):
    """
    Overlay a histogram of simulated delays with the histogram of the real data.
    """
    real_delays = np.asarray(model.get('observed_delays', []), dtype=float)
    simulated_delays = np.asarray(simulation_result.get('samples', []), dtype=float)

    if real_delays.size == 0:
        raise ValueError('Model does not contain observed_delays to compare against.')
    if simulated_delays.size == 0:
        raise ValueError('Simulation result does not contain sampled delays to plot.')

    plt.figure(figsize=(12, 7))
    plt.hist(real_delays, bins=bins, density=True, alpha=0.35,
             label='Real Data')
    plt.hist(simulated_delays, bins=bins, density=True, alpha=0.35,
             label='Simulated Data')

    if model.get('pdf') is not None and model.get('x_grid') is not None:
        plt.plot(model['x_grid'], model['pdf'], 'r-', lw=2.5,
                 label='Fitted Mixture PDF')

    plt.axvline(np.mean(real_delays), color='blue', linestyle='--', linewidth=2,
                label=f'Real Mean: {np.mean(real_delays):.1f}')
    plt.axvline(np.mean(simulated_delays), color='orange', linestyle='--', linewidth=2,
                label=f'Simulated Mean: {np.mean(simulated_delays):.1f}')

    route_id = model.get('route_id', 'Unknown Route')
    plt.xlabel('Delay (seconds)')
    plt.ylabel('Density')
    plt.title(f'{route_id}: Simulated Delay Histogram vs Real Data')
    plt.xlim(-2100, 2100)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

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
    data['to_stop_arrival_datetime'] = pd.to_datetime(data['to_stop_arrival_datetime'])
    data['hour'] = data['to_stop_arrival_datetime'].dt.hour + data['to_stop_arrival_datetime'].dt.minute / 60
    data['rush_hour'] = data['hour'].apply(lambda x: 'rush hour' if (6.5 <= x <= 9) or (15.5 <= x <= 18.5) else 'non rush hour')
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
#     plt.show()

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
#     plt.subplots_adjust(hspace=0.9)
#     plt.show()

    # Figure 3: Delay by route_id with season conditioning
    # We will define seasons based on the month of the to_stop_arrival_datetime
    # column:
    # Winter: December, January, February
    # Spring: March, April, May
    # Summer: June, July, August
    # Fall: September, October, November
    data['month'] = data['to_stop_arrival_datetime'].dt.month
    data['season'] = data['month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else 'Spring' if x in [3, 4, 5] else 'Summer' if x in [6, 7, 8] else 'Fall')
    fig3, axs3 = plt.subplots(n_routes, 1, figsize=(10, 5 * n_routes), sharex=True)
    for ax, route_id in zip(axs3, route_ids):
        subset = data[data['route_id'] == route_id]
        if subset.empty:
            continue

        seasons = subset['season'].unique()
        for season in seasons:
            season_subset = subset[subset['season'] == season]
            mean_delay = season_subset['delay_sec'].mean()
            color_index = list(seasons).index(season) % color_map.N
            color = color_map(color_index)
            ax.hist(season_subset['delay_sec'], bins=200, density=True, alpha=0.5, label=f'Season: {season}', color=color)
            ax.axvline(mean_delay, color=color, linestyle='--', label=f'Mean ({season})')

        ax.set_title(f'Conditional Distribution of Delay_sec | Route ID = {route_id} | Season')
        ax.set_ylabel('Density')
        ax.set_xlim(min_val - 100, max_val + 100)
        ax.legend()
        axs3[-1].set_xlabel('Delay (seconds)')
    fig3.tight_layout()
    plt.show()

def create_model():
    data = pd.read_csv("final_delays_with_dates.csv")

    # Clean delay data (same as graph_data)
    data['delay_sec'] = ((data['delay_sec'] + 43200) % 86400) - 43200

    # Extract 1D data for GMM
    X = data['delay_sec'].values.reshape(-1, 1)

    # Fit GMM (K = 2)
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X)

    # Extract parameters
    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    weights = gmm.weights_

    print("Learned Parameters:")
    for k in range(2):
        print(f"Component {k+1}:")
        print(f"  Mean = {means[k]:.2f}")
        print(f"  Std = {np.sqrt(variances[k]):.2f}")
        print(f"  Weight = {weights[k]:.3f}")

    # Component means (already from GMM)
    mu1, mu2 = means

    # Overall mixture mean
    mixture_mean = np.sum(weights * means)

    # Plot histogram + fitted PDF
    x = np.linspace(-2000, 2000, 1000)

    # Mixture PDF
    pdf = np.zeros_like(x)
    for k in range(2):
        pdf += weights[k] * norm.pdf(x, means[k], np.sqrt(variances[k]))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(data['delay_sec'], bins=3000, density=True, alpha=0.5, label='Data')

    # Plot total mixture
    plt.plot(x, pdf, 'r-', lw=2, label='Fitted GMM')

    # Plot individual components
    for k in range(2):
        comp_pdf = weights[k] * norm.pdf(x, means[k], np.sqrt(variances[k]))
        plt.plot(x, comp_pdf, '--', label=f'Component {k+1}')

    min_val = -2000
    max_val = 2000

    # Component means
    plt.axvline(mu1, color='orange', linestyle='--', linewidth=2,label=f'Component 1 Mean: {mu1:.1f}')
    plt.axvline(mu2, color='green', linestyle='--', linewidth=2, label=f'Component 2 Mean: {mu2:.1f}')

    # Overall mean
    plt.axvline(mixture_mean, color='red', linestyle='--', linewidth=2, label=f'Mixture Mean: {mixture_mean:.1f}')
    plt.xlabel("Delay (seconds)")
    plt.ylabel("Density")
    plt.title("Gaussian Mixture Model Fit to Delay Data")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(min_val - 100, max_val + 100)
    plt.show()

def create_deranged_model():
    data = pd.read_csv("final_delays_with_dates.csv")

    # Clean delay data
    data['delay_sec'] = ((data['delay_sec'] + 43200) % 86400) - 43200
    data = data.dropna(subset=['delay_sec', 'precipitation', 'route_id', 'to_stop_arrival_datetime'])

    # Build the conditioning variables you are actually using
    data['precip_bin'] = pd.cut(
        data['precipitation'],
        bins=[-0.01, 0, 0.05, 0.09, np.inf],
        labels=['none', 'light rain', 'moderate rain', 'heavy rain']
    )

    data['to_stop_arrival_datetime'] = pd.to_datetime(data['to_stop_arrival_datetime'])
    data['month'] = data['to_stop_arrival_datetime'].dt.month
    data['season'] = data['month'].apply(
        lambda x: 'Winter' if x in [12, 1, 2]
        else 'Spring' if x in [3, 4, 5]
        else 'Summer' if x in [6, 7, 8]
        else 'Fall'
    )

    precip_levels = ['none', 'light rain', 'moderate rain', 'heavy rain']
    season_levels = ['Winter', 'Spring', 'Summer', 'Fall']
    route_ids = sorted(data['route_id'].dropna().unique())

    total_n = len(data)

    # Combined mixture over all condition-specific GMMs
    x = np.linspace(-2000, 2000, 2000)
    combined_pdf = np.zeros_like(x)

    all_components = []

    for route_id in route_ids:
        for precip in precip_levels:
            for season in season_levels:
                subset = data[
                    (data['route_id'] == route_id) &
                    (data['precip_bin'] == precip) &
                    (data['season'] == season)
                ]['delay_sec'].dropna()

                n_subset = len(subset)
                if n_subset < 5:
                    continue

                X_subset = subset.values.reshape(-1, 1)

                try:
                    # Same model logic as create_model(), but on each condition subset
                    gmm = GaussianMixture(n_components=2, random_state=42)
                    gmm.fit(X_subset)

                    means = gmm.means_.flatten()
                    variances = gmm.covariances_.flatten()
                    weights = gmm.weights_

                    # Weight of this condition in the full dataset
                    condition_weight = n_subset / total_n

                    for k in range(2):
                        global_weight = condition_weight * weights[k]
                        mu = means[k]
                        sigma = np.sqrt(variances[k])

                        all_components.append({
                            'route_id': route_id,
                            'precip_bin': precip,
                            'season': season,
                            'condition_n': n_subset,
                            'condition_weight': condition_weight,
                            'local_component': k + 1,
                            'local_weight': weights[k],
                            'global_weight': global_weight,
                            'mu': mu,
                            'sigma': sigma
                        })

                        combined_pdf += global_weight * norm.pdf(x, mu, sigma)

                except Exception as e:
                    print(f"Skipping condition route={route_id}, rain={precip}, season={season}: {e}")

    if len(all_components) == 0:
        print("No valid condition-specific GMMs were fitted.")
        return

    # Normalize in case some subsets were skipped
    total_weight = sum(comp['global_weight'] for comp in all_components)
    if total_weight > 0:
        combined_pdf /= total_weight
        for comp in all_components:
            comp['global_weight'] /= total_weight

    # Overall mixture mean
    mixture_mean = sum(comp['global_weight'] * comp['mu'] for comp in all_components)

    # Real data KDE
    kde = gaussian_kde(data['delay_sec'])
    real_density = kde(x)

    # Plot
    plt.figure(figsize=(12, 7))
    plt.hist(data['delay_sec'], bins=3000, density=True, alpha=0.35, label='Data')
    plt.plot(x, real_density, 'b-', lw=2, label='Real Data KDE')
    plt.plot(x, combined_pdf, 'r-', lw=2, label='Combined 64-Condition GMM')

    # Overall mean
    plt.axvline(mixture_mean, color='red', linestyle='--', linewidth=2,
                label=f'Mixture Mean: {mixture_mean:.1f}')

    plt.xlabel("Delay (seconds)")
    plt.ylabel("Density")
    plt.title("Combined GMM from All Conditions")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(-2100, 2100)
    plt.show()

    # Print summary
    print(f"Number of fitted condition-specific components: {len(all_components)}")
    print(f"Overall mixture mean: {mixture_mean:.2f}")

    summary = pd.DataFrame(all_components).sort_values('global_weight', ascending=False)
    print(summary[['route_id', 'precip_bin', 'season',
                   'local_component', 'global_weight', 'mu', 'sigma']].head(20))

def create_X_line_model(X):
    data = pd.read_csv("final_delays_with_dates.csv")

    # Clean delay data
    data['delay_sec'] = ((data['delay_sec'] + 43200) % 86400) - 43200
    # data = data[(data['delay_sec'] > -2000) & (data['delay_sec'] < 2000)]
    # data['delay_sec'] = np.clip(data['delay_sec'], -1200, 1200)

    # Filter to X line only
    data = data[data['route_id'] == 'Green-' + X].copy()
    data = data.dropna(subset=['delay_sec', 'precipitation', 'to_stop_arrival_datetime'])

    # Condition variables
    data['precip_bin'] = pd.cut(
        data['precipitation'],
        bins=[-0.01, 0, 0.05, 0.09, np.inf],
        labels=['none', 'light rain', 'moderate rain', 'heavy rain']
    )

    data['to_stop_arrival_datetime'] = pd.to_datetime(data['to_stop_arrival_datetime'])
    data['month'] = data['to_stop_arrival_datetime'].dt.month
    data['season'] = data['month'].apply(
        lambda x: 'Winter' if x in [12, 1, 2]
        else 'Spring' if x in [3, 4, 5]
        else 'Summer' if x in [6, 7, 8]
        else 'Fall'
    )

    precip_levels = ['none', 'light rain', 'moderate rain', 'heavy rain']
    season_levels = ['Winter', 'Spring', 'Summer', 'Fall']

    total_n = len(data)
    if total_n < 2:
        print(f"Not enough {X}-line data to fit model.")
        return

    x = np.linspace(-2000, 2000, 2000)
    combined_pdf = np.zeros_like(x, dtype=float)

    components = []

    for precip in precip_levels:
        for season in season_levels:
            subset = data[
                (data['precip_bin'] == precip) &
                (data['season'] == season)
            ]['delay_sec'].dropna()

            n_subset = len(subset)
            if n_subset < 2:
                continue

            X_subset = subset.values.reshape(-1, 1)

            # If very small group, a 2-component GMM may be unstable
            if n_subset < 5:
                mu = subset.mean()
                sigma = max(subset.std(ddof=0), 1e-6)
                global_weight = n_subset / total_n

                components.append({
                    'precip_bin': precip,
                    'season': season,
                    'condition_n': n_subset,
                    'local_component': 1,
                    'local_weight': 1.0,
                    'global_weight': global_weight,
                    'mu': mu,
                    'sigma': sigma
                })

                combined_pdf += global_weight * norm.pdf(x, mu, sigma)
                continue

            try:
                gmm = GaussianMixture(n_components=3, random_state=42)
                gmm.fit(X_subset)

                means = gmm.means_.flatten()
                variances = gmm.covariances_.flatten()
                weights = gmm.weights_

                # Sort by mean so component labels are consistent
                order = np.argsort(means)
                means = means[order]
                variances = variances[order]
                weights = weights[order]

                condition_weight = n_subset / total_n

                for k in range(3):
                    mu = means[k]
                    sigma = max(np.sqrt(variances[k]), 1e-6)
                    local_weight = weights[k]
                    global_weight = condition_weight * local_weight

                    components.append({
                        'precip_bin': precip,
                        'season': season,
                        'condition_n': n_subset,
                        'local_component': k + 1,
                        'local_weight': local_weight,
                        'global_weight': global_weight,
                        'mu': mu,
                        'sigma': sigma
                    })

                    combined_pdf += global_weight * norm.pdf(x, mu, sigma)

            except Exception as e:
                print(f"Skipping ({precip}, {season}): {e}")

    if len(components) == 0:
        print("No valid condition-specific components were fitted.")
        return

    # Normalize if any conditions were skipped
    total_weight = sum(comp['global_weight'] for comp in components)
    if total_weight > 0:
        combined_pdf /= total_weight
        components = _normalize_component_weights(components)

    mixture_mean = sum(comp['global_weight'] * comp['mu'] for comp in components)

    # Real KDE
    if len(data['delay_sec']) >= 2:
        kde = gaussian_kde(data['delay_sec'])
        real_density = kde(x)
    else:
        real_density = None

    # summary_by_mean = pd.DataFrame(components).sort_values(['mu', 'precip_bin', 'season'])
    # print(summary_by_mean[['precip_bin', 'season', 'condition_n',
    #                        'local_component', 'local_weight',
    #                        'global_weight', 'mu', 'sigma']])

    # Plot
    plt.figure(figsize=(12, 7))

    plt.hist(data['delay_sec'], bins=300, density=True, alpha=0.3, label=f'{X} Line Data')

    if real_density is not None:
        plt.plot(x, real_density, 'b-', lw=2, label='Real KDE')

    # Plot individual components
    for i, comp in enumerate(components):
        comp_pdf = comp['global_weight'] * norm.pdf(x, comp['mu'], comp['sigma'])
        plt.plot(x, comp_pdf, '--', alpha=0.35)

    # Plot combined model
    plt.plot(x, combined_pdf, 'r-', lw=2.5, label='16-Condition, K=3-per-condition Model')

    plt.axvline(mixture_mean, color='red', linestyle='--', linewidth=2,
                label=f'Mixture Mean: {mixture_mean:.1f}')

    plt.xlabel("Delay (seconds)")
    plt.ylabel("Density")
    plt.title(f"{X} Line: 16 Conditions with K=3 GMM per Condition")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(-2100, 2100)
    plt.show()

    print(f"Condition-components used: {len(components)} (max 32)")
    print(f"Mixture mean: {mixture_mean:.2f}")

    summary_by_weight = pd.DataFrame(components).sort_values('global_weight', ascending=False)
    print(summary_by_weight[['precip_bin', 'season', 'condition_n',
                             'local_component', 'local_weight',
                             'global_weight', 'mu', 'sigma']])

    return {
        'model_type': 'conditional_line_gmm',
        'route_id': f'Green-{X}',
        'components': components,
        'mixture_mean': mixture_mean,
        'x_grid': x,
        'pdf': combined_pdf,
        'summary': summary_by_weight.reset_index(drop=True),
        'observed_delays': data['delay_sec'].to_numpy()
    }

def create_X_line_modelK(Y, K):
    data = pd.read_csv("final_delays_with_dates.csv")

    # Clean delay data
    data['delay_sec'] = ((data['delay_sec'] + 43200) % 86400) - 43200
    data = data[data['route_id'] == 'Green-' + Y].copy()
    data = data.dropna(subset=['delay_sec'])

    # Optional: keep the same viewing window as your other plots
    data = data[(data['delay_sec'] >= -2000) & (data['delay_sec'] <= 2000)]

    X = data['delay_sec'].values.reshape(-1, 1)

    if len(X) < K:
        print(f"Not enough data to fit a K={K} GMM.")
        return

    gmm = GaussianMixture(n_components=K, random_state=42)
    gmm.fit(X)

    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    weights = gmm.weights_

    # Sort by mean so component 1 is always the left one
    order = np.argsort(means)
    means = means[order]
    variances = variances[order]
    weights = weights[order]

    print("Learned Parameters:")
    for k in range(K):
        print(f"Component {k+1}:")
        print(f"  Mean = {means[k]:.2f}")
        print(f"  Std = {np.sqrt(variances[k]):.2f}")
        print(f"  Weight = {weights[k]:.3f}")

    mixture_mean = np.sum(weights * means)

    x = np.linspace(-2000, 2000, 2000)

    pdf = np.zeros_like(x, dtype=float)
    for k in range(K):
        pdf += weights[k] * norm.pdf(x, means[k], np.sqrt(variances[k]))

    kde = gaussian_kde(data['delay_sec'])
    real_density = kde(x)

    plt.figure(figsize=(12, 7))
    plt.hist(data['delay_sec'], bins=300, density=True, alpha=0.3, label='Data')
    plt.plot(x, real_density, 'b-', lw=2, label='Real KDE')
    plt.plot(x, pdf, 'r-', lw=2.5, label=f'K={K} GMM')

    for k in range(K):
        comp_pdf = weights[k] * norm.pdf(x, means[k], np.sqrt(variances[k]))
        plt.plot(x, comp_pdf, '--', lw=2, label=f'Component {k+1}')
    
    for k in range(K):
        plt.axvline(means[k], color=plt.cm.tab10(k), linestyle='--', linewidth=2,
                    label=f'Component {k+1} Mean: {means[k]:.1f}')
    plt.axvline(mixture_mean, color='red', linestyle='--', linewidth=2,
                label=f'Mixture Mean: {mixture_mean:.1f}')

    plt.xlabel("Delay (seconds)")
    plt.ylabel("Density")
    plt.title(f"Global K={K} Gaussian Mixture Model")
    plt.xlim(-2100, 2100)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

    components = []
    for k in range(K):
        components.append({
            'local_component': k + 1,
            'global_weight': weights[k],
            'mu': means[k],
            'sigma': max(np.sqrt(variances[k]), 1e-6)
        })

    summary = pd.DataFrame(components).sort_values('mu').reset_index(drop=True)

    return {
        'model_type': 'global_line_gmm',
        'route_id': f'Green-{Y}',
        'K': K,
        'components': components,
        'mixture_mean': mixture_mean,
        'x_grid': x,
        'pdf': pdf,
        'summary': summary,
        'observed_delays': data['delay_sec'].to_numpy()
    }

def main():
    # graph_data()
    # create_model()
    # create_deranged_model()

    conditional_model = create_X_line_model('E')
    global_model = create_X_line_modelK('E', 4)

    if conditional_model is not None:
        conditional_simulation = run_delay_simulation(
            conditional_model,
            n_steps=10000,
            initial_delay=0.0,
            random_state=42
        )
        print("\nConditional-model simulation results:")
        print("First 10 delay increments:", conditional_simulation['increments'][:10])
        print("First 10 cumulative delays:", conditional_simulation['cumulative_delays'][:10])
        plot_simulation_vs_real_data(conditional_model, conditional_simulation, bins=1000)

    if global_model is not None:
        global_simulation = run_delay_simulation(
            global_model,
            n_steps=10000,
            initial_delay=0.0,
            random_state=42
        )
        print("\nGlobal-model simulation results:")
        print("First 10 delay increments:", global_simulation['increments'][:10])
        print("First 10 cumulative delays:", global_simulation['cumulative_delays'][:10])
        plot_simulation_vs_real_data(global_model, global_simulation, bins=300)


if __name__ == "__main__":
    main()