from flask import Blueprint, render_template, request, redirect, url_for, flash
import yfinance as yf
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import io
import base64

monte_carlo_bp = Blueprint('montecarlo', __name__)

def get_pdr(ticker):
    stock = yf.Ticker(ticker)
    historical_data = stock.history(period='max')
    
    if historical_data.empty:
        raise ValueError(f"No price data found for {ticker}. The stock might be delisted or the ticker symbol could be incorrect.")
    
    last_500_days = historical_data['Close'].iloc[-500:]
    pdr = np.log(last_500_days / last_500_days.shift(1)).dropna()
    return pdr

def get_drift_and_std_dev(pdr):
    adr = pdr.mean()
    std_dev = pdr.std()
    variance = std_dev ** 2
    drift = adr - (variance / 2)
    return drift, std_dev

def project_future_prices(ticker, num_simulations=500, num_days=500):
    stock = yf.Ticker(ticker)
    historical_data = stock.history(period='1d')
    
    if historical_data.empty:
        raise ValueError(f"No price data found for {ticker}. The stock might be delisted or the ticker symbol could be incorrect.")
    
    last_price = historical_data['Close'].iloc[-1]
    pdr = get_pdr(ticker)
    drift, std_dev = get_drift_and_std_dev(pdr)

    # Pre-generate random values for all simulations
    random_values = norm.ppf(np.random.rand(num_simulations, num_days)) * std_dev

    simulations = np.zeros((num_simulations, num_days))
    simulations[:, 0] = last_price

    for i in range(1, num_days):
        simulations[:, i] = simulations[:, i-1] * np.exp(drift + random_values[:, i-1])

    return simulations

def plot_simulations(simulations, ticker):
    plt.figure(figsize=(10, 6))

    # Plot all individual simulations with transparency
    for simulation in simulations:
        plt.plot(simulation, alpha=0.4)

    # Calculate and plot the most probable, most optimistic, and least optimistic lines
    mean_simulation = np.mean(simulations, axis=0)
    best_simulation = np.max(simulations, axis=0)
    worst_simulation = np.min(simulations, axis=0)

    last_price = simulations[0, 0]
    mean_pct_change = ((mean_simulation[-1] - last_price) / last_price) * 100
    best_pct_change = ((best_simulation[-1] - last_price) / last_price) * 100
    worst_pct_change = ((worst_simulation[-1] - last_price) / last_price) * 100
    
    plt.plot(mean_simulation, color='black', label=f'Most Probable (Mean): {mean_simulation[-1]:.2f} ({mean_pct_change:+.2f}%)')
    plt.plot(best_simulation, color='green', label=f'Most Optimistic: {best_simulation[-1]:.2f} ({best_pct_change:+.2f}%)')
    plt.plot(worst_simulation, color='red', label=f'Least Optimistic: {worst_simulation[-1]:.2f} ({worst_pct_change:+.2f}%)')

    # Add labels and legend
    plt.title(f'Monte Carlo Simulations for {ticker}')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend(loc='upper left')

    # Set x-axis limit to 500 days
    plt.xlim(0, 499)

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Encode plot to base64 string
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    return plot_data

@monte_carlo_bp.route('/montecarlo', methods=['GET', 'POST'])
def monte_carlo():
    if request.method == 'POST':
        ticker = request.form.get('ticker')
        if not ticker:
            flash('Please enter a stock ticker', 'error')
            return redirect(url_for('montecarlo.monte_carlo'))

        try:
            simulations = project_future_prices(ticker)
            plot_data = plot_simulations(simulations, ticker)
            return render_template('monte_carlo.html', ticker=ticker, plot_data=plot_data)
        except ValueError as e:
            flash(str(e), 'error')
            return redirect(url_for('montecarlo.monte_carlo'))

    return render_template('monte_carlo.html')
