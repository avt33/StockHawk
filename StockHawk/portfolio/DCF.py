from flask import Blueprint, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np

DCF_bp = Blueprint('DCF', __name__)

def calculate_growth_rate(values):
    growth_rates = []
    for i in range(1, len(values)):
        growth_rate = (values[i] - values[i - 1]) / values[i-1]
        growth_rates.append(growth_rate)
    average_growth_rate = np.mean(growth_rates)
    
    return average_growth_rate

def future_FCF(ticker):
    stock = yf.Ticker(ticker)
    income_statement = stock.financials
    cash_flow = stock.cash_flow

    historical_revenue = income_statement.loc['Total Revenue'].iloc[:4]
    growth_rate = calculate_growth_rate(historical_revenue[::-1])
    
    current = cash_flow.loc['Free Cash Flow'].iloc[0]
    future_fcf = []
    for i in range(5):
        current = current * (1 + growth_rate)
        future_fcf.append(current)
    
    return future_fcf

def get_tax_rate(ticker):
    stock = yf.Ticker(ticker)

    income_statement = stock.financials
    
    tp_data = income_statement.loc['Tax Provision'].iloc[:3]
    pti_data = income_statement.loc['Pretax Income'].iloc[:3]
    tax_rate = ((tp_data.iloc[2] / pti_data.iloc[2]) + (tp_data.iloc[1] / pti_data.iloc[1]) + (tp_data.iloc[0] / pti_data.iloc[0])) / 3
    
    return tax_rate

def get_WACC(ticker):
    stock = yf.Ticker(ticker)
    stock_info = stock.info
    balance_sheet = stock.balance_sheet
    income_statement = stock.financials
    cash_flow = stock.cash_flow

    market_return = 0.08
    beta = stock_info.get('beta')
    treasure = 0.04083
    debt_value = balance_sheet.loc['Total Debt'].iloc[0]

    # getting cost of debt
    tp_data = income_statement.loc['Tax Provision'].iloc[:3]
    pti_data = income_statement.loc['Pretax Income'].iloc[:3]
    
    cod = ((tp_data.iloc[2] / pti_data.iloc[2]) + (tp_data.iloc[1] / pti_data.iloc[1]) + (tp_data.iloc[0] / pti_data.iloc[0])) / 3

    # getting enterprise value
    share_price = stock_info.get('currentPrice')
    outstanding_shares = stock_info.get('sharesOutstanding')
    equity_value = share_price * outstanding_shares

    tax_rate = get_tax_rate(ticker)

    cost_of_equity = treasure + beta * (market_return - treasure)
    EDE = equity_value / (equity_value + debt_value)
    DDE = debt_value / (equity_value + debt_value)
    WACC = EDE * cost_of_equity + DDE * cod * (1 - tax_rate)

    return WACC

def get_multiple(ticker):
    stock = yf.Ticker(ticker)
    stock_info = stock.info
    income_statement = stock.financials
    balance_sheet = stock.balance_sheet

    # getting enterprise value
    market_cap = stock_info.get('marketCap')
    total_debt = balance_sheet.loc['Total Debt'].iloc[0]
    cash_and_equivalents = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0]
    enterprise_value = market_cap + total_debt - cash_and_equivalents

    #calcualtions
    ebitda = income_statement.loc['EBITDA'].iloc[0]
    multiple = enterprise_value / ebitda

    return multiple

def future_EBITDA(ticker):
    # getting stock
    stock = yf.Ticker(ticker)

    # getting stock financials
    income_statement = stock.financials

    # past growth rates
    ebitda_data = income_statement.loc['EBITDA'].iloc[:4]
    historical_revenue = income_statement.loc['Total Revenue'].iloc[:4]
    growth_rate = calculate_growth_rate(historical_revenue[::-1])

    # future ebit
    current = ebitda_data.iloc[0]
    future_ebitda = []
    for i in range(5):
        current = current * (1 + growth_rate)
        future_ebitda.append(current)

    return future_ebitda

def get_terminal_value(ticker):
    stock = yf.Ticker(ticker)

    gdp = 0.03
    WACC = get_WACC(ticker)
    ebitda = future_EBITDA(ticker)
    multiple = get_multiple(ticker)

    terminal_value = ebitda[4] * multiple
    
    return terminal_value

def get_discounting(ticker):
    stock = yf.Ticker(ticker)

    fcf = future_FCF(ticker)
    WACC = get_WACC(ticker)
    terminal_value = get_terminal_value(ticker)

    discounting = []
    for i in range(5):
        value = 1 / ((1 + WACC) ** (i+1))
        discounting.append(value)
    
    enter_value = (discounting[0] * fcf[0]) + (discounting[1] * fcf[1]) + (discounting[1] * fcf[1]) + (discounting[2] * fcf[2]) + (discounting[3] * fcf[3]) + (discounting[4] * fcf[4]) + (discounting[4] * terminal_value)
    return enter_value

def enter_to_eq(ticker):
    stock = yf.Ticker(ticker)
    stock_info = stock.info
    balance_sheet = stock.balance_sheet

    cash_and_cash_equiv = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0]
    EnterV = get_discounting(ticker)
    debt_value = balance_sheet.loc['Total Debt'].iloc[0]

    finEqval = EnterV + cash_and_cash_equiv - debt_value
    
    outstanding_shares = stock_info.get('sharesOutstanding')

    shareprice = (finEqval / outstanding_shares)

    return shareprice

def get_terminal_value_percentage(ticker):
    terminal_value = get_terminal_value(ticker)
    WACC = get_WACC(ticker)
    PVTV = terminal_value / ((1 + WACC) ** 5)
    enter_value = get_discounting(ticker)
    terminal_value_percentage = (PVTV / enter_value) * 100

    return terminal_value_percentage

def get_eps(ticker):
    stock = yf.Ticker(ticker)
    income_statement = stock.financials
    stock_info = stock.info

    net_income = income_statement.loc['Net Income'].iloc[0]
    outstanding_shares = stock_info.get('sharesOutstanding')

    eps = net_income / outstanding_shares
    return eps

def get_discounting_data(ticker):
    stock = yf.Ticker(ticker)

    fcf = future_FCF(ticker)
    WACC = get_WACC(ticker)
    terminal_value = get_terminal_value(ticker)

    discounting = []
    for i in range(5):
        value = 1 / ((1 + WACC) ** (i+1))
        discounting.append(float(value))
    
    return discounting

def get_stock_price_and_intrinsic_value(ticker):
    stock = yf.Ticker(ticker)
    stock_info = stock.info
    stock_data = yf.Ticker(ticker).history(period="1y")
    cash_flow = stock.cash_flow

    high_prices = stock_data['High']
    low_prices = stock_data['Low']
    price_range = [high_prices.min(), low_prices.max()]

    beta = stock_info.get('beta')

    eps = get_eps(ticker)

    market_price = stock_info['currentPrice']
    eps = stock_info['trailingEps']
    pe_ratio = market_price / eps

    multiple = get_multiple(ticker)

    fcf = cash_flow.loc['Free Cash Flow'].iloc[0]

    terminal_value = get_terminal_value(ticker)

    WACC = get_WACC(ticker)

    terminal_value_percentage = get_terminal_value_percentage(ticker)

    current_price = stock_info.get('currentPrice')
    intrinsic_value = enter_to_eq(ticker)

    future_fcf_data = future_FCF(ticker)
    future_ebitda_data = future_EBITDA(ticker)
    discounting_data = get_discounting(ticker)

    extra_values = {
        'Beta': beta,
        'EPS': eps,
        'P/E Ratio': pe_ratio,
        'EV/EBITDA Multiple': multiple,
        'Free Cash Flow': fcf,
        'Weighted Average Cost of Capital': WACC,
        'Terminal Value Percentage': terminal_value_percentage
    }


    return current_price, intrinsic_value, price_range, beta, eps, pe_ratio, multiple, fcf, terminal_value, WACC, terminal_value_percentage, extra_values, future_fcf_data, future_ebitda_data, discounting_data


@DCF_bp.route('/DCF', methods=['GET', 'POST'])
def DCF():
    results = None
    error_message = None

    if request.method == 'POST':
        ticker = request.form['ticker']
        try:
            current_price, intrinsic_value, price_range, beta, eps, pe_ratio, multiple, fcf, terminal_value, WACC, terminal_value_percentage, extra_values, future_fcf_data, future_ebitda_data, discounting_data = get_stock_price_and_intrinsic_value(ticker)
            if intrinsic_value < 0:
                error_message = "Couldn't calculate properly"
            else:
                results = {
                    'current_price': round(current_price, 2),
                    'intrinsic_value': round(intrinsic_value, 2),
                    'price_range': [round(price_range[0], 2), round(price_range[1], 2)],
                    'beta': round(beta, 2),
                    'eps': round(eps, 2),
                    'pe_ratio': round(pe_ratio, 2),
                    'multiple': round(multiple, 2),
                    'fcf': round((fcf / 1000000),2),
                    'terminal_value': round((terminal_value / 1000000), 2),
                    'WACC': round((WACC * 100), 2),
                    'terminal_value_percentage': round(terminal_value_percentage, 2),
                    'extra_values': extra_values,
                    'future_fcf_data': future_FCF(ticker),
                    'future_ebitda_data': future_EBITDA(ticker),
                    'discounting_data': get_discounting(ticker)
                }
        except Exception as e:
            error_message = "Error: couldn't compute"
    return render_template('DCF.html', results=results, error_message=error_message)