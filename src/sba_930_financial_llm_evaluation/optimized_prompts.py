
# Optimized prompts for financial analysis and risk assessment

OPTIMIZED_FORECAST_PROMPTS = {
    "stock_market": (
        "You are a financial analyst. "
        "Inflation and interest rates are rising. "
        "Explain two specific ways these conditions "
        "could affect stock prices over the next six months. "
        "Mention borrowing costs and company profits. "
        "Keep the response brief and factual."
    ),

    "commodity_prices": (
        "You are a financial analyst. "
        "Explain how inflation, supply disruptions, "
        "and market demand could influence commodity prices "
        "over the next six months. "
        "Provide a brief explanation of each factor."
    ),

    "inflation_forecast": (
        "You are an economic analyst. "
        "Inflation increased from 2 percent to 5 percent "
        "over the previous year. "
        "Explain how interest rates, consumer demand, "
        "and energy prices could influence inflation "
        "over the next six months."
    ),
}

OPTIMIZED_RISK_PROMPTS = {
    "market_volatility": (
        "You are a financial risk analyst. "
        "Stock prices have fallen 15 percent in one month. "
        "Identify the main financial risk. "
        "Explain why the risk matters and recommend "
        "one practical mitigation strategy."
    ),

    "credit_risk": (
        "You are a financial risk analyst. "
        "A company has high debt, declining revenue, "
        "and difficulty making loan payments. "
        "Identify the main financial risk and recommend "
        "one specific strategy to reduce it."
    ),

    "liquidity_risk": (
        "You are a financial risk analyst. "
        "A financial institution cannot meet short-term "
        "withdrawal requests because it lacks cash. "
        "Identify the main risk and recommend one "
        "practical strategy to improve liquidity."
    ),
}

if __name__ == "__main__":
    print("Optimized Financial Forecasting Prompts:")

    for task, prompt in OPTIMIZED_FORECAST_PROMPTS.items():
        print(f"\n{task}: {prompt}")

    print("\nOptimized Risk Analysis Prompts:")

    for task, prompt in OPTIMIZED_RISK_PROMPTS.items():
        print(f"\n{task}: {prompt}")

