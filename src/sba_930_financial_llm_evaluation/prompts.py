
# Financial forecasting prompts
# These will be used to evaluate both LLMs.

FORECASTING_PROMPTS = {
    "stock_market": (
        "Financial analysis task: "
        "The stock market has experienced rising inflation "
        "and increasing interest rates. "
        "Explain how these economic conditions may affect "
        "stock market performance over the next six months."
    ),

    "commodity_prices": (
        "Financial analysis task: "
        "Historical commodity prices have increased during "
        "periods of inflation. "
        "Explain what factors could influence commodity "
        "prices over the next six months."
    ),

    "inflation_forecast": (
        "Financial forecasting task: "
        "Inflation increased from 2 percent to 5 percent "
        "over the previous year. "
        "Explain what factors could cause inflation "
        "to rise or fall over the next six months."
    ),
}


if __name__ == "__main__":
    for task_name, prompt in FORECASTING_PROMPTS.items():
        print(f"\nTask: {task_name}")
        print(f"Prompt: {prompt}")
