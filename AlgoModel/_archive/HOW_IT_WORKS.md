# How it works

1. Hold SPY paid in cash.
2. Each quarter, score candidate pairs with Gatev distance on the prior year of prices only.
3. If the spread is extreme, **tomorrow** buy the lagging stock (no short).
4. Fund buys by selling some SPY; keep a minimum SPY fraction.
5. Exit when the spread normalizes; park cash back in SPY.

Optional: only step 3 when SPY is above its 200-day moving average (drawdown control).
