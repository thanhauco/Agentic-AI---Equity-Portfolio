# Issue #1: ZeroDivisionError when normalizing portfolio weights

**Description:**
If the total weight of selected stocks is zero, the `PortfolioBuilder` crashes with a `ZeroDivisionError` during normalization.

**Steps to Reproduce:**

1. Run `build_portfolio` with a list of stocks that all have 0 weight.
2. Observe crash.

**Labels:** bug, high-priority
