## Sample script for removing variables based on detefcted Multicollinearity

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

# Sample dataframe
np.random.seed(0)
X1 = np.random.randn(100)
X2 = 2 * X1 + np.random.randn(100)  # X2 is highly correlated with X1
X3 = -1 * X1 + np.random.randn(100)  # X3 is highly correlated with X1 and X2

data = {"X1": X1, "X2": X2, "X3": X3}
df = pd.DataFrame(data)
features = df.columns.tolist()

# Define the maximum acceptable VIF
max_vif = 5.0

# Iteratively remove features with the highest VIF (looking for VIF above `max_vif`)
while True:
    X = sm.add_constant(df[features])
    vifs = pd.Series(
        [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        index=X.columns,
        dtype="float64",
    )
    print("VIF:")
    print(vifs)

    vifs = vifs.drop("const")
    max_vif_feature, max_vif_value = vifs.idxmax(), vifs.max()
    if max_vif_value <= max_vif:
        break

    features.remove(max_vif_feature)
    print(f"Removed {max_vif_feature} due to high VIF value: {max_vif_value:.2f}")

# The final dataframe with the remaining features
df_final = df[features]

print("Remaining Features:")
print(df_final.columns.tolist())
